"""Full local MacBERT fine-tuning with group-isolated semantic corpus and frozen test evaluation."""

from contextlib import nullcontext
import math
from pathlib import Path
import random
import numpy as np
from ..provenance import canonical_json, fingerprint, file_hash
from .corpus import prepare_corpus
from .dataset import ScoreDataset, Collator
from .evaluate import mean_baseline, median_baseline, report
from .checkpoint import device_for, save_checkpoint, load_checkpoint


def validate_config(config):
    # PSEUDOCODE: reject invalid training hyperparameters and require a pinned model revision.
    required = {'model_id', 'revision', 'seed', 'max_length', 'batch_size', 'gradient_accumulation',
                'epochs', 'patience', 'encoder_lr', 'head_lr', 'weight_decay', 'warmup_ratio',
                'huber_delta', 'dropout', 'max_grad_norm', 'threads', 'device'}
    if set(config) != required or config['model_id'] != 'hfl/chinese-macbert-base':
        raise ValueError('Unsupported training configuration.')
    if len(config['revision']) != 40 or any(c not in '0123456789abcdef' for c in config['revision']):
        raise ValueError('Model revision must be an immutable commit.')
    for key in ('seed', 'max_length', 'batch_size', 'gradient_accumulation', 'epochs', 'patience', 'threads'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError('Invalid integer: ' + key)
    for key in ('encoder_lr', 'head_lr', 'huber_delta', 'max_grad_norm'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError('Invalid positive parameter: ' + key)
    if any(type(config[k]) not in (int, float) or not math.isfinite(config[k]) for k in ('dropout', 'warmup_ratio', 'weight_decay')) or not 8 <= config['max_length'] <= 512 or not 0 <= config['dropout'] < 1 or not 0 <= config['warmup_ratio'] < 1 or config['weight_decay'] < 0:
        raise ValueError('Invalid token limit, regularization or warmup.')


def _loader(rows, tokenizer, config, shuffle=False):
    # PSEUDOCODE: tokenize without truncation -> pad each batch -> use an explicit shuffle generator.
    import torch
    from torch.utils.data import DataLoader
    return DataLoader(ScoreDataset(rows, tokenizer, config['max_length']), batch_size=config['batch_size'],
                      shuffle=shuffle, num_workers=0, collate_fn=Collator(tokenizer),
                      generator=torch.Generator().manual_seed(config['seed']))


def predict_rows(model, tokenizer, rows, config, device):
    # PSEUDOCODE: run deterministic evaluation in original row order and preserve continuous outputs.
    import torch
    from .schema import CATEGORIES
    model.eval(); predictions = []
    with torch.inference_mode():
        for batch in _loader(rows, tokenizer, config):
            outputs = model(**{k: v.to(device) for k, v in batch.items() if k not in ('labels', 'categories')})
            for i, category in enumerate(batch['categories']):
                predictions.append(dict(zip(CATEGORIES[category][1], (outputs[category][i].float().cpu() * 100).tolist())))
    return predictions


def train(rows, base_path, output_path, config, *, allow_constructed_training=False):
    # PSEUDOCODE: freeze reviewed groups and local model -> train using validation only -> evaluate locked test once.
    import json
    import torch
    from transformers import BertTokenizer, get_linear_schedule_with_warmup
    from .model import ScoringModel, regression_loss
    validate_config(config)
    partitions, corpus = prepare_corpus(rows, allow_constructed_training=allow_constructed_training)
    base_path, output_path = Path(base_path), Path(output_path)
    receipt = json.loads((base_path / 'download.json').read_text(encoding='utf-8'))
    if receipt['model_id'] != config['model_id'] or receipt['revision'] != config['revision'] or not receipt.get('files'):
        raise ValueError('Base model download receipt differs from training config.')
    actual = {p.relative_to(base_path).as_posix() for p in base_path.rglob('*') if p.is_file() and p.name != 'download.json'}
    if actual != set(receipt['files']) or not {'config.json', 'vocab.txt'} <= actual or not any(n.endswith(('.safetensors', '.bin')) for n in actual):
        raise ValueError('Base model receipt must cover the complete model and tokenizer inventory.')
    for name, digest in receipt['files'].items():
        source = (base_path / name).resolve()
        if not source.is_relative_to(base_path.resolve()) or file_hash(source) != digest:
            raise ValueError('Base model hash mismatch.')
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    torch.set_num_threads(config['threads'])
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    tokenizer = BertTokenizer.from_pretrained(base_path, local_files_only=True)
    for items in partitions.values():
        ScoreDataset(items, tokenizer, config['max_length'])
    output_path.mkdir(parents=True, exist_ok=False)
    device = device_for(config['device'])
    model = ScoringModel.pretrained(base_path, config['dropout']).to(device)
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': config['encoder_lr']},
                                  {'params': model.heads.parameters(), 'lr': config['head_lr']}], weight_decay=config['weight_decay'])
    loader = _loader(partitions['train'], tokenizer, config, True)
    total_steps = math.ceil(len(loader) / config['gradient_accumulation']) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config['warmup_ratio']), total_steps)
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and dtype == torch.float16)
    baseline = mean_baseline(partitions['train'])
    median_reference = median_baseline(partitions['train'])
    best, stale, history, best_path = float('inf'), 0, [], None
    for epoch in range(config['epochs']):
        model.train(); optimizer.zero_grad(set_to_none=True); loss_sum = 0.; count = 0
        for index, batch in enumerate(loader):
            group_start = index // config['gradient_accumulation'] * config['gradient_accumulation']
            group_samples = min(len(partitions['train']) - group_start * config['batch_size'], config['batch_size'] * config['gradient_accumulation'])
            n = len(batch['categories'])
            context = torch.autocast('cuda', dtype=dtype) if device.type == 'cuda' else nullcontext()
            with context:
                outputs = model(**{k: v.to(device) for k, v in batch.items() if k not in ('labels', 'categories')})
                loss = regression_loss(outputs, batch['labels'].to(device), batch['categories'], config['huber_delta'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss; no final model published.')
            scaler.scale(loss * n / group_samples).backward()
            loss_sum += float(loss.detach()) * n; count += n
            if (index + 1) % config['gradient_accumulation'] == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'], error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        validation = report(partitions['validation'], predict_rows(model, tokenizer, partitions['validation'], config, device), baseline, median_reference)
        metric = validation['macro_category_mae']
        history.append({'epoch': epoch + 1, 'training_loss': loss_sum / count, 'validation': validation})
        if metric < best:
            best, stale = metric, 0
            best_path = save_checkpoint(output_path / f'epoch-{epoch + 1}', model, tokenizer, config,
                {'run_id': output_path.name, 'dataset': corpus, 'mean_baseline': baseline, 'median_baseline': median_reference, 'epoch': epoch + 1,
                 'selection_metric': 'validation.macro_category_mae', 'test_used_for_selection': False})
        else:
            stale += 1
        (output_path / 'history.json').write_text(canonical_json(history), encoding='utf-8')
        if stale >= config['patience']:
            break
    model, tokenizer, manifest, identity = load_checkpoint(best_path)
    model.to(device)
    predictions = predict_rows(model, tokenizer, partitions['test'], config, device)
    final = {'best_checkpoint': str(best_path), 'identity': identity, 'corpus': corpus,
             'test': report(partitions['test'], predictions, baseline, median_reference), 'history': history}
    (output_path / 'result.json').write_text(canonical_json(final), encoding='utf-8')
    return final
