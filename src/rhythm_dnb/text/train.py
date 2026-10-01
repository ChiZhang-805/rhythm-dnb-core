"""Source-backed text training with single-device or distributed execution and locked evaluation."""

from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import numpy as np
from ..provenance import canonical_json, fingerprint, file_hash
from .corpus import prepare_corpus
from .dataset import ScoreDataset, Collator
from .evaluate import mean_baseline, median_baseline, report
from .checkpoint import save_checkpoint, load_checkpoint

RUNTIME_DEFAULTS = {'precision': 'auto', 'gradient_checkpointing': True, 'num_workers': 0}


def validate_config(config):
    # PSEUDOCODE: normalize hardware options -> reject unknown fields and invalid training choices.
    config = {**RUNTIME_DEFAULTS, **config}
    required = {'model_id', 'revision', 'seed', 'max_length', 'batch_size', 'gradient_accumulation',
                'epochs', 'patience', 'encoder_lr', 'head_lr', 'weight_decay',
                'warmup_ratio', 'huber_delta', 'dropout', 'max_grad_norm', 'threads', 'device'} | set(RUNTIME_DEFAULTS)
    if set(config) != required or config['model_id'] != 'hfl/chinese-macbert-base':
        raise ValueError('Unsupported training configuration.')
    revision = config['revision']
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Base model revision must be an immutable commit.')
    for key in ('seed', 'max_length', 'batch_size', 'gradient_accumulation', 'epochs', 'patience', 'threads'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError('Invalid integer: ' + key)
    if config['seed'] >= 2 ** 32:
        raise ValueError('Training seed must fit the NumPy random-state range.')
    if type(config['num_workers']) is not int or config['num_workers'] < 0 or type(config['gradient_checkpointing']) is not bool:
        raise ValueError('Invalid data-loader/checkpointing settings.')
    if config['device'] not in ('auto', 'cpu', 'cuda') or config['precision'] not in ('auto', 'fp32', 'fp16', 'bf16'):
        raise ValueError('Unsupported device or precision.')
    for key in ('encoder_lr', 'head_lr', 'huber_delta', 'max_grad_norm'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError('Invalid positive parameter: ' + key)
    if any(type(config[k]) not in (int, float) or not math.isfinite(config[k]) for k in ('dropout', 'warmup_ratio', 'weight_decay')) or not 8 <= config['max_length'] <= 512 or not 0 <= config['dropout'] < 1 or not 0 <= config['warmup_ratio'] < 1 or config['weight_decay'] < 0:
        raise ValueError('Invalid token limit, regularization or warmup.')
    return config


def _check_base(base_path, config):
    # PSEUDOCODE: verify every local encoder/tokenizer file against the pinned acquisition receipt.
    receipt = json.loads((base_path / 'download.json').read_text(encoding='utf-8'))
    if receipt['model_id'] != config['model_id'] or receipt['revision'] != config['revision'] or not receipt.get('files'):
        raise ValueError('Base model receipt differs from training config.')
    actual = {p.relative_to(base_path).as_posix() for p in base_path.rglob('*') if p.is_file() and p.name != 'download.json'}
    if actual != set(receipt['files']) or not {'config.json', 'vocab.txt'} <= actual or not any(n.endswith(('.safetensors', '.bin')) for n in actual):
        raise ValueError('Base receipt must cover the complete model and tokenizer inventory.')
    for name, digest in receipt['files'].items():
        source = (base_path / name).resolve()
        if not source.is_relative_to(base_path.resolve()) or file_hash(source) != digest:
            raise ValueError('Base model hash mismatch.')
    return fingerprint(receipt)


def _inputs(batch, device):
    # PSEUDOCODE: move only encoder inputs, keeping labels, category names and weights separate.
    return {k: v.to(device, non_blocking=device.type == 'cuda') for k, v in batch.items()
            if k not in ('labels', 'categories', 'sample_weights')}


def predict_rows(model, tokenizer, rows, config, device):
    # PSEUDOCODE: evaluate every original row exactly once, without distributed padding or truncation.
    import torch
    from torch.utils.data import DataLoader
    from .schema import CATEGORIES
    loader = DataLoader(ScoreDataset(rows, tokenizer, config['max_length']), batch_size=config['batch_size'],
                        shuffle=False, collate_fn=Collator(tokenizer), num_workers=0)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            outputs = model(**_inputs(batch, device))
            for index, category in enumerate(batch['categories']):
                predictions.append(dict(zip(CATEGORIES[category][1], (outputs[category][index].float().cpu() * 100).tolist())))
    return predictions


def _epoch(model, loader, optimizer, scheduler, scaler, runtime, config, sample_count):
    # PSEUDOCODE: accumulate exact global sample means -> synchronize final microbatch -> advance successful steps.
    import torch
    from .model import regression_loss
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = count = 0.
    updates = skipped = 0
    accumulation = config['gradient_accumulation']
    width = config['batch_size'] * runtime.world_size
    for index, batch in enumerate(loader):
        last = (index + 1) % accumulation == 0 or index + 1 == len(loader)
        group_start = index // accumulation * accumulation
        global_samples = min(sample_count - group_start * width, width * accumulation)
        context = model.no_sync() if runtime.world_size > 1 and not last else nullcontext()
        with context:
            with runtime.autocast():
                outputs = model(**_inputs(batch, runtime.device))
                losses = regression_loss(outputs, batch['labels'].to(runtime.device), batch['categories'],
                                         config['huber_delta'], reduction='none')
                weights = batch['sample_weights'].to(runtime.device)
                weighted = (losses * weights).sum()
                # DDP averages process gradients; undo it before dividing by the actual global sample count.
                loss = weighted * runtime.world_size / global_samples
            if runtime.sum([int(not torch.isfinite(loss))])[0]:
                raise ValueError('Nonfinite training loss on at least one process.')
            scaler.scale(loss).backward()
        loss_sum += float(weighted.detach())
        count += float(weights.sum())
        if last:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'], error_if_nonfinite=not scaler.is_enabled())
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
                updates += 1
            else:
                skipped += 1  # Overflow retries must not advance the learning-rate schedule.
            optimizer.zero_grad(set_to_none=True)
    total_loss, total_count = runtime.sum([loss_sum, count])
    if int(total_count) != sample_count:
        raise RuntimeError('Distributed epoch omitted or duplicated real training samples.')
    return {'training_loss': total_loss / total_count, 'training_samples': int(total_count),
            'optimizer_steps': updates, 'overflow_skipped_steps': skipped}


def _train(partitions, corpus, base_path, base_id, output_path, config, runtime):
    # PSEUDOCODE: initialize shared inputs -> shard training -> select on validation -> test the chosen model once.
    import torch
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from transformers import BertTokenizer, get_linear_schedule_with_warmup
    from .model import ScoringModel
    from .runtime import ShardedBatches
    signature = fingerprint({'corpus': corpus['id'], 'base': base_id, 'config': config})
    if len(set(runtime.gather(signature))) != 1:
        raise ValueError('Processes received different corpora, base files or training settings.')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if runtime.device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    torch.set_num_threads(config['threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tokenizer = BertTokenizer.from_pretrained(base_path, local_files_only=True)
    datasets = {name: ScoreDataset(rows, tokenizer, config['max_length']) for name, rows in partitions.items()}
    runtime.primary(lambda: output_path.mkdir(parents=True, exist_ok=False))
    execution = runtime.describe()
    runtime.primary(lambda: (output_path / 'execution.json').write_text(canonical_json(execution), encoding='utf-8'))
    core = ScoringModel.pretrained(base_path, config['dropout']).to(runtime.device)
    if config['gradient_checkpointing']:
        core.encoder.config.use_cache = False
        core.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW([{'params': [p for p in core.encoder.parameters() if p.requires_grad], 'lr': config['encoder_lr']},
                                  {'params': core.heads.parameters(), 'lr': config['head_lr']}], weight_decay=config['weight_decay'])
    model = DistributedDataParallel(core, device_ids=[runtime.device.index] if runtime.device.type == 'cuda' else None,
                                    broadcast_buffers=False) if runtime.world_size > 1 else core
    microbatches = math.ceil(len(datasets['train']) / (config['batch_size'] * runtime.world_size))
    total_steps = math.ceil(microbatches / config['gradient_accumulation']) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config['warmup_ratio']), total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=runtime.precision == 'fp16')
    baseline, median_reference = mean_baseline(partitions['train']), median_baseline(partitions['train'])
    best, stale, history, best_path = float('inf'), 0, [], None
    for epoch in range(config['epochs']):
        sampler = ShardedBatches(len(datasets['train']), config['batch_size'], runtime.rank, runtime.world_size, config['seed'], epoch)
        options = {'multiprocessing_context': 'spawn', 'persistent_workers': True} if config['num_workers'] else {}
        loader = DataLoader(datasets['train'], batch_sampler=sampler, collate_fn=Collator(tokenizer),
                            num_workers=config['num_workers'], pin_memory=runtime.device.type == 'cuda', **options)
        progress = _epoch(model, loader, optimizer, scheduler, scaler, runtime, config, len(datasets['train']))
        validation = runtime.primary(lambda: report(partitions['validation'],
            predict_rows(core, tokenizer, partitions['validation'], config, runtime.device), baseline, median_reference))
        metric = validation['macro_category_mae']
        if not math.isfinite(metric):
            raise ValueError('Validation metric is not finite.')
        history.append({'epoch': epoch + 1, **progress, 'validation': validation})
        if metric < best:
            best, stale = metric, 0
            best_path = runtime.primary(lambda: str(save_checkpoint(output_path / f'epoch-{epoch + 1}', core, tokenizer, config,
                {'run_id': output_path.name, 'dataset': corpus, 'base_id': base_id, 'execution': execution,
                 'mean_baseline': baseline, 'median_baseline': median_reference, 'epoch': epoch + 1,
                 'selection_metric': 'validation.macro_category_mae', 'test_used_for_selection': False})))
        else:
            stale += 1
        runtime.primary(lambda: (output_path / 'history.json').write_text(canonical_json(history), encoding='utf-8'))
        if stale >= config['patience']:
            break
    del loader, model, core, optimizer, scheduler, scaler
    if runtime.device.type == 'cuda':
        torch.cuda.empty_cache()

    def finish():
        # PSEUDOCODE: reload the selected checkpoint -> evaluate untouched test rows -> save one final report.
        selected, selected_tokenizer, _, identity = load_checkpoint(best_path)
        selected.to(runtime.device)
        predictions = predict_rows(selected, selected_tokenizer, partitions['test'], config, runtime.device)
        final = {'best_checkpoint': best_path, 'identity': identity, 'corpus': corpus, 'execution': execution,
                 'test': report(partitions['test'], predictions, baseline, median_reference), 'history': history}
        (output_path / 'result.json').write_text(canonical_json(final), encoding='utf-8')
        return final

    return runtime.primary(finish)


def train(rows, base_path, output_path, config):
    # PSEUDOCODE: require reviewed real-source corpus and verified base files before starting training.
    from .runtime import TrainingRuntime
    config = validate_config(config)
    partitions, corpus = prepare_corpus(rows)
    base_path, output_path = Path(base_path).resolve(), Path(output_path).resolve()
    base_id = _check_base(base_path, config)
    with TrainingRuntime(config) as runtime:
        return _train(partitions, corpus, base_path, base_id, output_path, config, runtime)
