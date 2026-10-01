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
from .checkpoint import save_checkpoint, load_checkpoint, load_tokenizer
from .evidence import calibrate_evidence

from .config import validate_config
from .weights import check_base as _check_base


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
                keys = CATEGORIES[category][1]
                predictions.append({'scores': dict(zip(keys, (outputs[category][index].float().cpu() * 100).tolist())),
                    'evidence': dict(zip(keys, torch.sigmoid(outputs['_evidence'][category][index].float()).cpu().tolist()))})
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
                                         config['huber_delta'], reduction='none', evidence_weight=config['evidence_loss_weight'])
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


def _train(partitions, corpus, base_path, base_id, output_path, config, runtime, *, development_only=False):
    # PSEUDOCODE: initialize shared inputs -> shard training -> select on validation -> test the chosen model once.
    import torch
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup
    from .model import ScoringModel
    from .runtime import ShardedBatches
    signature = fingerprint({'corpus': corpus['id'], 'base': base_id, 'config': config, 'development_only': development_only})
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
    tokenizer = load_tokenizer(base_path)
    datasets = {name: ScoreDataset(rows, tokenizer, config['max_length']) for name, rows in partitions.items()}
    runtime.primary(lambda: output_path.mkdir(parents=True, exist_ok=False))
    execution = runtime.describe()
    runtime.primary(lambda: (output_path / 'execution.json').write_text(canonical_json(execution), encoding='utf-8'))
    core = ScoringModel.pretrained(base_path, config['dropout'], config=config, device=runtime.device, dtype=runtime.dtype)
    if config['gradient_checkpointing']:
        core.encoder.config.use_cache = False
        core.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW([{'params': [p for p in core.encoder.parameters() if p.requires_grad], 'lr': config['encoder_lr']},
                                  {'params': list(core.heads.parameters()) + list(core.evidence_heads.parameters()), 'lr': config['head_lr']}],
                                  weight_decay=config['weight_decay'], betas=(config['adam_beta1'], config['adam_beta2']), eps=config['adam_epsilon'])
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
        if not progress['optimizer_steps']:
            raise ValueError('Every optimizer step overflowed; no trained checkpoint can be selected.')
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
    if development_only:
        result = {'best_checkpoint': best_path, 'corpus': corpus, 'execution': execution, 'history': history,
                  'validation_mae': best, 'test': None, 'status': 'development_only_uncalibrated'}
        runtime.primary(lambda: (output_path / 'result.json').write_text(canonical_json(result), encoding='utf-8'))
        return result

    def finish():
        # PSEUDOCODE: reload the selected checkpoint -> evaluate untouched test rows -> save one final report.
        selected, selected_tokenizer, manifest, _ = load_checkpoint(best_path, base_path=base_path, device=runtime.device, dtype=runtime.dtype)
        calibration_rows = partitions.get('calibration', [])
        calibration_predictions = predict_rows(selected, selected_tokenizer, calibration_rows, config, runtime.device) if calibration_rows else []
        evidence = calibrate_evidence(calibration_rows, calibration_predictions,
            selection_rows=partitions['validation'],
            selection_predictions=predict_rows(selected, selected_tokenizer, partitions['validation'], config, runtime.device),
            target_precision=config['evidence_precision'], alpha=config['evidence_alpha'])
        metadata = {k: v for k, v in manifest.items() if k not in ('files', 'status', 'storage', 'contract', 'config', 'purpose')}
        final_path = save_checkpoint(output_path / 'model', selected, selected_tokenizer, config, {**metadata, 'evidence_calibration': evidence})
        identity = file_hash(final_path / 'manifest.json')
        predictions = predict_rows(selected, selected_tokenizer, partitions['test'], config, runtime.device)
        final = {'best_checkpoint': str(final_path), 'identity': identity, 'corpus': corpus, 'execution': execution,
                 'evidence_calibration': evidence, 'test': report(partitions['test'], predictions, baseline, median_reference, evidence), 'history': history}
        (output_path / 'result.json').write_text(canonical_json(final), encoding='utf-8')
        points = {'identity': identity, 'rows': [{'example_id': row['example_id'], 'truth': row['scores'], **prediction}
                  for row, prediction in zip(partitions['test'], predictions)]}
        (output_path / 'test-predictions.json').write_text(canonical_json(points), encoding='utf-8')
        return final

    return runtime.primary(finish)


def train(rows, base_path, output_path, config, *, development_only=False):
    # PSEUDOCODE: require reviewed real-source corpus and verified base files before starting training.
    from .runtime import TrainingRuntime
    config = validate_config(config)
    partitions, corpus = prepare_corpus(rows, require_calibration=config['model_id'] == 'Qwen/Qwen3-8B', development_only=development_only)
    base_path, output_path = Path(base_path).resolve(), Path(output_path).resolve()
    base_id = _check_base(base_path, config)
    with TrainingRuntime(config) as runtime:
        return _train(partitions, corpus, base_path, base_id, output_path, config, runtime, development_only=development_only)
