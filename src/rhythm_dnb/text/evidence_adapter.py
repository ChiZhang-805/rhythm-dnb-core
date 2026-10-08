"""Train a separate context-sensitive evidence adapter; never replace or certify intensity scores."""

import json
import math
from pathlib import Path
import random
import time
import numpy as np
from ..provenance import canonical_json, fingerprint
from .schema import CATEGORIES, METRICS
from .labels import evidence_target
from .features import validate_feature_rows
from .guard import binary_metrics, choose_empirical_threshold, selective_metrics


def evidence_report(rows, predictions, *, thresholds=None, target_precision=None, threshold_boundary='observed'):
    # PSEUDOCODE: evaluate explicit evidence labels per metric; select thresholds only when explicitly requested.
    if len(rows) != len(predictions) or not rows or (thresholds is None) == (target_precision is None):
        raise ValueError('Aligned evidence rows need either validation selection or fixed thresholds.')
    heads, families = {}, {}
    for metric, *_ in METRICS:
        pairs = [(row, prediction['evidence'][metric], evidence_target(row, metric))
                 for row, prediction in zip(rows, predictions) if metric in row['scores'] and evidence_target(row, metric) is not None]
        if not pairs:
            heads[metric] = {'n': 0, 'threshold': thresholds[metric] if thresholds is not None else None}
            continue
        labels = [p[2] for p in pairs]; probabilities = [p[1] for p in pairs]
        threshold = thresholds[metric] if thresholds is not None else choose_empirical_threshold(labels, probabilities, target_precision, boundary=threshold_boundary)
        heads[metric] = {**binary_metrics(labels, probabilities), 'threshold': threshold,
                        **selective_metrics(labels, probabilities, threshold)}
        for row, probability, label in pairs:
            family = row.get('family_id', row['group_id'])
            accepted = threshold is not None and probability >= threshold
            current = families.setdefault(family, {'n': 0, 'positive': 0, 'false_acceptances': 0, 'missed_supported': 0})
            current['n'] += 1; current['positive'] += int(label)
            current['false_acceptances'] += int(accepted and not label)
            current['missed_supported'] += int(not accepted and label)
    available = [h for h in heads.values() if h['n']]
    if not available:
        raise ValueError('Evidence report has no explicit known labels.')
    positives, negatives = sum(h['positive'] for h in available), sum(h['negative'] for h in available)
    accepted_supported = sum(h['accepted_supported'] for h in available)
    false_acceptances = sum(h['false_acceptances'] for h in available)
    return {'heads': heads, 'families': families, 'macro_log_loss': float(np.mean([h['log_loss'] for h in available])),
            'macro_brier': float(np.mean([h['brier'] for h in available])),
            'false_acceptances': false_acceptances,
            'missed_supported': sum(h['missed_supported'] for h in available),
            'supported_recall': accepted_supported / positives if positives else None,
            'false_acceptance_rate': false_acceptances / negatives if negatives else None,
            'macro_supported_recall': float(np.mean([h['supported_recall'] for h in available if h['positive']])) if positives else None,
            'heads_with_no_supported_acceptance': [k for k, h in heads.items() if h.get('positive', 0) and not h['accepted_supported']],
            'heads_without_both_reference_classes': [k for k, h in heads.items() if not h.get('positive') or not h.get('negative')],
            'automatic_model_promotion': False,
            'evidence_calibrated': False, 'independent_human_gold': False}


def prepare_evidence(payload):
    # PSEUDOCODE: seal train/validation only; forbid numerical labels and require both evidence classes for all metrics.
    rows = payload['rows']
    validate_feature_rows(rows)
    if payload.get('id') != fingerprint(rows) or payload.get('independent_human_gold') is not False:
        raise ValueError('Experimental evidence corpus must have a verified identity and explicit non-gold status.')
    if {r['split'] for r in rows} != {'train', 'validation'}:
        raise ValueError('Evidence training accepts train and validation only; never load a sealed test.')
    if any(any(v is not None for v in r['scores'].values()) or r.get('scope_targets') for r in rows):
        raise ValueError('Evidence-only learning forbids numerical and scope targets.')
    if any(not any(evidence_target(r, k) is not None for k in r['scores']) for r in rows):
        raise ValueError('Every training row needs explicit evidence supervision.')
    partitions = {role: [r for r in rows if r['split'] == role] for role in ('train', 'validation')}
    counts = {}
    for role, selected in partitions.items():
        counts[role] = {}
        for metric, *_ in METRICS:
            labels = [evidence_target(r, metric) for r in selected if metric in r['scores']]
            labels = [v for v in labels if v is not None]
            if set(labels) != {0., 1.}:
                raise ValueError(role + ': both evidence classes required for ' + metric)
            counts[role][metric] = {'positive': int(sum(labels)), 'negative': len(labels) - int(sum(labels))}
    from .review import evidence_support_coverage
    corpus = {'id': payload['id'], 'initial_checkpoint_id': payload['initial_checkpoint_id'], 'counts': counts,
              'joint_evidence_coverage': evidence_support_coverage(rows),
              'development_groups': {key: sorted({r[key] for r in rows if r.get(key)})
                                     for key in ('group_id', 'participant_id', 'family_id')},
              'development_text_hashes': sorted({fingerprint(''.join(r['text'].split())) for r in rows}),
              'independent_human_gold': False, 'test_loaded': False,
              'provenance': payload.get('provenance', {})}
    return partitions, corpus


def _continuation_evidence(checkpoint, source_payload, partitions, parent, base_id, config):
    # PSEUDOCODE: bind a prior evidence adapter and its exact development data; prohibit old gradient-training rows in new validation.
    previous, identity = _inspect_evidence(checkpoint, parent)
    old_partitions, old_corpus = prepare_evidence(source_payload)
    if old_corpus['id'] != previous['dataset']['id'] or old_corpus['initial_checkpoint_id'] != parent or previous['base_id'] != base_id:
        raise ValueError('Evidence continuation source corpus or base differs from the checkpoint.')
    for key in ('model_id', 'revision', 'quantization', 'lora_rank', 'lora_alpha', 'dropout'):
        if previous['config'][key] != config[key]:
            raise ValueError('Evidence continuation architecture mismatch: ' + key)
    ancestors = previous.get('evidence_initialization') or {}
    groups = {key: sorted({r[key] for r in old_partitions['train'] if r.get(key)} | set(ancestors.get('training_groups', {}).get(key, [])))
              for key in ('group_id', 'participant_id', 'family_id')}
    texts = sorted({fingerprint(''.join(r['text'].split())) for r in old_partitions['train']} | set(ancestors.get('training_text_hashes', [])))
    for key, values in groups.items():
        if set(values) & {r[key] for r in partitions['validation'] if r.get(key)}:
            raise ValueError('Evidence continuation validation overlaps prior gradient-training groups: ' + key)
    if set(texts) & {fingerprint(''.join(r['text'].split())) for r in partitions['validation']}:
        raise ValueError('Evidence continuation validation text was used for gradient training.')
    return {'checkpoint_id': identity, 'corpus_id': old_corpus['id'], 'optimizer_reset': True,
        'training_groups': groups, 'training_text_hashes': texts,
        'development_groups': {key: sorted(set(values) | set(ancestors.get('development_groups', {}).get(key, [])))
                               for key, values in old_corpus['development_groups'].items()},
        'development_text_hashes': sorted(set(old_corpus['development_text_hashes']) | set(ancestors.get('development_text_hashes', []))),
        'previous_validation_may_be_reused_for_development': True}


def train_evidence(payload, checkpoint, base, output_dir, config, *, initialize_evidence=None, initial_evidence_corpus=None):
    # PSEUDOCODE: validate evidence-only protocol -> continue a separate encoder adapter -> select solely on validation loss.
    import torch
    from torch.utils.data import DataLoader
    from torch.nn.parallel import DistributedDataParallel
    from transformers import get_linear_schedule_with_warmup
    from .config import validate_config
    from .checkpoint import load_checkpoint, save_checkpoint, inspect_checkpoint
    from .continuation import validate_initialization
    from .runtime import TrainingRuntime, ShardedBatches
    from .dataset import ScoreDataset, Collator
    from .train import _epoch, predict_rows
    from .resume import implementation_identity
    config = validate_config(config)
    if config['scope_loss_weight'] or config['evidence_loss_weight'] != 1. or not config['train_experimental_evidence']:
        raise ValueError('Evidence-only protocol requires evidence weight 1, no scope loss and explicit experimental evidence training.')
    partitions, corpus = prepare_evidence(payload)
    previous, identity = inspect_checkpoint(checkpoint, allow_experimental=True)
    parent = validate_initialization(checkpoint, partitions, corpus, previous['base_id'], config, True)
    if (initialize_evidence is None) != (initial_evidence_corpus is None):
        raise ValueError('Evidence continuation needs both checkpoint and its exact original development corpus.')
    continuation = _continuation_evidence(initialize_evidence, initial_evidence_corpus, partitions,
        identity, previous['base_id'], config) if initialize_evidence is not None else None
    if previous['config'].get('scope_loss_weight', 0):
        raise ValueError('Evidence-only initialization requires a scoring model without scope heads.')
    output = Path(output_dir).resolve()
    with TrainingRuntime(config) as runtime:
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        if runtime.device.type == 'cuda':
            torch.cuda.manual_seed_all(config['seed'])
        torch.set_num_threads(config['threads'])
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        execution = {**runtime.describe(), 'training_implementation': implementation_identity(),
                     'evidence_implementation': fingerprint(Path(__file__).read_text(encoding='utf-8'))}
        signature = fingerprint({'corpus': corpus['id'], 'config': config, 'parent': identity, 'evidence_initialization': continuation})
        if len(set(runtime.gather(signature))) != 1:
            raise ValueError('Distributed evidence training inputs differ.')
        runtime.primary(lambda: output.mkdir(parents=True, exist_ok=False))
        core, tokenizer, _, loaded_id = load_checkpoint(initialize_evidence or checkpoint, base_path=base, device=runtime.device,
            dtype=runtime.dtype, allow_experimental=True, allow_evidence=continuation is not None, trainable=True)
        if loaded_id != (continuation['checkpoint_id'] if continuation is not None else identity):
            raise ValueError('Parent checkpoint changed before training.')
        core.heads.requires_grad_(False)
        frozen = {k: v.detach().cpu().clone() for k, v in core.heads.state_dict().items()}
        if continuation is None:
            with torch.no_grad():
                for category, (_, keys) in CATEGORIES.items():
                    head = core.evidence_heads[category]; head.weight.zero_()
                    head.bias.copy_(head.bias.new_tensor([math.log(corpus['counts']['train'][k]['positive'] /
                                                                  corpus['counts']['train'][k]['negative']) for k in keys]))
        if config['gradient_checkpointing']:
            core.encoder.config.use_cache = False
            core.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        dataset = ScoreDataset(partitions['train'], tokenizer, config['max_length'], explicit_evidence_only=True)
        optimizer = torch.optim.AdamW([
            {'params': [p for p in core.encoder.parameters() if p.requires_grad], 'lr': config['encoder_lr']},
            {'params': core.evidence_heads.parameters(), 'lr': config['head_lr']}], weight_decay=config['weight_decay'],
            betas=(config['adam_beta1'], config['adam_beta2']), eps=config['adam_epsilon'])
        steps = math.ceil(math.ceil(len(dataset)/(config['batch_size']*runtime.world_size))/config['gradient_accumulation'])*config['epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer, int(steps*config['warmup_ratio']), steps)
        scaler = torch.amp.GradScaler('cuda', enabled=runtime.precision == 'fp16')
        model = DistributedDataParallel(core, device_ids=[runtime.device.index] if runtime.device.type == 'cuda' else None,
                                        broadcast_buffers=False) if runtime.world_size > 1 else core
        from collections import Counter
        counts = Counter(r['category'] for r in partitions['train'])
        weights = {c: len(dataset)/(len(counts)*n) for c, n in counts.items()} if config['balance_categories'] else None
        protocol = {'config': config, 'dataset': corpus, 'parent_scoring_checkpoint_id': identity,
            'evidence_initialization': continuation,
            'execution': execution, 'signature': signature, 'selection_metric': 'validation_macro_log_loss',
            'threshold_selection': 'validation_supported_recall_then_fewer_false_acceptances_with_gap_midpoint', 'evidence_calibrated': False,
            'score_model_modified': False, 'numeric_outputs_from_this_adapter_valid': False,
            'head_initialization': 'preserved previous evidence heads' if continuation else 'zero weights; training evidence class log odds',
            'category_loss_weights': weights}
        runtime.primary(lambda: (output/'protocol.json').write_text(canonical_json(protocol), encoding='utf-8'))
        best, stale, history, best_path = math.inf, 0, [], None
        if continuation is not None:
            initial_validation = runtime.primary(lambda: evidence_report(partitions['validation'],
                predict_rows(core, tokenizer, partitions['validation'], config, runtime.device), target_precision=config['evidence_precision'],
                threshold_boundary='validation_gap_midpoint'))
            best = initial_validation['macro_log_loss']
            best_path = runtime.primary(lambda: str(save_checkpoint(output/'initial-model', core, tokenizer, config,
                {**protocol, 'run_id': output.name, 'base_id': previous['base_id'], 'epoch': 0,
                 'initialization': parent, 'validation': initial_validation, 'evidence_trained': True,
                 'thresholds': {k: v['threshold'] for k, v in initial_validation['heads'].items()}}, purpose='experimental_evidence')))
            history.append({'epoch': 0, 'validation': initial_validation, 'optimizer_steps': 0,
                            'role': 'initial_evidence_reselected_on_current_validation'})
        for epoch in range(config['epochs']):
            started = time.monotonic()
            loader = DataLoader(dataset, batch_sampler=ShardedBatches(len(dataset), config['batch_size'], runtime.rank,
                runtime.world_size, config['seed'], epoch), collate_fn=Collator(tokenizer), num_workers=config['num_workers'])
            progress = _epoch(model, loader, optimizer, scheduler, scaler, runtime, config, len(dataset), category_weights=weights)
            if not progress['optimizer_steps']:
                raise ValueError('No successful optimizer updates in evidence training.')
            validation = runtime.primary(lambda: evidence_report(partitions['validation'],
                predict_rows(core, tokenizer, partitions['validation'], config, runtime.device), target_precision=config['evidence_precision'],
                threshold_boundary='validation_gap_midpoint'))
            history.append({'epoch': epoch+1, **progress, 'validation': validation, 'seconds': time.monotonic()-started})
            if any(not torch.equal(value, frozen[key]) for key, value in ((k, v.detach().cpu()) for k, v in core.heads.state_dict().items())):
                raise RuntimeError('Frozen intensity heads changed during evidence-only learning.')
            if validation['macro_log_loss'] < best:
                best, stale = validation['macro_log_loss'], 0
                best_path = runtime.primary(lambda: str(save_checkpoint(output/f'epoch-{epoch+1}', core, tokenizer, config,
                    {**protocol, 'run_id': output.name, 'base_id': previous['base_id'], 'epoch': epoch+1,
                     'initialization': parent, 'validation': validation, 'evidence_trained': True,
                     'thresholds': {k: v['threshold'] for k, v in validation['heads'].items()}}, purpose='experimental_evidence')))
            else:
                stale += 1
            runtime.primary(lambda: (output/'history.json').write_text(canonical_json(history), encoding='utf-8'))
            if runtime.rank == 0:
                print(canonical_json({'epoch': epoch+1, 'validation_log_loss': validation['macro_log_loss'],
                                      'seconds': history[-1]['seconds']}), flush=True)
            if stale >= config['patience']:
                break
        result = {'best_checkpoint': best_path, 'validation_log_loss': best, 'history': history,
                  'parent_scoring_checkpoint_id': identity, 'evidence_calibrated': False}
        runtime.primary(lambda: (output/'result.json').write_text(canonical_json(result), encoding='utf-8'))
        return result


def refine_evidence_thresholds(payload, checkpoint, base, output_dir, *, device='cuda'):
    # PSEUDOCODE: retain original validation recall while removing dominated cutoffs -> center the gap -> copy unchanged weights with lineage.
    import shutil
    from .checkpoint import load_checkpoint
    from .train import predict_rows
    partitions, corpus = prepare_evidence(payload)
    manifest, identity = _inspect_evidence(checkpoint)
    if corpus['id'] != manifest['dataset']['id'] or payload['initial_checkpoint_id'] != manifest['parent_scoring_checkpoint_id']:
        raise ValueError('Threshold refinement requires the exact original development corpus and scorer.')
    model, tokenizer, loaded, loaded_id = load_checkpoint(checkpoint, base_path=base, device=device, allow_evidence=True)
    if identity != loaded_id or manifest != loaded:
        raise ValueError('Evidence checkpoint changed during threshold refinement.')
    predictions = predict_rows(model, tokenizer, partitions['validation'], manifest['config'], next(model.heads.parameters()).device)
    previous = evidence_report(partitions['validation'], predictions, thresholds=manifest['thresholds'])
    updated = evidence_report(partitions['validation'], predictions, target_precision=manifest['config']['evidence_precision'],
                              threshold_boundary='validation_gap_midpoint')
    if any(previous['heads'][k]['accepted_supported'] != updated['heads'][k]['accepted_supported'] or
           previous['heads'][k]['false_acceptances'] < updated['heads'][k]['false_acceptances'] for k in previous['heads']):
        raise ValueError('Threshold refinement must preserve supported validation recall without adding false acceptances.')
    destination = Path(output_dir).resolve(); source = Path(checkpoint).resolve()
    if destination.is_relative_to(source):
        raise ValueError('Threshold refinement output must be outside the original checkpoint.')
    destination.mkdir(parents=True, exist_ok=False)
    for name in manifest['files']:
        path = destination/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, path)
    refinement = {'source_checkpoint_id': identity, 'validation_id': fingerprint(partitions['validation']),
        'boundary': 'validation_gap_midpoint', 'test_used': False, 'weights_changed': False,
        'selection_objective': 'maximum_supported_recall_then_minimum_false_acceptances_at_target_precision',
        'validation_supported_recall_preserved': True,
        'validation_false_acceptances_removed': previous['false_acceptances'] - updated['false_acceptances'],
        'validation_decisions_changed': previous['false_acceptances'] != updated['false_acceptances'],
        'evidence_calibrated': False}
    (destination/'manifest.json').write_text(canonical_json({**manifest, 'threshold_refinement': refinement,
        'threshold_selection': 'validation_supported_recall_then_fewer_false_acceptances_with_gap_midpoint', 'validation': updated,
        'thresholds': {k: v['threshold'] for k, v in updated['heads'].items()}}), encoding='utf-8')
    _, final_identity = _inspect_evidence(destination)
    return {**refinement, 'checkpoint': str(destination), 'checkpoint_id': final_identity}


def evaluate_evidence(payload, checkpoint, base, output_dir, *, device='cuda', regression_only=False):
    # PSEUDOCODE: reject exposed fit cases -> evaluate fixed adapter/thresholds -> preserve per-example evidence diagnostics.
    from .checkpoint import load_checkpoint
    from .train import predict_rows
    rows = payload['rows']; validate_feature_rows(rows)
    if payload.get('id') != fingerprint(rows) or {r['split'] for r in rows} != {'test'}:
        raise ValueError('Evaluation requires a sealed separate test corpus.')
    manifest, _ = _inspect_evidence(checkpoint)
    exposed = manifest['dataset']
    ancestors = manifest.get('initialization') or {}
    evidence_ancestors = manifest.get('evidence_initialization') or {}
    for key, values in exposed['development_groups'].items():
        if (set(values) | set(ancestors.get('exposure_groups', {}).get(key, [])) |
                set(evidence_ancestors.get('development_groups', {}).get(key, []))) & {r[key] for r in rows if r.get(key)}:
            raise ValueError('Evidence test overlaps fitted development groups: ' + key)
    if (set(exposed['development_text_hashes']) | set(ancestors.get('exposure_text_hashes', [])) |
            set(evidence_ancestors.get('development_text_hashes', []))) & {fingerprint(''.join(r['text'].split())) for r in rows}:
        raise ValueError('Evidence test text was used for fitting/selection.')
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    model, tokenizer, loaded_manifest, identity = load_checkpoint(checkpoint, base_path=base, device=device, allow_evidence=True)
    if manifest != loaded_manifest:
        raise ValueError('Evidence checkpoint changed during evaluation.')
    predictions = predict_rows(model, tokenizer, rows, manifest['config'], next(model.heads.parameters()).device)
    report = {**evidence_report(rows, predictions, thresholds=manifest['thresholds']), 'checkpoint_id': identity,
              'corpus_id': payload['id'], 'role': 'development_regression' if regression_only else 'authored_heldout_diagnostic'}
    (output/'report.json').write_text(canonical_json(report), encoding='utf-8')
    (output/'predictions.json').write_text(canonical_json([{'example_id': r['example_id'], 'evidence': p['evidence']}
        for r, p in zip(rows, predictions)]), encoding='utf-8')
    return report


def _inspect_evidence(checkpoint, parent_id=None):
    # PSEUDOCODE: require dedicated evidence purpose and explicit parent binding before allocating a second encoder.
    from .checkpoint import inspect_checkpoint
    manifest, identity = inspect_checkpoint(checkpoint, allow_evidence=True)
    if manifest['purpose'] != 'experimental_evidence' or (parent_id is not None and manifest.get('parent_scoring_checkpoint_id') != parent_id):
        raise ValueError('Evidence adapter must match the exact scoring checkpoint.')
    thresholds = manifest.get('thresholds', {})
    if set(thresholds) != {m[0] for m in METRICS} or any(v is not None and
        (type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1) for v in thresholds.values()):
        raise ValueError('Evidence adapter has invalid metric thresholds.')
    return manifest, identity


class EvidenceAdapter:
    def __init__(self, checkpoint, parent_id, base, device):
        # PSEUDOCODE: bind a separate experimental adapter to immutable intensity weights and defer device allocation.
        self.manifest, self.identity = _inspect_evidence(checkpoint, parent_id)
        self.checkpoint, self.base, self.device = checkpoint, base, device
        self.model = None

    def predict(self, category, text):
        # PSEUDOCODE: compute evidence only with the separate adapter; never use its altered encoder to score intensity.
        import torch
        from .checkpoint import load_checkpoint, device_for
        from .inference import resolve_precision, inference_profile
        from .dataset import encode
        if self.model is None:
            target = device_for(self.device) if isinstance(self.device, str) else self.device
            dtype = resolve_precision('auto', target, self.manifest['storage'])
            with torch.autocast(target.type, enabled=False):
                self.model, self.tokenizer, _, identity = load_checkpoint(self.checkpoint, base_path=self.base,
                    device=target, dtype=dtype, allow_evidence=True)
            if identity != self.identity:
                self.model = None
                raise ValueError('Evidence checkpoint changed after initialization.')
            self.model.eval()
            self.inference_dtype = dtype
        target = next(self.model.evidence_heads.parameters()).device
        encoded = encode(self.tokenizer, category, text, self.manifest['config']['max_length'])
        profile = inference_profile(self.identity, self.manifest, self.model, target, self.inference_dtype)
        with torch.inference_mode(), torch.autocast(target.type, enabled=False):
            outputs = self.model(**{k: torch.tensor([v], device=target) for k, v in encoded.items()})
            probabilities = torch.sigmoid(outputs['_evidence'][category][0].float()).cpu().tolist()
        values = dict(zip(CATEGORIES[category][1], probabilities))
        if any(not math.isfinite(v) for v in values.values()):
            raise ValueError('Evidence adapter returned nonfinite probabilities.')
        return {'evidence_estimates': values, 'evidence_adapter_identity': self.identity,
                'evidence_inference_profile': profile,
                'evidence_guard_kind': 'experimental_context_adapter', 'acceptance_certified': False,
                'experimental_acceptance': {k: self.manifest['thresholds'][k] is not None and v >= self.manifest['thresholds'][k]
                                            for k, v in values.items()}}
