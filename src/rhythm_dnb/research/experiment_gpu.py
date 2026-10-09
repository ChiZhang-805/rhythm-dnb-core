"""CPU preflight, isolated resumable GPU fitting, and content-addressed text inference."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

from ..provenance import file_hash, fingerprint, canonical_json
from ..text.experiment import prepare_development, train_experiment
from ..text.expansion import template_identity
from ..text.schema import CATEGORIES
from .experiment_data import load_packet, save_json, TEXT


def implementation_id():
    # PSEUDOCODE: bind experiment execution to all package source files, including mathematical kernels.
    root = Path(__file__).resolve().parents[1]
    return fingerprint({p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob('*.py'))})


def training_configs(protocol, config):
    # PSEUDOCODE: freeze two bounded learning-rate comparisons and keep the remaining optimization settings identical.
    from ..text.config import validate_config
    result = []
    for rate in protocol['text_learning_rates']:
        result.append(validate_config({**config, 'seed': protocol['seed'], 'encoder_lr': rate,
                       'head_lr': rate * 5, 'balance_categories': True, 'device': 'cuda',
                       'require_all_validation_metrics': True}))
    if not result or len({fingerprint(c) for c in result}) != len(result):
        raise ValueError('Training candidates must be distinct and nonempty.')
    return result


def inference_tasks(data):
    # PSEUDOCODE: strip outcomes and cohort roles -> deduplicate identical category/text inputs for stable singleton inference.
    tasks = {}
    for role in ('reference', 'train', 'validation', 'test'):
        for row in data[role]:
            for category, text in row['texts'].items():
                key = fingerprint([category, text])
                tasks[key] = {'category': category, 'text': text}
    return tasks


def preflight(packet, base, config, output):
    # PSEUDOCODE: verify pinned base and every token sequence on CPU -> publish a portable GPU plan without allocating a GPU.
    from ..text.weights import check_base
    from ..text.checkpoint import load_tokenizer
    from ..text.dataset import encode
    from ..text.review import training_readiness
    from .experiment import validate_protocol
    data, manifest = load_packet(packet)
    validate_protocol(data['protocol'])
    configs = training_configs(data['protocol'], config)
    base_id = check_base(base, configs[0])
    partitions, _ = prepare_development(data['text-development'])
    readiness = training_readiness(partitions, configs[0], experimental=True)
    if readiness['blockers']:
        raise ValueError('Training blocked: ' + '; '.join(readiness['blockers']))
    tokenizer = load_tokenizer(base)
    lengths, failures = [], []
    tasks = inference_tasks(data)
    inputs = [(r['example_id'], r['category'], r['text']) for rows in partitions.values() for r in rows]
    inputs += [(key, r['category'], r['text']) for key, r in tasks.items()]
    for identity, category, text in inputs:
        try:
            lengths.append(len(encode(tokenizer, category, text, configs[0]['max_length'])['input_ids']))
        except ValueError as error:
            failures.append({'id': identity, 'error': str(error)})
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    plan = {'packet_id': manifest['id'], 'base_id': base_id, 'implementation_id': implementation_id(),
            'configs': configs, 'readiness': readiness, 'max_tokens': max(lengths, default=0),
            'invalid_inputs': failures, 'unique_cohort_texts': len(tasks),
            'initialization': 'pinned_base_only', 'train_counts': {k: len(v) for k, v in partitions.items()},
            'estimated_max_optimizer_steps_per_candidate': math.ceil(len(partitions['train']) /
                (configs[0]['batch_size'] * configs[0]['gradient_accumulation'])) * configs[0]['epochs'],
            'status': 'ready_for_gpu' if not failures else 'blocked_by_invalid_text',
            'inference_precision': 'fp32', 'text_predictions_are_uncalibrated': True}
    save_json(output / 'gpu-plan.json', plan)
    return plan


def verify_new_checkpoint(checkpoint, data, packet_manifest):
    # PSEUDOCODE: reject old/continued checkpoints and check all DNB identities/templates against both fitted text roles.
    from ..text.checkpoint import inspect_checkpoint
    manifest, identity = inspect_checkpoint(checkpoint, allow_experimental=True)
    corpus = data['text-development']
    if (manifest.get('initialization') or manifest['dataset']['id'] != corpus['manifest']['id']
            or manifest['dataset'].get('initialization_required') != 'pinned_base_only'):
        raise ValueError('DNB evaluation requires this packet trained directly from the pinned base.')
    fitted_people = set(manifest['dataset']['development_groups'].get('participant_id', []))
    cohort_people = {'rhythm:' + r['participant_id'] for role in ('reference', 'train', 'validation', 'test') for r in data[role]}
    fitted_templates = {template_identity(r['text']) for r in corpus['rows']}
    cohort_templates = {template_identity(r['text']) for r in inference_tasks(data).values()}
    if fitted_people & cohort_people or fitted_templates & cohort_templates:
        raise ValueError('DNB cohort leaked into text fitting or validation.')
    return manifest, identity


def train_candidate(packet, base, plan_path, candidate, output, *, resume=None):
    # PSEUDOCODE: run only a specified candidate using development data and an optional verified epoch state.
    data, manifest = load_packet(packet, names={'protocol', 'text-development'})
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if plan['packet_id'] != manifest['id'] or plan['implementation_id'] != implementation_id() or plan['status'] != 'ready_for_gpu':
        raise ValueError('GPU plan no longer matches data/code; repeat CPU preflight.')
    from ..text.weights import check_base
    config = plan['configs'][candidate]
    if check_base(base, config) != plan['base_id']:
        raise ValueError('Base model differs from preflight.')
    return train_experiment(data['text-development'], base, output, config,
                            save_resume_state=True, resume_state=resume)


def fit_gpu(packet, base, plan_path, output, worker_script):
    # PSEUDOCODE: isolate candidate processes -> reuse completed candidates or resume epochs -> select on text validation alone.
    import torch
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Run the coordinator once, not through torchrun; individual trainers support DDP separately.')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU stage requires CUDA; CPU preparation is complete.')
    data, packet_manifest = load_packet(packet)
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if plan['packet_id'] != packet_manifest['id'] or plan['implementation_id'] != implementation_id() or plan['status'] != 'ready_for_gpu':
        raise ValueError('GPU plan differs from data/code.')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    binding = {'packet_id': packet_manifest['id'], 'plan_id': fingerprint(plan)}
    bound = output / 'binding.json'
    if bound.exists():
        if json.loads(bound.read_text(encoding='utf-8')) != binding:
            raise ValueError('GPU output belongs to another plan.')
    else:
        save_json(bound, binding)
    candidates = []
    for i, config in enumerate(plan['configs']):
        root = output / ('rate-' + str(config['encoder_lr']))
        attempts = sorted(root.glob('attempt-*'))
        complete = [p / 'result.json' for p in attempts if (p / 'result.json').exists()]
        if complete:
            result_path = complete[-1]
        else:
            receipts = [p for a in attempts for p in (a / 'resume').glob('epoch-*.json')]
            receipts.sort(key=lambda p: (json.loads(p.read_text())['epoch'], p.as_posix()))
            resume = receipts[-1] if receipts else None
            if resume:
                metadata = json.loads(resume.read_text(encoding='utf-8'))
                history = json.loads((resume.parent.parent / 'history.json').read_text(encoding='utf-8'))
                selected = (resume.parent / metadata['best_checkpoint']).resolve()
                best_manifest, best_id = verify_new_checkpoint(selected, data, packet_manifest)
                best_error = best_manifest['selected_validation_mae']
                best_epoch = best_manifest['epoch']
                finished = metadata['epoch'] >= config['epochs'] or metadata['epoch'] - best_epoch >= config['patience']
                if finished:
                    # An interruption after the final epoch can precede publication of result.json.
                    if best_id != metadata['best_identity'] or best_manifest['config'] != config:
                        raise ValueError('Completed epoch receipt no longer matches the selected model.')
                    result_path = resume.parent.parent / 'result.json'
                    save_json(result_path, {'best_checkpoint': str(selected), 'validation_mae': best_error,
                              'history': history, 'recovered_from_completed_epoch': True})
                    complete.append(result_path)
            attempt = root / f'attempt-{len(attempts) + 1:03d}'
            command = [sys.executable, str(worker_script), '_train', '--packet', str(Path(packet).resolve()),
                       '--base', str(Path(base).resolve()), '--plan', str(Path(plan_path).resolve()),
                       '--candidate', str(i), '--output', str(attempt)]
            if resume:
                command += ['--resume', str(resume)]
            if not complete:
                print('Training candidate ' + str(i + 1), flush=True)
                subprocess.run(command, check=True)
                result_path = attempt / 'result.json'
        result = json.loads(result_path.read_text(encoding='utf-8'))
        checkpoint = Path(result['best_checkpoint']).resolve()
        trained, identity = verify_new_checkpoint(checkpoint, data, packet_manifest)
        if trained['config'] != config or not math.isfinite(result['validation_mae']):
            raise ValueError('Candidate settings or validation result changed.')
        candidates.append({'checkpoint': checkpoint.relative_to(output).as_posix(), 'identity': identity,
                           'validation_mae': result['validation_mae'], 'config': config})
    best = min(candidates, key=lambda c: (c['validation_mae'], c['checkpoint']))
    selection = {**binding, 'selected': best, 'candidates': candidates, 'test_used_for_selection': False}
    target = output / 'selection.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != selection:
            raise ValueError('Existing selection differs; preserve it and inspect.')
    else:
        save_json(target, selection)
    return output / best['checkpoint']


def score_text(packet, checkpoint, base, output):
    # PSEUDOCODE: verify independence -> predict singleton texts -> atomically save each result for interruption-safe resumption.
    from ..text.predict import TextPredictor
    data, manifest = load_packet(packet)
    model, identity = verify_new_checkpoint(checkpoint, data, manifest)
    tasks = inference_tasks(data)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    cache = output / 'cache'; cache.mkdir(exist_ok=True)
    binding = {'packet_id': manifest['id'], 'model_id': identity, 'implementation_id': implementation_id(),
               'precision': 'fp32', 'batch_size': 1, 'tasks_id': fingerprint(tasks)}
    path = output / 'binding.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != binding:
            raise ValueError('Prediction cache identity differs.')
    else:
        save_json(path, binding)
    predictor = TextPredictor(checkpoint, base_path=base, device='cuda', allow_experimental=True, precision='fp32')
    profile, predictions = None, {}
    # Even on a cache hit, run one input to verify the actual current execution profile.
    for number, (key, task) in enumerate(sorted(tasks.items())):
        target = cache / (key + '.json')
        current = predictor.predict(**task) if number == 0 or not target.exists() else None
        if current is not None:
            current_profile = current['inference_profile']
            if profile is not None and profile != current_profile:
                raise ValueError('Inference profile changed mid-run.')
            profile = current_profile
        if target.exists():
            saved = json.loads(target.read_text(encoding='utf-8'))
            if saved['task'] != task or saved['model_id'] != identity or saved['profile'] != profile:
                raise ValueError('Cached prediction no longer matches current execution.')
        else:
            saved = {'task': task, 'model_id': identity, 'profile': profile, 'estimates': current['estimates']}
            # Each completed file is tiny; an interrupted temporary file can be safely replaced on resume.
            partial = target.with_suffix('.partial')
            partial.write_text(canonical_json(saved), encoding='utf-8')
            partial.replace(target)
        predictions[key] = saved['estimates']
        if (number + 1) % 100 == 0:
            print(f'Text inference: {number + 1}/{len(tasks)}', flush=True)
    result = {**binding, 'profile': profile, 'predictions': predictions, 'experimental_uncalibrated': True,
              'checkpoint_dataset_id': model['dataset']['id'], 'initialization': model.get('initialization')}
    result['id'] = fingerprint(result)
    target = output / 'predictions.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != result:
            raise ValueError('Existing complete predictions differ.')
    else:
        save_json(target, result)
    return target


def attach_predictions(data, manifest, path):
    # PSEUDOCODE: bind complete predictions to the frozen packet, selected checkpoint receipt and exact text content.
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    if fingerprint({k: v for k, v in result.items() if k != 'id'}) != result['id']:
        raise ValueError('Prediction file checksum failed.')
    tasks = inference_tasks(data)
    if (result['packet_id'] != manifest['id'] or result['tasks_id'] != fingerprint(tasks)
            or result['checkpoint_dataset_id'] != data['text-development']['manifest']['id']
            or result['initialization'] or result['implementation_id'] != implementation_id()
            or set(result['predictions']) != set(tasks) or not result['experimental_uncalibrated']):
        raise ValueError('Predictions do not belong to the independent authored experiment.')
    for key, estimates in result['predictions'].items():
        if set(estimates) != set(CATEGORIES[tasks[key]['category']][1]) or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100 for v in estimates.values()):
            raise ValueError('Invalid text prediction values.')
    for role in ('reference', 'train', 'validation', 'test'):
        for row in data[role]:
            for category, text in row['texts'].items():
                for key, value in result['predictions'][fingerprint([category, text])].items():
                    if 'text_' + key in TEXT:
                        row['features']['text_' + key] = value
    return result['id']
