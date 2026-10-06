"""Checksummed epoch-boundary recovery of model updates, optimizer, scheduler and random streams."""

import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from ..provenance import canonical_json, file_hash, fingerprint
from .checkpoint import inspect_checkpoint


def implementation_identity():
    # PSEUDOCODE: fingerprint the calculation files independently of checkout path and line endings.
    directory = Path(__file__).parent
    names = ('train.py', 'resume.py', 'model.py', 'dataset.py', 'runtime.py', 'config.py', 'schema.py',
             'checkpoint.py', 'continuation.py', 'experiment.py', 'annotations.py', 'labels.py',
             'review.py', 'evidence.py', 'evaluate.py', 'corpus.py')
    return fingerprint({name: (directory / name).read_text(encoding='utf-8') for name in names})


def random_state(runtime):
    # PSEUDOCODE: capture each process's Python, NumPy, CPU and assigned CUDA generator without pickle-only objects.
    numpy_state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(runtime.device) if runtime.device.type == 'cuda' else None}


def model_parameters(model):
    # PSEUDOCODE: retain learned parameters and all heads; reload immutable frozen backbone weights from the pinned base.
    return {name: value for name, value in model.named_parameters()
            if value.requires_grad or not name.startswith('encoder.')}


def save_state(directory, model, optimizer, scheduler, scaler, runtime, execution, signature, history, stale, best_path):
    # PSEUDOCODE: gather per-rank randomness -> save one complete epoch state -> publish its hash receipt last.
    states = runtime.gather(random_state(runtime))

    def publish():
        # PSEUDOCODE: reserve immutable epoch filenames; an interrupted partial file has no valid receipt.
        directory.mkdir(parents=True, exist_ok=True)
        epoch = history[-1]['epoch']
        target = directory / f'epoch-{epoch}.pt'
        receipt = target.with_suffix('.json')
        partial = target.with_suffix('.partial')
        if any(path.exists() for path in (target, receipt, partial)):
            raise ValueError('Resume-state output already exists.')
        payload = {'parameters': {name: value.detach().cpu() for name, value in model_parameters(model).items()},
                   'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                   'scaler': scaler.state_dict(), 'random': states, 'history': history, 'stale': stale}
        torch.save(payload, partial)
        partial.replace(target)
        metadata = {'state_file': target.name, 'sha256': file_hash(target), 'signature': signature,
                    'implementation': execution['training_implementation'], 'execution': execution, 'epoch': epoch,
                    'best_checkpoint': os.path.relpath(best_path, directory),
                    'best_identity': file_hash(Path(best_path) / 'manifest.json')}
        receipt.write_text(canonical_json(metadata), encoding='utf-8')
        return str(receipt.resolve())

    return runtime.primary(publish)


def read_state(receipt_path, signature, execution, config):
    # PSEUDOCODE: verify matching data/config/code/runtime and complete bytes before decoding tensor-only recovery data.
    receipt = Path(receipt_path).resolve()
    metadata = json.loads(receipt.read_text(encoding='utf-8'))
    if metadata['signature'] != signature or metadata['implementation'] != execution['training_implementation']:
        raise ValueError('Resume requires unchanged data, base, settings and training implementation.')
    if metadata['execution'] != execution:
        raise ValueError('Resume requires the same library versions, devices, precision and process layout.')
    target = (receipt.parent / metadata['state_file']).resolve()
    if target.parent != receipt.parent or file_hash(target) != metadata['sha256']:
        raise ValueError('Resume state path/hash mismatch.')
    state = torch.load(target, map_location='cpu', weights_only=True)
    history = state['history']
    if (not history or [entry['epoch'] for entry in history] != list(range(1, metadata['epoch'] + 1))
            or metadata['epoch'] >= config['epochs'] or state['stale'] >= config['patience']):
        raise ValueError('Resume needs a valid unfinished epoch history.')
    if len(state['random']) != execution['world_size']:
        raise ValueError('Resume random streams do not match the process layout.')
    best_path = (receipt.parent / metadata['best_checkpoint']).resolve()
    _, identity = inspect_checkpoint(best_path, allow_experimental=True)
    if identity != metadata['best_identity']:
        raise ValueError('Previously selected checkpoint changed after the resume snapshot.')
    return state, str(best_path)


def restore_state(state, model, optimizer, scheduler, scaler, runtime):
    # PSEUDOCODE: restore every learned tensor and optimizer counter, then restore randomness after model construction.
    expected = model_parameters(model)
    if set(state['parameters']) != set(expected):
        raise ValueError('Resume parameter inventory differs from the configured model.')
    with torch.no_grad():
        for name, parameter in expected.items():
            saved = state['parameters'][name]
            if saved.shape != parameter.shape or saved.dtype != parameter.dtype or not torch.isfinite(saved).all():
                raise ValueError('Invalid resume parameter: ' + name)
            parameter.copy_(saved)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    generators = state['random'][runtime.rank]
    random.setstate(generators['python'])
    numpy_state = generators['numpy']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(generators['torch'])
    if runtime.device.type == 'cuda':
        torch.cuda.set_rng_state(generators['cuda'], runtime.device)


def carry_best_checkpoint(source, output):
    # PSEUDOCODE: make the new run self-contained and verify the copied best model before allowing continuation.
    _, identity = inspect_checkpoint(source, allow_experimental=True)
    target = output / 'resume-best'
    shutil.copytree(source, target)
    _, copied = inspect_checkpoint(target, allow_experimental=True)
    if copied != identity:
        raise ValueError('Best checkpoint changed while copying it into the resumed run.')
    return str(target.resolve())
