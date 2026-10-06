"""Validate real training inputs and token lengths without loading a neural model."""

from pathlib import Path
import numpy as np
from .config import validate_config
from .corpus import prepare_corpus
from .checkpoint import load_tokenizer
from .dataset import ScoreDataset
from .weights import check_base


def check_training_inputs(rows, base_path, config, *, development_only=False):
    # PSEUDOCODE: apply training's corpus/base/token checks -> report sizes without allocating model weights.
    config = validate_config(config)
    partitions, corpus = prepare_corpus(rows, require_calibration=config['model_id'] == 'Qwen/Qwen3-8B',
                                        development_only=development_only)
    base_path = Path(base_path).resolve()
    base_id = check_base(base_path, config)
    tokenizer = load_tokenizer(base_path)
    from .review import training_readiness
    readiness = training_readiness(partitions, config, experimental=False)
    if readiness['blockers']:
        raise ValueError('Training readiness: ' + '; '.join(readiness['blockers']))
    tokens = {}
    for split, items in partitions.items():
        lengths = []
        dataset = ScoreDataset(items, tokenizer, config['max_length'])
        for encoded in dataset.features:
            lengths.append(len(encoded['input_ids']))
        tokens[split] = {'records': len(items), 'people': len(corpus['people'][split]),
                         'min': min(lengths), 'median': float(np.median(lengths)),
                         'p95': float(np.quantile(lengths, .95)), 'max': max(lengths)}
    return {'status': 'inputs_validated', 'development_only': development_only,
            'corpus_id': corpus['id'], 'base_id': base_id, 'config': config, 'tokens': tokens,
            'batch_size_per_device': config['batch_size'], 'gradient_accumulation': config['gradient_accumulation'],
            'effective_batch_per_device': config['batch_size'] * config['gradient_accumulation'],
            'model_loaded': False, 'gpu_memory_fit_verified': False,
            'readiness': readiness,
            'interpretation': 'Input integrity checks only; training, annotation validity and GPU capacity remain unverified.'}
