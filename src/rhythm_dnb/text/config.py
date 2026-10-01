"""One primary model, one comparator, and explicit research/compute settings."""

import math
from .weights import MODELS

DEFAULTS = {'precision': 'auto', 'gradient_checkpointing': True, 'num_workers': 0,
            'quantization': 'none', 'lora_rank': 16, 'lora_alpha': 32,
            'evidence_loss_weight': 0.1, 'evidence_precision': 0.95, 'evidence_min_samples': 20,
            'process_timeout_minutes': 120}


def validate_config(config):
    # PSEUDOCODE: normalize options -> validate pinned model and optimization/measurement settings.
    config = {**DEFAULTS, **config}
    required = {'model_id', 'revision', 'seed', 'max_length', 'batch_size', 'gradient_accumulation',
                'epochs', 'patience', 'encoder_lr', 'head_lr', 'weight_decay', 'warmup_ratio',
                'huber_delta', 'dropout', 'max_grad_norm', 'threads', 'device'} | set(DEFAULTS)
    if set(config) != required or config['model_id'] not in MODELS:
        raise ValueError('Unsupported training configuration.')
    revision = config['revision']
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Base model revision must be an immutable commit.')
    for key in ('seed', 'max_length', 'batch_size', 'gradient_accumulation', 'epochs', 'patience',
                'threads', 'lora_rank', 'lora_alpha', 'evidence_min_samples', 'process_timeout_minutes'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError('Invalid positive integer: ' + key)
    if config['seed'] >= 2 ** 32:
        raise ValueError('Training seed must fit the NumPy random-state range.')
    if type(config['num_workers']) is not int or config['num_workers'] < 0 or type(config['gradient_checkpointing']) is not bool:
        raise ValueError('Invalid data-loader/checkpointing settings.')
    if config['device'] not in ('auto', 'cpu', 'cuda') or config['precision'] not in ('auto', 'fp32', 'fp16', 'bf16'):
        raise ValueError('Unsupported device or precision.')
    if config['quantization'] not in ('none', 'nf4') or config['model_id'].startswith('hfl/') and config['quantization'] != 'none':
        raise ValueError('NF4 is supported only for the Qwen backbone.')
    for key in ('encoder_lr', 'head_lr', 'huber_delta', 'max_grad_norm', 'evidence_loss_weight'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError('Invalid positive parameter: ' + key)
    for key in ('dropout', 'warmup_ratio', 'weight_decay', 'evidence_precision'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
            raise ValueError('Invalid finite parameter: ' + key)
    limit = 512 if MODELS[config['model_id']] == 'bert' else 40960
    if not 8 <= config['max_length'] <= limit or not 0 <= config['dropout'] < 1 or not 0 <= config['warmup_ratio'] < 1 or config['weight_decay'] < 0 or not 0 < config['evidence_precision'] <= 1:
        raise ValueError('Invalid token limit, regularization or evidence precision.')
    return config
