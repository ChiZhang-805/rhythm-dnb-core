"""Pinned local base weights and explicit acquisition, shared by training and inference."""

import json
from pathlib import Path
import shutil
from ..provenance import canonical_json, file_hash, fingerprint

MODELS = {'Qwen/Qwen3-8B': 'qwen3', 'hfl/chinese-macbert-base': 'bert'}


def check_base(folder, config):
    # PSEUDOCODE: verify identity, complete inventory and every file hash before loading local weights.
    folder = Path(folder).resolve()
    receipt = json.loads((folder / 'download.json').read_text(encoding='utf-8'))
    if receipt.get('model_id') != config['model_id'] or receipt.get('revision') != config['revision']:
        raise ValueError('Base model receipt differs from configuration.')
    files = receipt.get('files', {})
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p != folder / 'download.json'}
    tokenizer_files = {'vocab.txt'} if MODELS[config['model_id']] == 'bert' else {'tokenizer.json', 'tokenizer_config.json'}
    if not files or actual != set(files) or not {'config.json', *tokenizer_files} <= actual or not any(n.endswith(('.safetensors', '.bin')) for n in actual):
        raise ValueError('Base receipt must cover the complete model and tokenizer inventory.')
    for name, digest in files.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or file_hash(path) != digest:
            raise ValueError('Base model path/hash mismatch: ' + name)
    architecture = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if architecture.get('model_type') != MODELS[config['model_id']]:
        raise ValueError('Base architecture differs from the selected model.')
    return fingerprint(receipt)


def acquire_base(config, output_path, cache_dir):
    # PSEUDOCODE: download a pinned official snapshot into an explicit cache -> copy runtime files -> seal receipt.
    from huggingface_hub import snapshot_download
    from .config import validate_config
    config = validate_config(config)
    output = Path(output_path).resolve()
    if output.exists():
        raise ValueError('Choose a new base-model directory; existing weights are preserved.')
    snapshot = Path(snapshot_download(config['model_id'], revision=config['revision'], cache_dir=cache_dir,
        allow_patterns=['*.safetensors', '*.safetensors.index.json', 'pytorch_model.bin', 'config.json',
                        'tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'vocab.json', 'merges.txt',
                        'special_tokens_map.json', 'added_tokens.json']))
    output.mkdir(parents=True, exist_ok=False)
    for path in snapshot.iterdir():
        if path.is_file():
            shutil.copyfile(path, output / path.name)
    receipt = {'model_id': config['model_id'], 'revision': config['revision'],
               'files': {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
    (output / 'download.json').write_text(canonical_json(receipt), encoding='utf-8')
    return {'path': str(output), 'identity': check_base(output, config)}
