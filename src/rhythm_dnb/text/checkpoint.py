"""Offline checkpoint I/O, independent of training and corpus code."""

import json
from pathlib import Path
from ..provenance import file_hash, canonical_json
from .schema import schema


def device_for(name):
    # PSEUDOCODE: resolve only supported devices and fail when requested hardware is absent.
    import torch
    if name not in ('cpu', 'cuda', 'auto'):
        raise ValueError('Unknown device.')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable.')
    return torch.device('cuda' if name != 'cpu' and torch.cuda.is_available() else 'cpu')


def inspect_checkpoint(folder):
    # PSEUDOCODE: validate completed semantic contract and every pinned file, including required weights.
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('contract') != schema() or manifest.get('status') != 'complete' or manifest.get('purpose') != 'full_dataset_finetune':
        raise ValueError('Checkpoint is incomplete or has a different semantic contract.')
    files = manifest.get('files', {})
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    if actual != set(files):
        raise ValueError('Checkpoint file inventory differs from its manifest.')
    if not {'model.safetensors', 'encoder/config.json'} <= set(files) or not any(n.startswith('tokenizer/') for n in files):
        raise ValueError('Checkpoint manifest omits required model files.')
    for name, digest in files.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or file_hash(path) != digest:
            raise ValueError('Checkpoint path/hash mismatch: ' + name)
    return manifest, file_hash(folder / 'manifest.json')


def load_checkpoint(folder):
    # PSEUDOCODE: verify before loading -> instantiate local architecture -> load safe tensor weights.
    from safetensors.torch import load_file
    from transformers import BertTokenizer
    from .model import ScoringModel
    folder = Path(folder)
    manifest, identity = inspect_checkpoint(folder)
    model = ScoringModel.from_config(folder / 'encoder', manifest['config']['dropout'])
    model.load_state_dict(load_file(str(folder / 'model.safetensors')), strict=True)
    tokenizer = BertTokenizer.from_pretrained(folder / 'tokenizer', local_files_only=True)
    return model, tokenizer, manifest, identity


def save_checkpoint(folder, model, tokenizer, config, metadata):
    # PSEUDOCODE: reserve a new checkpoint directory -> save weights/tokenizer -> publish checksum manifest last.
    from safetensors.torch import save_file
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(folder / 'model.safetensors'))
    model.encoder.config.save_pretrained(folder / 'encoder')
    tokenizer.save_pretrained(folder / 'tokenizer')
    files = {p.relative_to(folder).as_posix(): file_hash(p) for p in folder.rglob('*') if p.is_file()}
    manifest = {**metadata, 'purpose': 'full_dataset_finetune', 'status': 'complete',
                'contract': schema(), 'config': config, 'files': files}
    (folder / 'manifest.json').write_text(canonical_json(manifest), encoding='utf-8')
    return folder
