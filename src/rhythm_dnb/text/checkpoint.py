"""Verified offline checkpoints; Qwen adapters bind to an explicitly supplied immutable base."""

import json
from pathlib import Path
from ..provenance import file_hash, canonical_json
from .schema import schema
from .weights import check_base


def device_for(name):
    # PSEUDOCODE: resolve only supported devices and fail when requested hardware is absent.
    import torch
    if name not in ('cpu', 'cuda', 'auto'):
        raise ValueError('Unknown device.')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable.')
    return torch.device('cuda' if name != 'cpu' and torch.cuda.is_available() else 'cpu')


def inspect_checkpoint(folder, *, allow_experimental=False, allow_evidence=False):
    # PSEUDOCODE: validate completed semantic contract and every pinned file, including adapter and head weights.
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    purposes = ('full_dataset_finetune', 'experimental_semantic_regression') if allow_experimental else ('full_dataset_finetune',)
    if allow_evidence:
        purposes += ('experimental_evidence',)
    if manifest.get('contract') != schema() or manifest.get('status') != 'complete' or manifest.get('purpose') not in purposes:
        raise ValueError('Checkpoint is incomplete or has a different semantic contract.')
    files = manifest.get('files', {})
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p != folder / 'manifest.json'}
    if actual != set(files):
        raise ValueError('Checkpoint file inventory differs from its manifest.')
    required = {'model.safetensors', 'encoder/config.json'}
    if manifest.get('storage') == 'adapter':
        required |= {'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'}
    elif manifest.get('storage') != 'full':
        raise ValueError('Unknown checkpoint storage.')
    if not required <= set(files) or not any(n.startswith('tokenizer/') for n in files):
        raise ValueError('Checkpoint manifest omits required model files.')
    for name, digest in files.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or file_hash(path) != digest:
            raise ValueError('Checkpoint path/hash mismatch: ' + name)
    return manifest, file_hash(folder / 'manifest.json')


def load_tokenizer(folder):
    # PSEUDOCODE: load only local standard tokenizer files -> require explicit padding and right-aligned batching.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer needs a padding or end token.')
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return tokenizer


def load_checkpoint(folder, *, base_path=None, device='cpu', dtype=None, allow_experimental=False, trainable=False, allow_evidence=False):
    # PSEUDOCODE: verify checkpoint/base -> restore small adapters or full comparator -> load tokenizer.
    import torch
    from safetensors.torch import load_file
    from .model import ScoringModel
    folder = Path(folder)
    manifest, identity = inspect_checkpoint(folder, allow_experimental=allow_experimental, allow_evidence=allow_evidence)
    target = device_for(device) if isinstance(device, str) else device
    if manifest['storage'] == 'adapter':
        from peft import PeftModel
        if base_path is None or check_base(base_path, manifest['config']) != manifest['base_id']:
            raise ValueError('Qwen checkpoint needs the exact verified base directory via --base.')
        if dtype is None:
            dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if target.type == 'cuda' else torch.float32
        model = ScoringModel.pretrained(base_path, manifest['config']['dropout'], config=manifest['config'],
                                        device=target, dtype=dtype, training=False)
        model.encoder = PeftModel.from_pretrained(model.encoder, folder / 'adapter', is_trainable=trainable, local_files_only=True)
        heads = load_file(str(folder / 'model.safetensors'))
        expected = {k for k in model.state_dict() if not k.startswith('encoder.')}
        if set(heads) != expected:
            raise ValueError('Adapter checkpoint has incorrect score/evidence heads.')
        model.load_state_dict(heads, strict=False)
    else:
        model = ScoringModel.from_config(folder / 'encoder', manifest['config']['dropout'],
                                        scope_supervision=manifest['config'].get('scope_loss_weight', 0) > 0)
        model.load_state_dict(load_file(str(folder / 'model.safetensors')), strict=True)
        model.to(target)
    tokenizer = load_tokenizer(folder / 'tokenizer')
    return model, tokenizer, manifest, identity


def save_checkpoint(folder, model, tokenizer, config, metadata, *, purpose='full_dataset_finetune'):
    # PSEUDOCODE: reserve directory -> save adapters/heads or comparator -> publish complete checksum manifest last.
    from safetensors.torch import save_file
    if purpose not in ('full_dataset_finetune', 'experimental_semantic_regression', 'experimental_evidence'):
        raise ValueError('Unknown checkpoint purpose.')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    adapter = hasattr(model.encoder, 'peft_config')
    if adapter:
        model.encoder.save_pretrained(folder / 'adapter', safe_serialization=True, save_embedding_layers=False)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if not adapter or not k.startswith('encoder.')}
    save_file(state, str(folder / 'model.safetensors'))
    model.encoder.config.save_pretrained(folder / 'encoder')
    tokenizer.save_pretrained(folder / 'tokenizer')
    files = {p.relative_to(folder).as_posix(): file_hash(p) for p in folder.rglob('*') if p.is_file()}
    manifest = {**metadata, 'purpose': purpose, 'status': 'complete',
                'storage': 'adapter' if adapter else 'full', 'contract': schema(), 'config': config, 'files': files}
    (folder / 'manifest.json').write_text(canonical_json(manifest), encoding='utf-8')
    return folder
