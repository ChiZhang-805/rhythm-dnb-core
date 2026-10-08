"""Explicit inference arithmetic and reproducibility receipts, separate from weight identity."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from ..provenance import file_hash, fingerprint


def resolve_precision(name, target, storage):
    # PSEUDOCODE: preserve legacy automatic arithmetic; reject an explicit mode that cannot actually be applied.
    import torch
    if name not in ('auto', 'fp32', 'bf16', 'fp16'):
        raise ValueError('Unknown inference precision.')
    if storage == 'full':
        if name not in ('auto', 'fp32'):
            raise ValueError('Full comparator checkpoints currently use fp32 inference.')
        return torch.float32
    if name == 'auto':
        name = ('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if target.type == 'cuda' else 'fp32'
    if name != 'fp32' and target.type != 'cuda':
        raise ValueError('Reduced inference precision requires CUDA.')
    if name == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('Requested bf16 inference is not supported on this GPU.')
    return {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[name]


def inference_profile(identity, manifest, model, target, dtype):
    # PSEUDOCODE: fingerprint effective single-text execution, not merely the requested flag or checkpoint name.
    import torch
    libraries = {'torch': torch.__version__}
    for name in ('transformers', 'peft', 'bitsandbytes', 'safetensors'):
        try:
            libraries[name] = version(name)
        except PackageNotFoundError:
            libraries[name] = None
    folder = Path(__file__).parent
    profile = {
        'checkpoint_id': identity, 'base_id': manifest.get('base_id'),
        'compute_dtype': str(dtype).removeprefix('torch.'),
        'quantization': manifest['config'].get('quantization', 'none'),
        'head_dtype': str(next(model.heads.parameters()).dtype).removeprefix('torch.'),
        'batch_size': 1, 'padding': 'single_text_without_batch_padding',
        'attention_implementation': getattr(model.encoder.config, '_attn_implementation', None),
        'device_type': target.type, 'libraries': libraries,
        'cuda_build': torch.version.cuda if target.type == 'cuda' else None,
        'gpu': torch.cuda.get_device_name(target) if target.type == 'cuda' else None,
        'matmul_precision': torch.get_float32_matmul_precision(),
        'cuda_matmul_tf32': torch.backends.cuda.matmul.allow_tf32 if target.type == 'cuda' else None,
        'implementation': {name: file_hash(folder / name) for name in ('inference.py', 'predict.py', 'evidence_adapter.py', 'model.py', 'dataset.py', 'checkpoint.py')},
        'bitwise_reproducibility_guaranteed': False,
    }
    return {**profile, 'id': fingerprint(profile)}
