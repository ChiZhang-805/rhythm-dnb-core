"""Pinned local base weights and explicit acquisition, shared by training and inference."""

import json
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from hashlib import sha1
from pathlib import Path
import shutil
import sys
from urllib.request import Request, urlopen
from ..provenance import canonical_json, file_hash, fingerprint

MODELS = {'Qwen/Qwen3-8B': 'qwen3', 'hfl/chinese-macbert-base': 'bert'}
RUNTIME_FILES = ('*.safetensors', '*.safetensors.index.json', 'pytorch_model.bin', 'config.json', 'tokenizer.json',
                 'tokenizer_config.json', 'vocab.txt', 'vocab.json', 'merges.txt',
                 'special_tokens_map.json', 'added_tokens.json', 'LICENSE')


def _verify_shards(folder, names):
    # PSEUDOCODE: require every shard named by the index, never accepting a partial weight set.
    for name in names:
        if name.endswith('.safetensors.index.json'):
            index = json.loads((folder / name).read_text(encoding='utf-8'))
            shards = set(index.get('weight_map', {}).values())
            if not shards or not shards <= set(names) or any(not n.endswith('.safetensors') for n in shards):
                raise ValueError('Missing or invalid model shards.')
    if any('-of-' in n and n.endswith('.safetensors') for n in names) and not any(n.endswith('.safetensors.index.json') for n in names):
        raise ValueError('Sharded weights require their index.')


def _download_file(url, destination, size, sha256, git_blob):
    # PSEUDOCODE: resume an official file with checked ranges -> check upstream digest -> atomically publish cache file.
    destination = Path(destination)
    partial = destination.with_name(destination.name + '.incomplete')
    if not destination.exists():
        failures = 0
        while True:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == size:
                break
            if offset > size:
                raise ValueError('Cached partial file exceeds upstream size.')
            end = min(size - 1, offset + 64 * 1024 * 1024 - 1)
            ranged = bool(offset or end < size - 1)
            try:
                request = Request(url, headers={'Range': f'bytes={offset}-{end}'} if ranged else {})
                with urlopen(request, timeout=60) as response:
                    if ranged and (response.status != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{size}'):
                        raise ValueError('Server did not honor the resume range; partial data preserved.')
                    if not ranged and response.status != 200:
                        raise ValueError('Unexpected download response.')
                    with partial.open('ab') as stream:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                if partial.stat().st_size != end + 1:
                    raise OSError('Incomplete download; retrying from the saved offset.')
                failures = 0
                if ranged:
                    print(f'{destination.name}: {end + 1}/{size} bytes', file=sys.stderr, flush=True)
            except (OSError, TimeoutError):
                failures += 1
                if failures == 3:
                    raise
        candidate = partial
    else:
        candidate = destination
    if candidate.stat().st_size != size:
        raise ValueError('Downloaded size differs from official metadata.')
    digest = file_hash(candidate)
    if sha256:
        valid = digest == sha256
    else:
        blob = sha1(f'blob {size}\0'.encode())
        with candidate.open('rb') as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                blob.update(chunk)
        valid = blob.hexdigest() == git_blob
    if not valid:
        raise ValueError('Downloaded file differs from the official digest: ' + destination.name)
    if candidate == partial:
        partial.replace(destination)
    print(f'Verified {destination.name}: {size} bytes', file=sys.stderr, flush=True)
    return destination, digest


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
    _verify_shards(folder, actual)
    for name, digest in files.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or file_hash(path) != digest:
            raise ValueError('Base model path/hash mismatch: ' + name)
    architecture = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if architecture.get('model_type') != MODELS[config['model_id']]:
        raise ValueError('Base architecture differs from the selected model.')
    return fingerprint(receipt)


def acquire_base(config, output_path, cache_dir):
    # PSEUDOCODE: fetch pinned official metadata -> resume and verify files -> copy only the reviewed runtime inventory.
    from huggingface_hub import HfApi, hf_hub_url
    from .config import validate_config
    config = validate_config(config)
    output = Path(output_path).resolve()
    if output.exists():
        raise ValueError('Choose a new base-model directory; existing weights are preserved.')
    info = HfApi(endpoint='https://huggingface.co').model_info(config['model_id'], revision=config['revision'], files_metadata=True)
    if info.sha != config['revision']:
        raise ValueError('Upstream resolved a different model revision.')
    files = [item for item in info.siblings if any(fnmatch(item.rfilename, pattern) for pattern in RUNTIME_FILES)]
    if any(item.rfilename.endswith('.safetensors') for item in files):
        files = [item for item in files if item.rfilename != 'pytorch_model.bin']
    snapshot = Path(cache_dir).resolve() / 'verified' / config['model_id'].replace('/', '--') / config['revision']
    snapshot.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        # PSEUDOCODE: reject unexpected nested files and missing integrity metadata before downloading.
        if Path(item.rfilename).name != item.rfilename or type(item.size) is not int or item.size <= 0 or not item.blob_id:
            raise ValueError('Invalid upstream runtime file metadata.')
        return _download_file(hf_hub_url(config['model_id'], item.rfilename, revision=config['revision'], endpoint='https://huggingface.co'),
                              snapshot / item.rfilename, item.size, item.lfs.sha256 if item.lfs else None, item.blob_id)

    with ThreadPoolExecutor(max_workers=5) as pool:
        downloaded = list(pool.map(fetch, files))
    _verify_shards(snapshot, {p.name for p, _ in downloaded})
    output.mkdir(parents=True, exist_ok=False)
    for path, _ in downloaded:
        shutil.copyfile(path, output / path.name)
    receipt = {'model_id': config['model_id'], 'revision': config['revision'],
               'verification': 'official_sha256_or_git_blob_and_size',
               'files': {p.name: digest for p, digest in downloaded}}
    (output / 'download.json').write_text(canonical_json(receipt), encoding='utf-8')
    return {'path': str(output), 'identity': check_base(output, config)}
