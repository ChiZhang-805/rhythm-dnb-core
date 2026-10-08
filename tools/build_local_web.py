"""Package browser source and inference weights without training records or private experiment metadata."""

import argparse
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

from rhythm_dnb.provenance import canonical_json, file_hash
from rhythm_dnb.text.checkpoint import inspect_checkpoint


def public_manifest(manifest, identity):
    # PSEUDOCODE: retain inference-critical fields and source identity; omit participants, splits and training histories.
    fields = ('purpose', 'status', 'storage', 'contract', 'config', 'files', 'base_id')
    return {**{key: manifest[key] for key in fields}, 'run_id': 'public-text-scoring',
            'source_checkpoint_identity': identity, 'independent_human_gold': False,
            'description': 'Experimental Chinese text scoring; weights unchanged; training metadata omitted for distribution.'}


def build(root, checkpoint, output, license_path):
    # PSEUDOCODE: validate weights -> explicitly inventory distributable source -> write a new archive and checksums.
    root, checkpoint, output = Path(root).resolve(), Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new output path; existing packages are preserved.')
    manifest, identity = inspect_checkpoint(checkpoint, allow_experimental=True)
    if manifest['purpose'] != 'experimental_semantic_regression':
        raise ValueError('This public package is restricted to the experimental scorer.')
    public = public_manifest(manifest, identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'w', ZIP_DEFLATED, compresslevel=4) as archive:
        paths = [p for p in (root / 'src/rhythm_dnb').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        paths += [root / name for name in ('pyproject.toml', 'install-web.cmd', 'start-web.cmd',
                 'tools/local_web.py', 'docs/web-use.md', 'deploy/local-requirements.txt')]
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('Package path escapes source.')
            archive.write(path, 'rhythm-text-local/' + path.relative_to(root).as_posix())
        for name in manifest['files']:
            archive.write(checkpoint / name, 'rhythm-text-local/models/rhythm-text-expanded/' + name)
        archive.writestr('rhythm-text-local/models/rhythm-text-expanded/manifest.json', canonical_json(public))
        archive.write(root / 'docs/web-use.md', 'rhythm-text-local/README.md')
        archive.write(license_path, 'rhythm-text-local/QWEN-LICENSE.txt')
        archive.writestr('rhythm-text-local/MODEL-NOTICE.txt',
                        'Base: Qwen/Qwen3-8B, Apache License 2.0.\n'
                        'Modified: LoRA parameters and regression heads trained for experimental Chinese text scoring.\n'
                        'Original base weights are downloaded separately from the pinned official revision.\n'
                        'This package does not contain training data or a validated medical/DNB warning model.\n')
    receipt = {'archive': output.name, 'sha256': file_hash(output), 'bytes': output.stat().st_size,
               'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
               'source_checkpoint_identity': identity, 'public_manifest': public,
               'base_weights_included': False, 'training_data_included': False}
    output.with_suffix('.receipt.json').write_text(canonical_json(receipt), encoding='utf-8')
    output.with_suffix('.sha256').write_text(receipt['sha256'] + '  ' + output.name + '\n', encoding='utf-8')
    return {key: receipt[key] for key in ('archive', 'sha256', 'bytes', 'source_checkpoint_identity')}


def main():
    # PSEUDOCODE: require an explicit destination and the upstream model license.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--license', required=True)
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[1], args.checkpoint, args.output, args.license)))


if __name__ == '__main__':
    main()
