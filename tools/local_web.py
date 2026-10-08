"""Prepare an isolated local GPU runtime and launch the browser page without any cloud service."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv


def main():
    # PSEUDOCODE: install only on request -> download the exact base -> start the local server with unchanged weights.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--download-base', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    environment = {**os.environ, 'PYTHONPATH': str(root / 'src'), 'PYTHONUTF8': '1',
                   'PIP_CACHE_DIR': str(root / 'cache/pip')}
    python = root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    checkpoint = root / 'models/rhythm-text-expanded'
    if not (checkpoint / 'manifest.json').exists():
        raise SystemExit('Missing trained model. Download rhythm-text-local.zip from the GitHub local-web release first.')
    if args.prepare:
        if sys.version_info[:2] != (3, 13):
            raise SystemExit('Install Python 3.13 for this verified local runtime.')
        try:
            subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], check=True)
        except (OSError, subprocess.CalledProcessError):
            raise SystemExit('An NVIDIA GPU and working driver are required before downloading the runtime.')
        print('Preparing local runtime and base weights. Allow about 50 GB free disk space.', flush=True)
        if not python.exists():
            venv.create(root / '.venv', with_pip=True)
        subprocess.run([str(python), '-m', 'pip', 'install', 'torch==2.8.0', '--index-url', 'https://download.pytorch.org/whl/cu128'], env=environment, check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '-r', 'deploy/local-requirements.txt'], env=environment, check=True)
        subprocess.run([str(python), '-c', 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable; check the NVIDIA driver."'], env=environment, check=True)
        subprocess.run([str(python), str(Path(__file__).resolve()), '--download-base'], env=environment, check=True)
        print('Ready. Run start-web.cmd or python tools/local_web.py, then open http://127.0.0.1:7860', flush=True)
        return
    if args.download_base:
        from rhythm_dnb.text.weights import acquire_base, check_base
        config = json.loads((checkpoint / 'manifest.json').read_text(encoding='utf-8'))['config']
        base = root / 'models/qwen3-8b'
        if base.exists():
            check_base(base, config)
        else:
            acquire_base(config, base, root / 'cache/base-download')
        return
    if not python.exists():
        raise SystemExit('Run install-web.cmd or python tools/local_web.py --prepare first.')
    subprocess.run([str(python), '-m', 'rhythm_dnb.web'], env=environment, check=True)


if __name__ == '__main__':
    main()
