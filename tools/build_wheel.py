"""Build from a fresh source copy and reject missing, stale or extra Python modules in the wheel."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile


def verify_wheel(wheel, root):
    # PSEUDOCODE: compare every packaged module byte-for-byte with the current source inventory.
    expected = {p.relative_to(root / 'src').as_posix(): p.read_bytes() for p in (root / 'src/rhythm_dnb').rglob('*.py')}
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        actual = {n: archive.read(n) for n in names if n.endswith('.py')}
        if len(names) != len(set(names)) or actual != expected:
            raise ValueError('Wheel contains missing, stale or extra Python code.')
    return len(expected)


def build(root, output):
    # PSEUDOCODE: copy tracked current files into an isolated directory -> build -> verify -> retain only the wheel.
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new output directory; previous build evidence is preserved.')
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    commit = subprocess.check_output(['git', 'rev-parse', '--short=12', 'HEAD'], cwd=root).decode().strip()
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='wheel-source-', dir=output) as folder:
        source = Path(folder)
        digest = sha256()
        for name in sorted(n for n in tracked if n):
            path = root / name
            if not path.is_file():
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('Build input escapes the source repository.')
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            digest.update(name.encode()); digest.update(path.read_bytes())
        environment = {**os.environ, 'SETUPTOOLS_SCM_PRETEND_VERSION': f'0.dev0+g{commit}.c{digest.hexdigest()[:12]}'}
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '.', '--wheel-dir', str(output)],
                       cwd=source, env=environment, check=True)
        wheels = list(output.glob('*.whl'))
        if len(wheels) != 1:
            raise ValueError('Build did not produce exactly one wheel.')
        count = verify_wheel(wheels[0], root)
    return {'wheel': str(wheels[0]), 'verified_modules': count}


def main():
    # PSEUDOCODE: use an explicit output directory under the project or its owning build root.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(build(Path(__file__).resolve().parents[1], args.output_dir))


if __name__ == '__main__':
    main()
