"""Package consistent SQLite snapshots and restore verified data under an explicit project root."""

import argparse
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tempfile
import zipfile


def digest(stream):
    # PSEUDOCODE: hash a stream in bounded chunks without loading a dataset into memory.
    result = sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def database_inventory(path):
    # PSEUDOCODE: open a read-only database -> check integrity -> count each preserved table.
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as database:
        if database.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
            raise ValueError('SQLite integrity check failed: ' + str(path))
        names = [row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {name: database.execute('SELECT COUNT(*) FROM "' + name.replace('"', '""') + '"').fetchone()[0]
                for name in sorted(names)}


def snapshot(source, destination):
    # PSEUDOCODE: use SQLite backup to include committed WAL data without changing the live source.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb'):
        pass
    with closing(sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)) as original:
        with closing(sqlite3.connect(destination)) as copied:
            original.backup(copied, pages=1024)
    return database_inventory(destination)


def pack(text_database, rhythm_database, source_directory, output_directory, public_library=None):
    # PSEUDOCODE: freeze both databases -> collect complete source files -> record hashes -> write a private archive.
    text_database, rhythm_database, source_directory = map(Path, (text_database, rhythm_database, source_directory))
    if not text_database.is_file() or not rhythm_database.is_file() or not source_directory.is_dir():
        raise ValueError('Both databases and the actual source-data directory are required.')
    output = Path(output_directory).resolve()
    roots = [(source_directory.resolve(), 'data/sources')]
    if public_library is not None:
        public_library = Path(public_library).resolve()
        for name in ('00_registry', '01_raw_public'):
            if not (public_library / name).is_dir():
                raise ValueError('Public source library lacks ' + name)
            roots.append((public_library / name, 'data/public_library/' + name))
    if any(output.is_relative_to(root) for root, _ in roots):
        raise ValueError('Upload output must be outside source-data directories.')
    output.mkdir(parents=True, exist_ok=False)
    archive = output / 'dnb-data.zip'
    with tempfile.TemporaryDirectory(prefix='snapshots-', dir=output) as temporary:
        staging = Path(temporary).resolve()
        if not staging.is_relative_to(output):
            raise ValueError('Snapshot staging escaped its output directory.')
        entries, databases, excluded = {}, {}, []
        for source, name in ((text_database, 'master.sqlite'), (rhythm_database, 'rhythm.sqlite')):
            relative = 'data/legacy/' + name
            target = staging / name
            databases[relative] = {'original_path': str(source.resolve()), 'tables': snapshot(source, target)}
            entries[relative] = target
        for root, prefix in roots:
            for path in sorted(root.rglob('*')):
                if path.is_symlink():
                    raise ValueError('Source-data symlinks need an explicit materialized source: ' + str(path))
                if not path.is_file():
                    continue
                if path.suffix.lower() in ('.partial', '.tmp', '.pyc') or '__pycache__' in path.parts:
                    excluded.append(str(path))
                    continue
                entries[prefix + '/' + path.relative_to(root).as_posix()] = path
        manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'purpose': 'source_data_transfer_not_training_certification',
                    'databases': databases, 'source_roots': [{'original_path': str(root), 'relative_path': prefix} for root, prefix in roots],
                    'excluded_incomplete_or_temporary_files': excluded, 'files': {}}
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for name, path in sorted(entries.items()):
                before = path.stat()
                with path.open('rb') as source, zipped.open(name, 'w', force_zip64=True) as target:
                    checksum, size = sha256(), 0
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        target.write(block)
                        checksum.update(block)
                        size += len(block)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('Source changed during packaging; retry into a new output directory: ' + str(path))
                manifest['files'][name] = {'bytes': size, 'sha256': checksum.hexdigest()}
            zipped.writestr('data/transfer-manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        with zipfile.ZipFile(archive) as zipped:
            for name, entry in manifest['files'].items():
                with zipped.open(name) as stream:
                    if digest(stream) != entry['sha256']:
                        raise ValueError('Archive verification failed: ' + name)
        with archive.open('rb') as stream:
            checksum = digest(stream)
    (output / 'dnb-data.zip.sha256').write_text(checksum + '  dnb-data.zip\n', encoding='utf-8')
    return {'archive': str(archive), 'bytes': archive.stat().st_size, 'files': len(entries), 'sha256': checksum,
            'databases': databases, 'excluded_files': len(excluded)}


def unpack(archive, project_root):
    # PSEUDOCODE: reject existing data -> verify every relative path and digest -> restore without overwriting files.
    root = Path(project_root).resolve()
    if not (root / 'src/rhythm_dnb/cli.py').is_file() or not (root / 'pyproject.toml').is_file():
        raise ValueError('Choose the rhythm-dnb-core project root.')
    destination = root / 'data'
    if destination.is_symlink() or destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError('Data already exists; this command will not overwrite it.')
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive entries.')
        manifest = json.loads(zipped.read('data/transfer-manifest.json'))
        entries = manifest['files']
        if set(names) != set(entries) | {'data/transfer-manifest.json'}:
            raise ValueError('Archive inventory differs from its manifest.')
        for name in names:
            parts = PurePosixPath(name).parts
            if not parts or parts[0] != 'data' or '\\' in name or ':' in name or '..' in parts or not (root / name).resolve().is_relative_to(destination):
                raise ValueError('Unsafe archive path: ' + name)
            if name in entries:
                if zipped.getinfo(name).file_size != entries[name]['bytes']:
                    raise ValueError('Incorrect file size: ' + name)
                with zipped.open(name) as stream:
                    if digest(stream) != entries[name]['sha256']:
                        raise ValueError('Incorrect file hash: ' + name)
        for name in names:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(name) as source, target.open('xb') as copied:
                shutil.copyfileobj(source, copied, length=1024 * 1024)
    for name, entry in manifest['databases'].items():
        if name not in entries or database_inventory(root / name) != entry['tables']:
            raise ValueError('Restored database inventory differs: ' + name)
    return {'data_directory': str(destination), 'verified_files': len(entries), 'databases': len(manifest['databases']),
            'training_ready': False, 'note': 'Legacy source snapshots must still pass current corpus/provenance checks.'}


def main():
    # PSEUDOCODE: require explicit source/destination paths and print a compact transfer receipt.
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    create = commands.add_parser('pack')
    create.add_argument('--text-db', required=True)
    create.add_argument('--rhythm-db', required=True)
    create.add_argument('--sources', required=True)
    create.add_argument('--public-library')
    create.add_argument('--output-dir', required=True)
    restore = commands.add_parser('unpack')
    restore.add_argument('--archive', required=True)
    restore.add_argument('--project-root', required=True)
    args = parser.parse_args()
    result = (pack(args.text_db, args.rhythm_db, args.sources, args.output_dir, args.public_library)
              if args.operation == 'pack' else unpack(args.archive, args.project_root))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
