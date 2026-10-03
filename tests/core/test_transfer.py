"""SQLite WAL snapshots, portable restore paths and refusal to overwrite existing research data."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zipfile
from tools.transfer_data import pack, unpack


class TransferTests(unittest.TestCase):
    def package(self, root):
        for name in ('text', 'rhythm'):
            with closing(sqlite3.connect(root / (name + '.sqlite'))) as database:
                database.execute('CREATE TABLE samples(id INTEGER PRIMARY KEY, payload TEXT)')
                database.execute('INSERT INTO samples VALUES(1, ?)', (name,))
                database.commit()
        sources = root / 'sources'
        sources.mkdir()
        (sources / 'complete.csv').write_text('value\n1\n', encoding='utf-8')
        (sources / 'unfinished.zip.partial').write_bytes(b'partial')
        return pack(root / 'text.sqlite', root / 'rhythm.sqlite', sources, root / 'upload')

    def project(self, root):
        project = root / 'server folder' / 'rhythm-dnb-core'
        (project / 'src/rhythm_dnb').mkdir(parents=True)
        (project / 'src/rhythm_dnb/cli.py').write_text('', encoding='utf-8')
        (project / 'pyproject.toml').write_text('', encoding='utf-8')
        return project

    def test_portable_roundtrip_preserves_sources_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = self.package(root)
            self.assertEqual(receipt['files'], 3)
            self.assertEqual(receipt['excluded_files'], 1)
            project = self.project(root)
            result = unpack(receipt['archive'], project)
            self.assertEqual(result['verified_files'], 3)
            self.assertFalse(result['training_ready'])
            self.assertEqual((project / 'data/sources/complete.csv').read_bytes(), (root / 'sources/complete.csv').read_bytes())
            self.assertTrue((root / 'sources/unfinished.zip.partial').exists())
            with closing(sqlite3.connect(project / 'data/legacy/master.sqlite')) as database:
                self.assertEqual(database.execute('SELECT payload FROM samples').fetchone()[0], 'text')
            with self.assertRaises(FileExistsError):
                unpack(receipt['archive'], project)

    def test_committed_wal_records_are_included_without_uncommitted_changes(self):
        from tools.transfer_data import snapshot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'active.sqlite'
            with closing(sqlite3.connect(source)) as writer:
                writer.execute('PRAGMA journal_mode=WAL')
                writer.execute('CREATE TABLE samples(id INTEGER)')
                writer.execute('INSERT INTO samples VALUES (1)')
                writer.commit()
                writer.execute('INSERT INTO samples VALUES (2)')
                self.assertEqual(snapshot(source, root / 'snapshot.sqlite'), {'samples': 1})
                self.assertEqual(writer.execute('SELECT COUNT(*) FROM samples').fetchone()[0], 2)

    def test_corrupt_hash_and_path_traversal_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = self.package(root)
            project = self.project(root)
            with zipfile.ZipFile(receipt['archive']) as zipped:
                original = {name: zipped.read(name) for name in zipped.namelist()}
            for kind in ('hash', 'traversal'):
                entries = dict(original)
                if kind == 'hash':
                    entries['data/sources/complete.csv'] = b'value\n2\n'
                else:
                    manifest = json.loads(entries['data/transfer-manifest.json'])
                    name = 'data/../../outside.csv'
                    manifest['files'][name] = manifest['files'].pop('data/sources/complete.csv')
                    entries[name] = entries.pop('data/sources/complete.csv')
                    entries['data/transfer-manifest.json'] = json.dumps(manifest).encode()
                bad = root / (kind + '.zip')
                with zipfile.ZipFile(bad, 'w') as zipped:
                    for name, contents in entries.items():
                        zipped.writestr(name, contents)
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    unpack(bad, project)
                self.assertFalse((project / 'data').exists())
                self.assertFalse((root / 'outside.csv').exists())
