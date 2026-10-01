"""Prevent removed source files from silently reappearing in a distributable wheel."""

from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile
from tools.build_wheel import verify_wheel


class PackagingTests(unittest.TestCase):
    def test_packaged_inventory_rejects_stale_and_modified_code(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'src/rhythm_dnb'; source.mkdir(parents=True)
            (source / '__init__.py').write_bytes(b'# current\n')
            for label, content in (
                    ('current', {'rhythm_dnb/__init__.py': b'# current\n'}),
                    ('stale', {'rhythm_dnb/__init__.py': b'# current\n', 'rhythm_dnb/simulate.py': b'# removed\n'}),
                    ('modified', {'rhythm_dnb/__init__.py': b'# old\n'})):
                wheel = root / (label + '.whl')
                with ZipFile(wheel, 'w') as archive:
                    for name, data in content.items():
                        archive.writestr(name, data)
                if label == 'current':
                    self.assertEqual(verify_wheel(wheel, root), 1)
                else:
                    with self.assertRaisesRegex(ValueError, 'stale'):
                        verify_wheel(wheel, root)
