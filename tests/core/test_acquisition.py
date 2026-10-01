"""Acquisition tests use local bytes only; never download research weights in the test suite."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rhythm_dnb.text.weights import _download_file, _verify_shards


class AcquisitionTests(unittest.TestCase):
    def test_resume_requires_exact_range_and_official_hash(self):
        payload = b'0123456789'
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'weights.safetensors'
            partial = destination.with_name(destination.name + '.incomplete')
            partial.write_bytes(payload[:4])
            response = BytesIO(payload[4:])
            response.status = 206
            response.headers = {'Content-Range': 'bytes 4-9/10'}
            with patch('rhythm_dnb.text.weights.urlopen', return_value=response) as fetch:
                path, digest = _download_file('https://example.invalid/weights', destination, 10, sha256(payload).hexdigest(), None)
            self.assertEqual(fetch.call_args.args[0].headers['Range'], 'bytes=4-9')
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(digest, sha256(payload).hexdigest())
            self.assertFalse(partial.exists())
            with self.assertRaisesRegex(ValueError, 'official digest'):
                _download_file('unused', destination, 10, '0' * 64, None)

    def test_server_ignoring_range_preserves_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'weights.safetensors'
            partial = destination.with_name(destination.name + '.incomplete')
            partial.write_bytes(b'0123')
            response = BytesIO(b'0123456789'); response.status = 200; response.headers = {}
            with patch('rhythm_dnb.text.weights.urlopen', return_value=response):
                with self.assertRaisesRegex(ValueError, 'resume range'):
                    _download_file('https://example.invalid/weights', destination, 10, '0' * 64, None)
            self.assertEqual(partial.read_bytes(), b'0123')
            self.assertFalse(destination.exists())

    def test_partial_shard_set_is_not_a_complete_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index = 'model.safetensors.index.json'
            (root / index).write_text(json.dumps({'weight_map': {'a': 'model-00001-of-00002.safetensors', 'b': 'model-00002-of-00002.safetensors'}}))
            with self.assertRaisesRegex(ValueError, 'Missing'):
                _verify_shards(root, {index, 'model-00001-of-00002.safetensors'})
            with self.assertRaisesRegex(ValueError, 'index'):
                _verify_shards(root, {'model-00001-of-00002.safetensors'})
