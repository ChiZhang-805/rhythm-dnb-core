"""Exact sample weighting, distributed checkpoint ownership and available GPU execution."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from rhythm_dnb.text.runtime import ShardedBatches, TrainingRuntime
from rhythm_dnb.text.train import train, validate_config
from support.text_fixture import create_text_fixture


class TrainingRuntimeTests(unittest.TestCase):
    def test_incomplete_distributed_epoch_never_duplicates_real_examples(self):
        for size in (1, 5, 9, 17):
            ranks = [list(ShardedBatches(size, 2, rank, 4, 31, 0)) for rank in range(4)]
            self.assertEqual(len({len(batches) for batches in ranks}), 1)
            actual = [index for batches in ranks for batch in batches for index, weight in batch if weight]
            self.assertEqual(sorted(actual), list(range(size)))

    def test_hardware_settings_are_strict(self):
        with tempfile.TemporaryDirectory() as folder:
            _, config, _ = create_text_fixture(folder)
            for changes in ({'precision': 'int8'}, {'gradient_checkpointing': 1}, {'num_workers': -1}, {'device': 'cuda:100'}, {'seed': 2 ** 32}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    validate_config({**config, **changes})

    def test_cpu_does_not_silently_accept_cuda_precision(self):
        with self.assertRaises(ValueError):
            with TrainingRuntime({'device': 'cpu', 'precision': 'fp16'}):
                pass

    def test_primary_failure_is_propagated(self):
        with TrainingRuntime({'device': 'cpu', 'precision': 'fp32'}) as runtime:
            with self.assertRaisesRegex(RuntimeError, 'division by zero'):
                runtime.primary(lambda: 1 / 0)

    @unittest.skipIf(sys.platform == 'win32', 'Distributed training is validated on Linux; Windows uses a single device')
    def test_two_process_training_matches_one_global_batch(self):
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, config, rows = create_text_fixture(root)
            single = train(rows, base, root / 'single', config)
            # Both layouts have one global group of eight slots, with only five genuine samples.
            (root / 'training.json').write_text(json.dumps({**config, 'gradient_accumulation': 2}), encoding='utf-8')
            (root / 'corpus.json').write_text(json.dumps(rows), encoding='utf-8')
            environment = {**os.environ, 'USE_LIBUV': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
            worker = str(Path(__file__).resolve().parents[1] / 'support/train_worker.py')
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2', worker, str(root)]
            process = subprocess.run(command, env=environment, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
            self.assertEqual(process.returncode, 0, process.stdout[-4000:] + process.stderr[-8000:])
            distributed = json.loads((root / 'distributed/result.json').read_text(encoding='utf-8'))
            self.assertEqual(distributed['execution']['world_size'], 2)
            self.assertEqual(distributed['history'][0]['training_samples'], 5)
            self.assertEqual(distributed['test']['records'], 5)
            self.assertEqual(distributed['history'][0]['optimizer_steps'], 1)
            expected = load_file(str(Path(single['best_checkpoint']) / 'model.safetensors'))
            actual = load_file(str(Path(distributed['best_checkpoint']) / 'model.safetensors'))
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=1e-4, atol=2e-6, msg=key)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware is not present')
    def test_available_gpu_runs_training_and_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, config, rows = create_text_fixture(root)
            result = train(rows, base, root / 'cuda', {**config, 'device': 'cuda', 'precision': 'auto'})
            self.assertEqual(result['history'][0]['training_samples'], 5)
            self.assertEqual(result['history'][0]['optimizer_steps'], 1)
            self.assertEqual(result['test']['records'], 5)
            self.assertIn('cuda', result['execution']['devices'][0]['device'])
            self.assertTrue(Path(result['best_checkpoint']).is_dir())

    @unittest.skipIf(sys.platform == 'win32', 'Distributed training is validated on Linux; Windows uses a single device')
    def test_qwen_two_process_adapter_updates_match_global_batch(self):
        from core.test_text_measurement import qwen_fixture
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, config, rows = qwen_fixture(root)
            single = train(rows, base, root / 'single', config)
            (root / 'training.json').write_text(json.dumps({**config, 'gradient_accumulation': 1}), encoding='utf-8')
            (root / 'corpus.json').write_text(json.dumps(rows), encoding='utf-8')
            worker = str(Path(__file__).resolve().parents[1] / 'support/train_worker.py')
            process = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2', worker, str(root)],
                env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
            self.assertEqual(process.returncode, 0, process.stdout[-4000:] + process.stderr[-8000:])
            other = json.loads((root / 'distributed/result.json').read_text(encoding='utf-8'))
            self.assertEqual(other['history'][0]['training_samples'], 10)
            for name in ('model.safetensors', 'adapter/adapter_model.safetensors'):
                expected = load_file(str(Path(single['best_checkpoint']) / name))
                actual = load_file(str(Path(other['best_checkpoint']) / name))
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=1e-4, atol=2e-6, msg=key)
