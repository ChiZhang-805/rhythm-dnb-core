"""Epoch recovery must match uninterrupted dropout training and reject altered experiments."""

import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from safetensors.torch import load_file
import torch

from rhythm_dnb.text.experiment import train_experiment
from core.test_text_experiment import experimental_fixture


class TextResumeTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec('bitsandbytes'), 'NF4 needs CUDA')
    def test_quantized_cuda_recovery_retains_optimizer_and_dropout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, config, exported = experimental_fixture(root)
            config.update(epochs=2, patience=3, dropout=.2, device='cuda', precision='auto', quantization='nf4')
            corpus = json.loads((exported / 'development.json').read_text(encoding='utf-8'))
            original = train_experiment(corpus, base, root / 'original', config, save_resume_state=True)
            resumed = train_experiment(corpus, base, root / 'resumed', config,
                                       resume_state=root / 'original/resume/epoch-1.json')
            self.assertAlmostEqual(original['history'][1]['training_loss'], resumed['history'][1]['training_loss'], places=6)
            self.assertAlmostEqual(original['validation_mae'], resumed['validation_mae'], places=4)
            self.assertGreater(resumed['history'][1]['optimizer_steps'], 0)

    @unittest.skipIf(sys.platform == 'win32', 'Distributed recovery is verified on Linux')
    def test_distributed_recovery_restores_both_rank_random_streams(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, config, _ = experimental_fixture(root)
            config.update(epochs=3, patience=4, dropout=.2, warmup_ratio=.3)
            (root / 'training.json').write_text(json.dumps(config), encoding='utf-8')
            worker = Path(__file__).resolve().parents[1] / 'support/resume_worker.py'
            for stage in ('original', 'resumed'):
                process = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                    '--nproc-per-node=2', str(worker), str(root), stage],
                    env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
                    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
                self.assertEqual(process.returncode, 0, process.stdout[-4000:] + process.stderr[-8000:])
            original = json.loads((root / 'original/result.json').read_text(encoding='utf-8'))
            resumed = json.loads((root / 'resumed/result.json').read_text(encoding='utf-8'))
            self.assertEqual(resumed['execution']['world_size'], 2)
            for left, right in zip(original['history'], resumed['history']):
                self.assertEqual(left['training_loss'], right['training_loss'])
                self.assertEqual(left['validation'], right['validation'])

    def test_epoch_recovery_preserves_updates_scheduler_and_randomness(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, config, exported = experimental_fixture(root)
            config.update(epochs=3, patience=4, dropout=.2, warmup_ratio=.3)
            corpus = json.loads((exported / 'development.json').read_text(encoding='utf-8'))
            original = train_experiment(corpus, base, root / 'original', config, save_resume_state=True)
            without_snapshots = train_experiment(corpus, base, root / 'without-snapshots', config)
            receipt = root / 'original/resume/epoch-1.json'
            resumed = train_experiment(corpus, base, root / 'resumed', config, resume_state=receipt)
            self.assertEqual((root / 'original/epoch-1/manifest.json').read_bytes(),
                             (root / 'resumed/resume-best/manifest.json').read_bytes())
            for left, right in zip(original['history'], resumed['history']):
                self.assertEqual(left['training_loss'], right['training_loss'])
                self.assertEqual(left['validation'], right['validation'])
                self.assertEqual(left['optimizer_steps'], right['optimizer_steps'])
            self.assertEqual([h['training_loss'] for h in original['history']],
                             [h['training_loss'] for h in without_snapshots['history']])
            self.assertEqual(original['validation_mae'], resumed['validation_mae'])
            for name in ('model.safetensors', 'adapter/adapter_model.safetensors'):
                expected = load_file(str(Path(original['best_checkpoint']) / name))
                actual = load_file(str(Path(resumed['best_checkpoint']) / name))
                for key in expected:
                    torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'unchanged'):
                train_experiment(corpus, base, root / 'changed', {**config, 'encoder_lr': .002}, resume_state=receipt)
            self.assertFalse((root / 'changed').exists())
            state = receipt.with_suffix('.pt')
            with state.open('ab') as stream:
                stream.write(b'incomplete-or-modified')
            with self.assertRaisesRegex(ValueError, 'hash'):
                train_experiment(corpus, base, root / 'tampered', config, resume_state=receipt)

    def test_completed_training_cannot_restart_as_an_unfinished_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, config, exported = experimental_fixture(root)
            corpus = json.loads((exported / 'development.json').read_text(encoding='utf-8'))
            train_experiment(corpus, base, root / 'original', config, save_resume_state=True)
            with self.assertRaisesRegex(ValueError, 'unfinished'):
                train_experiment(corpus, base, root / 'resumed', config,
                                 resume_state=root / 'original/resume/epoch-1.json')
