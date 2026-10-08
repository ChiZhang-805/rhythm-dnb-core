"""Inference precision must be explicit, reproducible and unable to bypass evidence calibration."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from rhythm_dnb.text.inference import resolve_precision, inference_profile
from rhythm_dnb.text.predict import TextPredictor


class TextInferenceTests(unittest.TestCase):
    def test_precision_cannot_silently_fall_back_or_change_formal_calibration(self):
        cpu, cuda = torch.device('cpu'), torch.device('cuda')
        self.assertEqual(resolve_precision('auto', cpu, 'adapter'), torch.float32)
        self.assertEqual(resolve_precision('fp32', cpu, 'full'), torch.float32)
        for name in ('bf16', 'fp16'):
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                resolve_precision(name, cpu, 'adapter')
            with self.assertRaisesRegex(ValueError, 'comparator'):
                resolve_precision(name, cuda, 'full')
        with patch('torch.cuda.is_bf16_supported', return_value=False):
            self.assertEqual(resolve_precision('auto', cuda, 'adapter'), torch.float16)
            with self.assertRaisesRegex(ValueError, 'not supported'):
                resolve_precision('bf16', cuda, 'adapter')
        with patch('torch.cuda.is_bf16_supported', return_value=True):
            self.assertEqual(resolve_precision('auto', cuda, 'adapter'), torch.bfloat16)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            resolve_precision('typo', cpu, 'adapter')
        with patch('rhythm_dnb.text.predict.inspect_checkpoint', return_value=({'purpose': 'full_dataset_finetune'}, 'formal')):
            with self.assertRaisesRegex(ValueError, 'formal calibration'):
                TextPredictor('unused', precision='fp32', allow_experimental=True)
        with patch('rhythm_dnb.text.predict.inspect_checkpoint', return_value=({'purpose': 'experimental_semantic_regression'}, 'experiment')):
            with self.assertRaisesRegex(ValueError, 'frozen-feature guard'):
                TextPredictor('unused', precision='fp32', guard_dir='unused', allow_experimental=True)

    def test_receipt_identifies_arithmetic_separately_from_weights(self):
        model = SimpleNamespace(heads=torch.nn.Linear(1, 1), encoder=SimpleNamespace(config=SimpleNamespace(_attn_implementation='sdpa')))
        manifest = {'base_id': 'base', 'config': {'quantization': 'none'}}
        target = torch.device('cpu')
        first = inference_profile('same-weights', manifest, model, target, torch.float32)
        repeated = inference_profile('same-weights', manifest, model, target, torch.float32)
        changed = inference_profile('same-weights', manifest, model, target, torch.bfloat16)
        self.assertEqual(first, repeated)
        self.assertEqual(first['checkpoint_id'], changed['checkpoint_id'])
        self.assertNotEqual(first['id'], changed['id'])
        self.assertEqual(first['batch_size'], 1)
        self.assertEqual(first['compute_dtype'], 'float32')
        self.assertFalse(first['bitwise_reproducibility_guaranteed'])
