"""Real model execution on a tiny local unit-test fixture; no downloads or clinical claims."""

from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from rhythm_dnb.text.schema import CATEGORIES
from rhythm_dnb.text.corpus import prepare_corpus
from rhythm_dnb.text.annotations import compare_annotations
from rhythm_dnb.provenance import file_hash


def toy_corpus():
    # PSEUDOCODE: fabricate a complete contract fixture with disjoint text and person groups.
    rows = []
    for split in ('train', 'validation', 'test'):
        for category, (_, keys) in CATEGORIES.items():
            identity = split + category
            rows.append({'example_id': identity, 'category': category, 'text': '今天的心情描述' + identity,
                'scores': {key: 50. for key in keys}, 'split': split, 'group_id': identity,
                'participant_id': identity, 'origin': 'observed', 'review_status': 'accepted',
                'annotation_evidence_id': 'unit-test-fixture-only'})
    return rows


class TextTests(unittest.TestCase):
    def test_corpus_order_is_identical_on_every_worker(self):
        first, receipt = prepare_corpus(toy_corpus())
        second, reverse_receipt = prepare_corpus(reversed(toy_corpus()))
        self.assertEqual(first, second)
        self.assertEqual(receipt, reverse_receipt)

    def test_corpus_requires_person_and_text_group_identity(self):
        for key in ('participant_id', 'group_id', 'example_id'):
            rows = toy_corpus(); rows[0][key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare_corpus(rows)

    def test_constructed_evaluation_and_duplicate_text_rejected(self):
        rows = toy_corpus(); rows[-1]['origin'] = 'constructed'
        with self.assertRaises(ValueError):
            prepare_corpus(rows)
        rows = toy_corpus(); rows[-1]['text'] = rows[0]['text']
        with self.assertRaises(ValueError):
            prepare_corpus(rows)

    def test_annotation_disagreement_preserved(self):
        first = {'example_id': '1', 'category': 'stress', 'rater_id': 'a', 'scores': {'stress_intensity': 10}}
        second = {**first, 'rater_id': 'b', 'scores': {'stress_intensity': 40}}
        self.assertTrue(compare_annotations(first, second, tolerance=10)['adjudication_required'])
        with self.assertRaises(ValueError):
            compare_annotations(first, first, tolerance=10)

    def test_full_training_checkpoint_and_continuous_inference(self):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizer
        from rhythm_dnb.text.train import train
        from rhythm_dnb.text.predict import TextPredictor
        from rhythm_dnb.text.checkpoint import inspect_checkpoint
        from rhythm_dnb.text.model import ScoringModel
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); base = root / 'base'; base.mkdir()
            vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + list(dict.fromkeys('今天的心情描述压力睡眠饮食社交绪'))
            (base / 'vocab.txt').write_text('\n'.join(vocab), encoding='utf-8')
            tokenizer = BertTokenizer(vocab_file=str(base / 'vocab.txt')); tokenizer.save_pretrained(base)
            model = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                                         num_attention_heads=2, intermediate_size=32, max_position_embeddings=128))
            model.save_pretrained(base)
            config = {'model_id': 'hfl/chinese-macbert-base', 'revision': 'a' * 40, 'seed': 37,
                'max_length': 64, 'batch_size': 2, 'gradient_accumulation': 2, 'epochs': 2,
                'patience': 2, 'encoder_lr': .001, 'head_lr': .001, 'weight_decay': .01,
                'warmup_ratio': .1, 'huber_delta': .1, 'dropout': 0., 'max_grad_norm': 1., 'threads': 1, 'device': 'cpu'}
            receipt = {'model_id': config['model_id'], 'revision': config['revision'],
                       'files': {p.name: file_hash(p) for p in base.iterdir() if p.is_file()}}
            (base / 'download.json').write_text(json.dumps(receipt))
            result = train(toy_corpus(), base, root / 'run', config)
            predictor = TextPredictor(result['best_checkpoint'])
            prediction = predictor.predict('stress', '今天压力很大')
            self.assertEqual(set(prediction['scores']), {'stress_intensity'})
            self.assertIsNone(prediction['scores']['stress_intensity'])
            self.assertEqual(prediction['reasons']['stress_intensity'], 'uncalibrated_text_evidence')
            self.assertIsInstance(prediction['estimates']['stress_intensity'], float)
            self.assertNotEqual(prediction['estimates']['stress_intensity'], round(prediction['estimates']['stress_intensity']))
            self.assertEqual(result['test']['records'], 5)
            manifest, _ = inspect_checkpoint(result['best_checkpoint'])
            self.assertFalse(manifest['test_used_for_selection'])
            reloaded = ScoringModel.from_config(Path(result['best_checkpoint']) / 'encoder', 0.)
            reloaded.load_state_dict(predictor.model.state_dict()); reloaded.eval()
            batch = tokenizer('压力', '今天压力很大', return_tensors='pt')
            with torch.inference_mode():
                predicted = predictor.model(**{k: v.to(predictor.device) for k, v in batch.items()})['stress'].cpu()
                torch.testing.assert_close(reloaded(**batch)['stress'], predicted, atol=1e-5, rtol=1e-5)
            path = Path(result['best_checkpoint']) / 'manifest.json'
            damaged = json.loads(path.read_text(encoding='utf-8')); damaged['files'].pop('model.safetensors')
            path.write_text(json.dumps(damaged))
            with self.assertRaises(ValueError):
                inspect_checkpoint(path.parent)
