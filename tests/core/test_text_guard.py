"""Frozen evidence heads, held-out integrity and unchanged semantic predictions."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from rhythm_dnb.provenance import canonical_json, fingerprint, file_hash
from rhythm_dnb.text.schema import CATEGORIES, METRICS
from rhythm_dnb.text.features import validate_feature_rows, extract_features, load_features
from rhythm_dnb.text.guard import fit_guard, evaluate_guard, EvidenceGuard, choose_empirical_threshold, selective_metrics


def feature_fixture(directory, roles, checkpoint_id='software-only-checkpoint', width=3):
    # PSEUDOCODE: store separable software-test features with truthful category masks and partition identities.
    directory = Path(directory); directory.mkdir()
    rows, vectors, scores, prior = [], [], [], []
    for role in roles:
        for category, (_, keys) in CATEGORIES.items():
            for supported in (False, True):
                for index in range(2):
                    identity = f'{role}:{category}:{supported}:{index}'
                    rows.append({'example_id': identity, 'group_id': role + str(index), 'family_id': role + str(index),
                        'category': category, 'split': role, 'text': '软件测试文本 ' + identity,
                        'scores': {key: None for key in keys},
                        'label_states': {key: 'supported_unscored' if supported else 'insufficient_evidence' for key in keys}})
                    vector = np.zeros(width); vector[0] = 1 if supported else -1
                    vectors.append(vector)
                    mask = np.asarray([metric[1] == category for metric in METRICS])
                    scores.append(np.where(mask, 50., np.nan)); prior.append(np.where(mask, .5, np.nan))
    np.savez_compressed(directory/'features.npz', vectors=np.asarray(vectors, dtype=np.float32), scores=scores, prior_evidence=prior)
    (directory/'rows.json').write_text(canonical_json(rows), encoding='utf-8')
    manifest = {'checkpoint_id': checkpoint_id, 'rows_id': fingerprint(rows), 'rows': len(rows), 'width': width,
                'files': {name: file_hash(directory/name) for name in ('rows.json', 'features.npz')}}
    (directory/'manifest.json').write_text(canonical_json(manifest), encoding='utf-8')
    return directory


class TextGuardTests(unittest.TestCase):
    def test_frozen_fit_selection_separate_test_and_parent_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            development = feature_fixture(root/'development', ['train', 'validation'])
            test = feature_fixture(root/'test', ['test'])
            trained = fit_guard(development, root/'guard', regularization=[.001, .01], target_precision=.95)
            self.assertFalse(trained['test_loaded'])
            self.assertFalse(trained['intensity_weights_changed'])
            self.assertTrue(all(head['validation_selection']['precision'] == 1. for head in trained['heads'].values()))
            result = evaluate_guard(test, root/'guard', root/'evaluation')
            self.assertLess(result['macro_brier'], result['macro_constant_brier'])
            self.assertFalse(result['evidence_calibrated'])
            with self.assertRaisesRegex(ValueError, 'exact experimental checkpoint'):
                EvidenceGuard(root/'guard', 'wrong-parent')
            with self.assertRaisesRegex(ValueError, 'train and validation only'):
                fit_guard(test, root/'test-fit', regularization=[.1], target_precision=.95)
            overlapping = feature_fixture(root/'overlapping', ['train'])
            with self.assertRaisesRegex(ValueError, 'separately sealed'):
                evaluate_guard(overlapping, root/'guard', root/'bad-evaluation')
            guard = EvidenceGuard(root/'guard', 'software-only-checkpoint')
            with self.assertRaisesRegex(ValueError, 'Unsupported'):
                guard.probabilities([[1, 0, 0]], 'unknown-category')
            with (root/'guard/weights.npz').open('ab') as stream:
                stream.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'weights changed'):
                EvidenceGuard(root/'guard', 'software-only-checkpoint')

    def test_duplicate_text_cache_tampering_and_threshold_ties(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = feature_fixture(Path(folder)/'cache', ['train', 'validation'])
            rows, *_ = load_features(cache)
            duplicate = copy.deepcopy(rows)
            duplicate[-1]['text'] = duplicate[0]['text']
            with self.assertRaisesRegex(ValueError, 'duplicate text'):
                validate_feature_rows(duplicate)
            changed = copy.deepcopy(rows); changed[-1]['group_id'] = changed[0]['group_id']
            with self.assertRaisesRegex(ValueError, 'Cross-split'):
                validate_feature_rows(changed)
            (cache/'rows.json').write_text(canonical_json(rows[::-1]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'cache changed'):
                load_features(cache)
        self.assertIsNone(choose_empirical_threshold([0, 1], [.7, .7], .95))
        self.assertEqual(choose_empirical_threshold([0, 1, 1], [.2, .8, .9], .95), .8)
        self.assertEqual(selective_metrics([0, 1], [.3, .7], None)['accepted'], 0)
        with self.assertRaisesRegex(ValueError, 'finite probability'):
            selective_metrics([0, 1], [.3, .7], float('nan'))

    def test_feature_extraction_and_guard_leave_checkpoint_scores_unchanged(self):
        from core.test_text_experiment import experimental_fixture
        from rhythm_dnb.text.experiment import train_experiment
        from rhythm_dnb.text.predict import TextPredictor
        from rhythm_dnb.text.checkpoint import load_checkpoint
        from rhythm_dnb.text.dataset import encode
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, config, exported = experimental_fixture(root)
            payload = json.loads((exported/'development.json').read_text(encoding='utf-8'))
            result = train_experiment(payload, base, root/'training', config)
            checkpoint = Path(result['best_checkpoint'])
            before = {p.name: file_hash(p) for p in checkpoint.iterdir() if p.is_file()}
            rows = payload['rows']
            for row in rows:
                row['label_states'] = {key: 'supported' for key in row['scores']}
            extracted = extract_features(rows, checkpoint, base, root/'extracted', device='cpu', batch_size=3)
            _, vectors, scores, _, _ = load_features(root/'extracted')
            model, tokenizer, _, identity = load_checkpoint(checkpoint, base_path=base, allow_experimental=True)
            model.eval()
            for index, row in enumerate(rows):
                encoded = encode(tokenizer, row['category'], row['text'], config['max_length'])
                with torch.inference_mode():
                    outputs = model(**{k: torch.tensor([v]) for k, v in encoded.items()}, return_representation=True)
                columns = [i for i, m in enumerate(METRICS) if m[1] == row['category']]
                np.testing.assert_allclose(scores[index, columns], outputs[row['category']][0].numpy()*100, atol=2e-5)
                np.testing.assert_allclose(vectors[index], outputs['_representation'][0].numpy(), atol=2e-5)
            cache = feature_fixture(root/'guard-cache', ['train', 'validation'], identity, extracted['width'])
            fit_guard(cache, root/'guard', regularization=[.001], target_precision=.95)
            plain = TextPredictor(checkpoint, base_path=base, device='cpu', allow_experimental=True)
            guarded = TextPredictor(checkpoint, base_path=base, device='cpu', allow_experimental=True, guard_dir=root/'guard')
            original = plain.predict('stress', '今天很有压力')
            guarded_output = guarded.predict('stress', '今天很有压力')
            self.assertEqual(original['estimates'], guarded_output['estimates'])
            self.assertIsNone(guarded_output['scores']['stress_intensity'])
            self.assertFalse(guarded_output['acceptance_certified'])
            self.assertFalse(guarded_output['eligible_for_primary_dnb'])
            self.assertEqual(before, {p.name: file_hash(p) for p in checkpoint.iterdir() if p.is_file()})
