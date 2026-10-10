"""Check time causality, whole-person fitting, missing evidence and matched model search."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest
import json
import tempfile

import numpy as np

from rhythm_dnb.research.temporal_warning import (causal_history, causal_smooth, candidates,
    fit_candidate, person_weights, select_learner, select_smoothing)
from tools.run_temporal_warning import attach_preserved_predictions, check_config, learned_scores
from rhythm_dnb.provenance import fingerprint
from rhythm_dnb.research.experiment_gpu import inference_tasks
from rhythm_dnb.text.schema import CATEGORIES
from tools.run_dnbr_warning import read_json
from pathlib import Path


def records(person, offsets):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [{'record_id': f'{person}-{i}', 'participant_id': person,
             'observed_at': (start+timedelta(days=i)).isoformat(),
             'issued_at': (start+timedelta(days=i, hours=12)).isoformat()} for i in offsets]


class TemporalWarningTests(unittest.TestCase):
    def config(self):
        return read_json(Path(__file__).resolve().parents[2] / 'configs/temporal_warning.json')

    def test_future_and_other_people_cannot_change_the_past(self):
        rows = records('a', [0, 1, 2, 3]) + records('b', [0, 1])
        values = {r['record_id']: [float(i)] for i, r in enumerate(rows)}
        first, names, lineage = causal_history(rows, values, ['x'], 3)
        changed = deepcopy(values); changed['a-3'] = [1e9]; changed['b-0'] = [-1e9]
        second, _, _ = causal_history(rows[::-1], changed, ['x'], 3)
        for rid in ('a-0', 'a-1', 'a-2'):
            np.testing.assert_allclose(first[rid], second[rid], equal_nan=True)
        self.assertEqual(lineage['a-3'], ['a-1', 'a-2', 'a-3'])
        self.assertEqual(names, ['current:x', 'mean:x', 'sd:x', 'change:x'])
        np.testing.assert_allclose(first['a-2'], [2, 1, 1, 1])
        self.assertTrue(np.isnan(first['a-0'][2:]).all())

    def test_missing_history_gap_and_duplicate_or_future_record(self):
        rows = records('a', [0, 2, 3]); values = {'a-0': [1.], 'a-2': [np.nan], 'a-3': [4.]}
        output, _, _ = causal_history(rows, values, ['x'], 3)
        self.assertTrue(np.isnan(output['a-2'][0]))
        self.assertTrue(np.isnan(output['a-2'][2:]).all())
        self.assertTrue(np.isnan(output['a-3'][2:]).all())
        with self.assertRaises(ValueError):
            causal_history(rows+[rows[0]], values, ['x'], 3)
        future = deepcopy(rows); future[0]['observed_at'] = rows[-1]['issued_at']
        with self.assertRaises(ValueError):
            causal_history(future, values, ['x'], 3)

    def test_smoothing_uses_elapsed_time_and_preserves_abstention(self):
        rows = records('a', [0, 2, 3]); scores = {'a-0': 8., 'a-2': 0., 'a-3': None}
        self.assertEqual(causal_smooth(rows, scores, 0), scores)
        actual = causal_smooth(rows, scores, 1)
        self.assertAlmostEqual(actual['a-2'], 2/1.25)
        self.assertIsNone(actual['a-3'])
        scores['a-3'] = 1e8
        self.assertEqual(causal_smooth(rows, scores, 1)['a-2'], actual['a-2'])
        with self.assertRaises(ValueError):
            causal_smooth(rows+[rows[0]], scores, 1)
        with self.assertRaises(ValueError):
            causal_smooth(rows, {**scores, 'a-0': np.inf}, 1)

    def test_whole_person_development_and_same_candidate_budget(self):
        config = self.config(); check_config(config)
        self.assertEqual(len(candidates(config)), 7)
        rng = np.random.default_rng(23)
        y = np.repeat(np.tile([0, 1], 12), 2); people = np.repeat([f'p{i}' for i in range(24)], 2)
        x = np.c_[y+rng.normal(0, .3, len(y)), rng.normal(size=len(y))]
        model, receipt = select_learner(x, y, people, config, 42)
        tested = []
        for split in receipt['inner_splits']:
            self.assertFalse(set(split['fit_people']) & set(split['validation_people']))
            tested += split['validation_people']
        self.assertEqual(sorted(tested), sorted(set(people)))
        self.assertEqual(len(receipt['trials']), 7)
        before = model.predict_proba(x)
        model.predict_proba([[1e8, -1e8]])
        np.testing.assert_array_equal(model.predict_proba(x), before)
        # All four identical smoothers must tie in favor of the unchanged raw score.
        selected = select_smoothing({h: y.astype(float) for h in config['half_lives_days']}, y, people, config, 42)
        self.assertEqual(selected['half_life'], 0)

    def test_training_only_imputation_and_no_invented_current_evidence(self):
        config = self.config()
        x = np.array([[0., np.nan], [1., 2.], [4., 6.], [8., 4.]])
        model = fit_candidate(x, [0, 0, 1, 1], list('abcd'), {'family': 'ridge_logistic', 'c': 1}, config, 1)
        np.testing.assert_allclose(model[0].statistics_, [2.5, 4.])
        rows = records('a', [0, 1])
        cache = {'current_valid': {'history_baseline': {'a-0': True, 'a-1': False}},
                 'values': {'history_baseline': {'a-0': [0., np.nan], 'a-1': [np.nan, 4.]}}}
        result = learned_scores(rows, cache, 'history_baseline', model)
        self.assertIsNotNone(result['a-0']); self.assertIsNone(result['a-1'])
        np.testing.assert_allclose(person_weights(['a', 'a', 'b']), [.25, .25, .5])
        with self.assertRaises(ValueError):
            fit_candidate(x, [0, 0, 1, 1], list('abcd'), {'family': 'typo'}, config, 1)

    def test_rejects_unsupported_or_malformed_parameters(self):
        for key, value in [('half_lives_days', [1, 2]), ('logistic_c', [0]), ('tree_leaves', [2.5]),
                           ('tree_learning_rate', float('nan')), ('history_days', True), ('selection', 'test_accuracy')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_config({**self.config(), key: value})

    def test_preserved_predictions_require_exact_verified_parent_and_text(self):
        data = {r: [{'record_id': r, 'texts': {'emotion': '今天很难过'}, 'features': {}}]
                for r in ('reference', 'train', 'validation', 'test')}
        data['text-development'] = {'manifest': {'id': 'independent-training'}}
        tasks = inference_tasks(data)
        payload = {'packet_id': 'packet', 'tasks_id': fingerprint(tasks), 'checkpoint_dataset_id': 'independent-training',
                   'initialization': None, 'implementation_id': 'original-inference-code', 'experimental_uncalibrated': True,
                   'predictions': {k: {m: 42. for m in CATEGORIES[v['category']][1]} for k, v in tasks.items()}}
        payload['id'] = fingerprint(payload)
        parent = {'packet_id': 'packet', 'inference_id': payload['id'], 'implementation_id': 'original-inference-code'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.json'; path.write_text(json.dumps(payload), encoding='utf-8')
            restored = deepcopy(data)
            self.assertEqual(attach_preserved_predictions(restored, {'id': 'packet'}, path, parent), payload['id'])
            self.assertEqual(restored['test'][0]['features']['text_sadness_intensity'], 42.)
            changed = deepcopy(data); changed['test'][0]['texts']['emotion'] = '今天很开心'
            for d, p in ((changed, parent), (data, {**parent, 'inference_id': 'different'}),
                         (data, {**parent, 'implementation_id': 'different'})):
                with self.assertRaises(ValueError):
                    attach_preserved_predictions(d, {'id': 'packet'}, path, p)
            payload['predictions'][next(iter(tasks))]['sadness_intensity'] = 99.
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError):
                attach_preserved_predictions(data, {'id': 'packet'}, path, parent)


if __name__ == '__main__':
    unittest.main()
