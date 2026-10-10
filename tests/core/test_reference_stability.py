"""Check exact Pearson updates, unchanged DNB components, person resampling and invalid draws."""

import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

from rhythm_dnb.dnb.single_sample import sdnb_components
from rhythm_dnb.research.reference_stability import (network_changes, personal_components, reference_draws,
                                                    resample_scores, summarize_draws)
from tools.personal_warning import branches
from tools.run_reference_stability import augment_features, forecast_prefix, check_stability_config
from tools.run_dnbr_warning import read_json
from rhythm_dnb.research.temporal_warning import causal_history


class ReferenceStabilityTests(unittest.TestCase):
    def test_rank_one_update_matches_direct_pearson_and_kernel(self):
        rng = np.random.default_rng(891)
        r = rng.normal(size=(60, 10)); r[:, 1] = -.7*r[:, 0] + .3*r[:, 1]
        targets = rng.normal(size=(12, 10))*2; names = [f'x{i}' for i in range(10)]
        deviations, deltas = network_changes(r, targets)
        for x, deviation, delta in zip(targets, deviations, deltas):
            expected = np.corrcoef(np.vstack([r, x]), rowvar=False)-np.corrcoef(r, rowvar=False)
            np.fill_diagonal(expected, 0)
            np.testing.assert_allclose(delta, expected, atol=1e-14, rtol=1e-12)
            kernels = [sdnb_components(x, r, names, m, pair_convention='paper_k_squared') for m in branches(expected, names)]
            best = max(kernels, key=lambda v:v['score'])
            actual = personal_components(deviation, delta, 1e-8)
            self.assertAlmostEqual(actual['score'], best['score'], places=10)

    def test_targets_are_independent_and_future_targets_do_not_change_scores(self):
        rng = np.random.default_rng(26); r = rng.normal(size=(40, 5)); x = rng.normal(size=(3, 5))
        draws = reference_draws([f'p{i}' for i in range(40)], 6, 8)
        first, _ = resample_scores(r, x, draws, 1e-8)
        changed = x.copy(); changed[2] *= 20
        second, _ = resample_scores(r, changed, draws, 1e-8)
        for key in first:
            np.testing.assert_allclose(first[key][:, :2], second[key][:, :2])
        np.testing.assert_array_equal(draws, reference_draws([f'p{i}' for i in range(40)], 6, 8))

    def test_missing_and_undefined_ratios_are_not_hidden_by_averaging(self):
        median, spread, valid = summarize_draws([[1, 3], [2, np.nan], [3, 5], [4, 6]])
        self.assertTrue(valid[0]); self.assertFalse(valid[1]); self.assertTrue(np.isnan(median[1]))
        self.assertEqual(median[0], 2.5); self.assertEqual(spread[0], 1.5)
        self.assertIsNone(personal_components(np.ones(4), np.zeros((4, 4)), 1e-8))
        with self.assertRaises(ValueError):
            reference_draws(['same']*40, 10, 1)
        with self.assertRaises(ValueError):
            network_changes(np.ones((10, 4)), np.ones((2, 4)))

    def test_matched_features_and_prefix_are_outcome_blind_and_causal(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = [{'record_id': f'{p}-{i}', 'participant_id': p,
                 'issued_at': (start+timedelta(days=i)).isoformat(),
                 'observed_at': (start+timedelta(days=i)).isoformat(), 'outcome': {'value': i % 2}}
                for p in ('a', 'b') for i in range(6)]
        chosen = forecast_prefix(rows, 5)
        changed = deepcopy(rows)
        for row in changed:
            row['outcome'] = {'value': 1-row['outcome']['value']}
        self.assertEqual([r['record_id'] for r in chosen], [r['record_id'] for r in forecast_prefix(changed, 5)])
        metadata = [{k: r[k] for k in ('record_id', 'participant_id', 'issued_at', 'observed_at')} for r in chosen]
        values = {r['record_id']: [float(i)] for i, r in enumerate(chosen)}
        base, columns, lineage = causal_history(metadata, values, ['x'], 3)
        cache = {'values': {'history_baseline': base}, 'columns': {'history_baseline': columns}, 'lineage': lineage,
                 'current_valid': {'history_baseline': {rid: True for rid in base}}, 'smooth': {}}
        summaries = {name: {'median': {rid: float(i+1) for i, rid in enumerate(base)},
                            'iqr': {rid: .1 for rid in base}, 'valid': {rid: True for rid in base}}
                     for name in ('dnb', 'deviation', 'network_ratio')}
        alternate = deepcopy(cache)
        settings = {'history_days': 3, 'half_lives_days': [0, 1, 2, 4]}
        augment_features(chosen, cache, summaries, settings)
        modified = deepcopy(summaries); modified['dnb']['median']['a-4'] = 1e9
        modified['dnb']['median']['b-0'] = 1e9
        augment_features(chosen, alternate, modified, settings)
        for rid in base:
            control = cache['values']['history_robust_control'][rid]
            combined = cache['values']['history_robust_dnb'][rid]
            np.testing.assert_allclose(control, combined[:len(control)], equal_nan=True)
        for rid in ('a-0', 'a-1', 'a-2', 'a-3'):
            np.testing.assert_allclose(cache['values']['history_robust_dnb'][rid],
                                       alternate['values']['history_robust_dnb'][rid], equal_nan=True)

    def test_config_does_not_ignore_unsupported_choices(self):
        config = read_json(Path(__file__).resolve().parents[2] / 'configs/reference_stability.json')
        check_stability_config(config)
        for key, value in (('replicates', True), ('replicates', 1), ('summary_quantiles', [.1, .5, .9]),
                           ('selection', 'test_accuracy')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_stability_config({**config, key: value})


if __name__ == '__main__':
    unittest.main()
