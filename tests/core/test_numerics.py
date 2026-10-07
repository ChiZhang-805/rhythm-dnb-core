"""Independent formula properties and legacy numerical parity."""

import unittest
import json
from pathlib import Path
import numpy as np
from rhythm_dnb.dnb.classic import dnb_components
from rhythm_dnb.dnb.single_sample import sdnb_components, sample_network, discover_sample_modules
from rhythm_dnb.dnb.modules import candidate_modules
from rhythm_dnb.measures.scaling import fit_scaler, transform, circular_summary


class NumericsTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(77).normal(size=(60, 5))
        self.names = tuple('abcde')

    def test_classic_invariant_to_location_sign_and_column_order(self):
        a = dnb_components(self.x, self.names, ('a', 'b'))
        b = dnb_components(self.x * [-1, 1, -1, 1, -1] + 40, self.names, ('a', 'b'))
        c = dnb_components(self.x[:, ::-1], self.names[::-1], ('b', 'a'))
        self.assertAlmostEqual(a['score'], b['score'], places=12)
        self.assertEqual(a, c)

    def test_scale_not_removed_within_window(self):
        a = dnb_components(self.x, self.names, ('a', 'b'))
        b = dnb_components(self.x * 3, self.names, ('a', 'b'))
        self.assertAlmostEqual(b['score'], 3 * a['score'])

    def test_exact_analytic_signed_group(self):
        t = np.arange(-4, 5, dtype=float); t /= t.std(ddof=1)
        z = t * t; z -= z.mean(); z /= z.std(ddof=1)
        x = np.column_stack([t, -t, .6 * t + .8 * z])
        result = dnb_components(x, ('a', 'b', 'c'), ('a', 'b'))
        self.assertAlmostEqual(result['sd_in'], 1)
        self.assertAlmostEqual(result['pcc_in'], 1)
        self.assertAlmostEqual(result['pcc_out'], .6)
        self.assertAlmostEqual(result['score'], 1 / .6)

    def test_paper_pair_convention_and_reference_immutable(self):
        reference = self.x.copy(); target = np.array([4., -4, 2, 0, 1])
        a = sdnb_components(target, reference, self.names, ('a', 'b'))
        b = sdnb_components(target, reference, self.names, ('a', 'b'), pair_convention='paper_k_squared')
        self.assertAlmostEqual(b['score'], a['score'] / 2)
        np.testing.assert_array_equal(reference, self.x)

    def test_joint_missing_rows_and_constants_abstain(self):
        x = self.x.copy(); x[:20, 4] = np.nan
        result = dnb_components(x, self.names, ('a', 'b'))
        self.assertEqual(result['n_valid'], 40)
        x[:, 4] = 1
        self.assertFalse(dnb_components(x, self.names, ('a', 'b'))['valid'])
        x[:, 4] = np.inf
        self.assertIsNone(dnb_components(x, self.names, ('a', 'b'))['score'])

    def test_strict_type_and_complement(self):
        for value in (True, '4', 2 + 1j):
            with self.subTest(value=value), self.assertRaises(ValueError):
                x = self.x.astype(object); x[0, 0] = value
                dnb_components(x, self.names, ('a', 'b'))
        with self.assertRaises(ValueError):
            dnb_components(self.x, self.names, self.names)

    def test_migration_golden_outputs(self):
        fixture = json.loads((Path(__file__).resolve().parents[1] / 'fixtures/migration.json').read_text(encoding='utf-8'))
        for row in fixture['kernel']:
            x = np.random.default_rng(row['seed']).normal(size=(30, 5))
            results = {'classic': dnb_components(x, self.names, ('a', 'b', 'd')),
                       'single_sample': sdnb_components(x[0], x[1:], self.names, ('a', 'b'), pair_convention='paper_k_squared')}
            for name, result in results.items():
                self.assertEqual(set(result), set(row[name]))
                for key, expected in row[name].items():
                    with self.subTest(seed=row['seed'], method=name, component=key):
                        if type(expected) is float:
                            np.testing.assert_allclose(result[key], expected, rtol=1e-11, atol=1e-13)
                        elif key == 'module':
                            self.assertEqual(list(result[key]), expected)
                        else:
                            self.assertEqual(result[key], expected)

    def test_search_cardinality_and_budget(self):
        modules = candidate_modules(tuple(f'x{i}' for i in range(12)), method='bounded_exhaustive', sizes=(2, 3, 4))
        self.assertEqual(len(modules), 781)
        with self.assertRaises(ValueError):
            candidate_modules(self.names, method='bounded_exhaustive', maximum_candidates=1)

    def test_clock_scaler_does_not_split_midnight(self):
        x = np.column_stack([[23, 0, 1, 23.5, .5], [3, 2, 1, 0, 4], [0, 1, 3, 4, 2]])
        scaler = fit_scaler(x, ('sleep_midpoint_h', 'b', 'c'))
        z = transform(x, scaler)
        self.assertLess(abs(z[0, 0] - z[1, 0]), 2)
        np.testing.assert_allclose(z.std(axis=0, ddof=1), 1)
        self.assertEqual(circular_summary([0, 6, 12, 18]), (None, None))

    def test_single_sample_exploration_has_signed_edges(self):
        result = sample_network([7, -7, 6, 0, 0], self.x, self.names, deviation_quantile=.3)
        self.assertTrue(any(e['delta'] < 0 for e in result['edges']))
        self.assertTrue(all(e['q'] >= e['p'] - 1e-12 for e in result['edges']))
        modules = discover_sample_modules([7, -7, 6, 0, 0], self.x, self.names, deviation_quantile=.3)
        self.assertIn('best', modules)

    def test_exploration_rejects_covariance_overflow_like_primary_sdnb(self):
        target = [1e308, -1e308, 1e308, 0., 0.]
        primary = sdnb_components(target, self.x, self.names, ('a', 'b'))
        self.assertEqual(primary['reason'], 'nonfinite_sample_statistics')
        for method in (sample_network, discover_sample_modules):
            with self.subTest(method=method.__name__), self.assertRaisesRegex(ValueError, 'nonfinite_sample_statistics'):
                method(target, self.x, self.names)
        np.testing.assert_array_equal(self.x, np.random.default_rng(77).normal(size=(60, 5)))
