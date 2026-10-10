"""Guard explanatory counts and independent DNB arithmetic without changing scientific selection."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

from rhythm_dnb.dnb.classic import dnb_components


def load_tool(name='diagnose_discovery'):
    # PSEUDOCODE: load the standalone diagnostic without importing it into the frozen scientific package.
    path = Path(__file__).resolve().parents[2] / ('tools/' + name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class DiscoveryDiagnosticTests(unittest.TestCase):
    def test_independent_components_match_signed_and_permuted_data(self):
        rng = np.random.default_rng(71)
        matrix = rng.normal(size=(37, 5)); matrix[:, 2] = -3 * matrix[:, 0] + matrix[:, 2]
        names = ['e', 'd', 'c', 'b', 'a']; module = ['a', 'c', 'e']
        observed = load_tool().manual_components(matrix, names, module)
        known = dnb_components(matrix, names, module)
        np.testing.assert_allclose(observed, [known[k] for k in ('sd_in', 'pcc_in', 'pcc_out')], atol=1e-12)

    def test_counts_distinguish_direction_from_reliability(self):
        rows = [dict(stable_components=[1, .2, .4], pre_event_components=[2, .8, .1],
                     stability=.4, max_stat_adjusted_p=.65, all_three_conditions=True, eligible=False),
                dict(stable_components=[1, .2, .4], pre_event_components=[.9, .8, .1],
                     stability=.9, max_stat_adjusted_p=.02, all_three_conditions=False, eligible=False)]
        protocol = dict(epsilon=1e-8, module_stability=.7, discovery_alpha=.05)
        output = load_tool().gate_summary({'candidates': rows}, protocol)
        self.assertEqual(output['all_three'], 1)
        self.assertEqual(output['bootstrap_pass_any'], 1)
        self.assertEqual(output['three_and_bootstrap'], 0)
        self.assertEqual(output['all_gates'], 0)
        rows[0]['eligible'] = True
        with self.assertRaisesRegex(ValueError, 'gate decisions'):
            load_tool().gate_summary({'candidates': rows}, protocol)

    def test_diagnostic_refuses_to_overwrite_existing_directory(self):
        with self.assertRaises(FileExistsError):
            load_tool().diagnose('unread', 'unread', None, Path(__file__).parent)

    def test_minp_matches_literal_empirical_definition_with_ties(self):
        tool = load_tool('check_discovery_power')
        values = np.array([[4., 1.], [4., 2.], [1., 2.], [0., 0.]])
        p = np.array([[np.mean(values[:, j] >= v) for j, v in enumerate(row)] for row in values])
        expected = [np.mean(p.min(axis=1) <= value) for value in p[0]]
        measured = tool.permutation_adjustments(values)
        np.testing.assert_allclose(measured['minP'], expected)
        self.assertTrue(np.all(measured['minP'] >= measured['unadjusted']))
        self.assertTrue(np.all(measured['minP'] > 0))
        zeros = tool.permutation_adjustments(np.zeros((9, 3)))
        np.testing.assert_equal(zeros['minP'], np.ones(3))
        np.testing.assert_equal(zeros['maxT'], np.ones(3))

    def test_minp_is_invariant_to_candidate_scales(self):
        tool = load_tool('check_discovery_power')
        values = np.random.default_rng(123).uniform(size=(50, 4))
        a = tool.permutation_adjustments(values)
        b = tool.permutation_adjustments(values * [1, 2, 100, .01])
        np.testing.assert_equal(a['minP'], b['minP'])
        with self.assertRaises(ValueError):
            tool.permutation_adjustments([[0, 1], [2, np.nan]])
