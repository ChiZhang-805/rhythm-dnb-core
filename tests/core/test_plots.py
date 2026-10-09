"""Diagnostic exports retain missingness, signed correlation and exclusive output paths."""

from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rhythm_dnb.research.plots import plot_network_diagnostics, plot_pair_scatter, _subplots


class PlotTests(unittest.TestCase):
    def test_background_export_does_not_require_pyplot_or_a_gui(self):
        figure, _ = _subplots()
        self.assertIsNone(figure.canvas.manager)
        with tempfile.TemporaryDirectory() as folder:
            with ThreadPoolExecutor(max_workers=1) as worker:
                result = worker.submit(plot_pair_scatter, [[1., 2.], [2., 4.]], ('a', 'b'), [('a', 'b')],
                                       Path(folder) / 'background', provenance_label='Artificial software test').result()
            self.assertTrue(Path(result['figure']).is_file())

    def test_sparse_network_is_reported_invalid_without_a_false_score(self):
        with tempfile.TemporaryDirectory() as folder:
            result = plot_network_diagnostics({'unit-test-only': [[1., 2., 3.]]}, ('a', 'b', 'c'),
                                              ('a', 'b'), Path(folder) / 'network', provenance_label='Artificial software test')
            self.assertFalse(result['components']['unit-test-only']['valid'])
            self.assertIsNone(result['components']['unit-test-only']['score'])
            self.assertTrue(Path(result['figure']).exists())

    def test_negative_scatter_uses_only_finite_pairs_and_preserves_exports(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'scatter'
            values = [[1., -1.], [2., -2.], [3., -3.], [np.nan, 7.]]
            result = plot_pair_scatter(values, ('a', 'b'), [('a', 'b')], path, provenance_label='Artificial software test')
            self.assertEqual(result['pairs'][0]['n'], 3)
            self.assertAlmostEqual(result['pairs'][0]['correlation'], -1.)
            with self.assertRaises(FileExistsError):
                plot_pair_scatter(values, ('a', 'b'), [('a', 'b')], path, provenance_label='Artificial software test')
