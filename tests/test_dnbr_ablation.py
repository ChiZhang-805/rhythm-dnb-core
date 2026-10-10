"""Check paired uncertainty and input identity for the DNB network-contribution experiment."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rhythm_dnb.contracts import EvaluationDay
from tools.run_dnbr_ablation import matrix, paired_accuracy_interval


class AblationTests(unittest.TestCase):
    def test_paired_people_and_known_improvement(self):
        # PSEUDOCODE: compare perfect predictions to inverted ones; duplicate days cannot multiply the sample size.
        rows = [EvaluationDay(str(p), datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=d),
                              float(p % 2), p % 2) for p in range(20) for d in range(3)]
        wrong = [replace(d, score=1 - d.score) for d in rows]
        equal = paired_accuracy_interval(rows, rows, 51, 100)
        better = paired_accuracy_interval(rows, wrong, 51, 100)
        np.testing.assert_allclose(equal['percentile_95_difference'], [0, 0])
        np.testing.assert_allclose(better['percentile_95_difference'], [1, 1])
        with self.assertRaises(ValueError):
            paired_accuracy_interval(rows + rows[:1], wrong + wrong[:1], 51, 100)
        with self.assertRaises(ValueError):
            paired_accuracy_interval(rows, list(reversed(wrong)), 51, 100)

    def test_abstention_is_not_selected_away(self):
        # PSEUDOCODE: incomplete coverage must not create an artificially favorable paired accuracy.
        rows = [EvaluationDay('a', datetime(2026, 1, 1, tzinfo=timezone.utc), 0., 0)]
        result = paired_accuracy_interval(rows, [replace(rows[0], score=None)], 51, 100)
        self.assertIsNone(result['percentile_95_difference'])

    def test_matrix_orientation_and_ambiguous_records(self):
        # PSEUDOCODE: preserve mixed feature signs and record order; reject duplicated identities.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'matrix.csv'
            path.write_text('feature,a,b\nsleep,-1,2\nfood,3,4\n', encoding='utf-8')
            features, records, values = matrix(path)
            self.assertEqual(features, ['sleep', 'food'])
            self.assertEqual(records, ['a', 'b'])
            np.testing.assert_array_equal(values, [[-1, 3], [2, 4]])
            path.write_text('feature,a,a\nsleep,1,2\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                matrix(path)


if __name__ == '__main__':
    unittest.main()
