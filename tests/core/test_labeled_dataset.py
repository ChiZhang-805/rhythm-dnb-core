"""Protect data completion from invented values, overwritten labels, and clock errors."""

import math
import unittest

from rhythm_dnb.research.labeled_dataset import (
    OBJECTIVE, TEXT, hospital_cross_checks, restore, simulated_clocks, validate_input_columns,
)


class LabeledDatasetTests(unittest.TestCase):
    def test_simulated_clocks_cross_midnight(self):
        self.assertEqual(simulated_clocks(3, 8), ('23:00:00.000000', '07:00:00.000000'))
        self.assertEqual(simulated_clocks(23, 6), ('20:00:00.000000', '02:00:00.000000'))

    def test_invalid_clock_inputs_are_rejected(self):
        for args in [(24, 8), (-1, 8), (3, 0), (3, 24), (math.nan, 8)]:
            with self.assertRaises(ValueError):
                simulated_clocks(*args)

    def test_completion_preserves_unknown_and_zero(self):
        row = {'记录编号': 'test', 'known_zero': 0, 'unknown': None, 'empty': None}
        changes = []
        restore(row, 'known_zero', 0, changes, 'source')
        restore(row, 'unknown', None, changes, 'source')
        restore(row, 'empty', 0, changes, 'source')
        self.assertIsNone(row['unknown'])
        self.assertEqual(row['empty'], 0)
        self.assertEqual(len(changes), 1)

    def test_conflicting_source_cannot_overwrite_label(self):
        row = {'记录编号': 'test', 'label': 1}
        with self.assertRaises(ValueError):
            restore(row, 'label', 0, [], 'source')
        self.assertEqual(row['label'], 1)

    def test_blood_oxygen_conflict_is_flagged_without_repair(self):
        values = {'spO2_min_pct': 98, 'spO2_pct': 96}
        self.assertEqual(hospital_cross_checks(values), [('spO2_min_pct', 'spO2_pct', '最低血氧高于平均血氧')])
        self.assertEqual(values, {'spO2_min_pct': 98, 'spO2_pct': 96})

    def test_input_views_accept_symptoms_and_reject_future_outcomes(self):
        columns = ['record_id', 'participant_id', 'sequence_id', 'dataset', 'split', 'observed_at', 'issued_at']
        columns += list(OBJECTIVE) + ['text_' + k for k in TEXT]
        self.assertIn('text_sleep_onset_difficulty', columns)
        validate_input_columns('dnb_text_inputs', columns)
        for leaked in ('label', 'event_onset_at', 'DNB预警'):
            with self.assertRaises(ValueError):
                validate_input_columns('dnb_text_inputs', columns + [leaked])


if __name__ == '__main__':
    unittest.main()
