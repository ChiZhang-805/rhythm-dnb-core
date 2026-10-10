"""Check time leakage, circular clocks and calibration against the production warning policy."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

import numpy as np

from rhythm_dnb.contracts import EvaluationDay, OutcomeEvent
from rhythm_dnb.research.evaluate import event_metrics
from rhythm_dnb.research.expanded_warning import sequence_features, threshold_table, policy
from rhythm_dnb.research.simulated_followup import FEATURES


CONFIG = {'baseline_days': 28, 'epsilon': 1e-8, 'history_windows': [7, 14, 28],
          'rolling_windows': [7, 14, 28], 'consecutive': 2, 'cooldown_days': 7,
          'horizon_days': 7, 'min_lead_hours': 24, 'confirmation_days': 2,
          'max_false_alarms_per_30_days': 1.,
          'threshold_selection': 'calibration_balanced_accuracy_with_preserved_alarm_budget'}


class ExpandedWarningTests(unittest.TestCase):
    def records(self):
        rng = np.random.default_rng(3301); start = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
        rows = []
        for i in range(50):
            values = dict(zip(FEATURES, rng.normal(5, .3, len(FEATURES))))
            values['sleep_midpoint_h'] = float((23.95 + rng.normal(0, .15)) % 24)
            stamp = (start+timedelta(days=i)).isoformat()
            rows.append(dict(record_id=f'p-{i}', participant_id='p', day_index=i,
                             observed_at=stamp, issued_at=stamp, features=values))
        return rows

    def test_future_changes_and_outcome_fields_cannot_change_earlier_features(self):
        rows = self.records(); other = deepcopy(rows)
        for row in other:
            row['future_event_7d'] = 1; row['scenario'] = 'cheat'
            if row['day_index'] >= 45:
                row['features'] = {k: float(v*3) for k, v in row['features'].items()}
        left, _ = sequence_features(rows, CONFIG); right, _ = sequence_features(other, CONFIG)
        for i in range(28, 45):
            for key in ('control', 'network', 'personal_dnb'):
                np.testing.assert_allclose(left[f'p-{i}'][key], right[f'p-{i}'][key], atol=0, rtol=0)

    def test_midnight_clock_shift_preserves_scores_and_baseline_scale(self):
        rows = self.records(); shifted = deepcopy(rows)
        for row in shifted:
            for name in ('sleep_midpoint_h', 'exercise_hour'):
                row['features'][name] = (row['features'][name] + 8) % 24
        a, ca = sequence_features(rows, CONFIG); b, cb = sequence_features(shifted, CONFIG)
        np.testing.assert_allclose(ca['scale'], cb['scale'], atol=1e-13)
        self.assertLess(ca['scale'][ca['features'].index('sleep_midpoint_h')], .3)
        for rid in a:
            np.testing.assert_allclose(a[rid]['network'], b[rid]['network'], atol=1e-10)

    def test_all_calibration_thresholds_match_production_replay(self):
        rng = np.random.default_rng(71); start = datetime(2000, 1, 29, 12, tzinfo=timezone.utc)
        rows, events = [], []
        for person in range(5):
            onset = start+timedelta(days=25) if person % 2 else None
            if onset:
                events.append(OutcomeEvent(str(person), onset, onset+timedelta(days=2)))
            for i in range(42):
                issued = start+timedelta(days=i)
                label = None if onset and issued >= onset or i > 32 else int(onset is not None and 1 <= (onset-issued).days <= 7)
                score = None if i in (3, 19) else float(np.round(rng.uniform(0, 5), 1))
                rows.append(EvaluationDay(str(person), issued, score, label, onset if label == 1 else None, start+timedelta(days=60)))
        result = threshold_table(rows, CONFIG)
        for candidate in result['candidates']:
            expected = event_metrics(rows, candidate['threshold'], events=events, **policy(CONFIG))
            for key in ('risk_confusion', 'detected_events', 'false_alarms', 'unknown_alarms'):
                self.assertEqual(candidate[key], expected[key])
            self.assertAlmostEqual(candidate['balanced_accuracy'], expected['risk_balanced_accuracy'])

    def test_missing_calendar_and_constant_baseline_fail_explicitly(self):
        rows = self.records()
        with self.assertRaises(ValueError):
            sequence_features(rows[:12]+rows[13:], CONFIG)
        for row in rows:
            row['features']['resting_hr_bpm'] = 65.
        with self.assertRaisesRegex(ValueError, 'Constant baseline'):
            sequence_features(rows, CONFIG)


if __name__ == '__main__':
    unittest.main()
