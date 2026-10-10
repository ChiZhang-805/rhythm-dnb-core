"""Protect person isolation, chronological discovery and exact pooling of frozen warning policies."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from rhythm_dnb.contracts import EvaluationDay
from rhythm_dnb.research.evaluate import event_metrics
from tools.run_dnbr_cross_validation import validate_people, development_pairs, policy_decisions, ROTATIONS


class CrossValidationTests(unittest.TestCase):
    def test_all_people_are_tested_once_and_text_exposure_is_rejected(self):
        # PSEUDOCODE: exercise disjoint roles, repeated people and text development leakage.
        roles = ('reference', 'train', 'validation', 'test')
        data = {k: [{'participant_id': k, 'record_id': k, 'split': k}] for k in roles}
        data['text-development'] = {'manifest': {'excluded_dnb_people': ['rhythm:' + k for k in roles]}, 'rows': []}
        self.assertEqual(set(validate_people(data)), set(roles))
        self.assertEqual(sorted(r[2] for r in ROTATIONS), sorted(roles[1:]))
        for r in ROTATIONS:
            self.assertEqual(set(r), set(roles[1:]))
        bad = deepcopy(data)
        bad['test'][0]['participant_id'] = 'train'
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            validate_people(bad)
        bad = deepcopy(data)
        bad['text-development']['rows'] = [{'participant_id': 'rhythm:test'}]
        with self.assertRaisesRegex(ValueError, 'contains DNB'):
            validate_people(bad)

    def test_pairing_excludes_event_and_late_records_without_changing_input(self):
        # PSEUDOCODE: ensure only the earliest and final eligible pre-event observation enter discovery.
        start = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        rows = [{'record_id': str(i), 'participant_id': 'person', 'issued_at': (start + timedelta(days=i)).isoformat(),
                 'outcome': {'event_onset_at': (start + timedelta(days=5)).isoformat()}} for i in range(8)]
        original = deepcopy(rows)
        self.assertEqual(development_pairs(rows[::-1], {'discovery_pre_event_lead_hours': 48}),
                         [{'participant_id': 'person', 'stable_record': '0', 'pre_event_record': '3'}])
        self.assertEqual(rows, original)
        rows[-1]['outcome']['event_onset_at'] = start.isoformat()
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            development_pairs(rows, {'discovery_pre_event_lead_hours': 48})

    def test_pooled_decisions_preserve_threshold_equality_gaps_and_cooldown(self):
        # PSEUDOCODE: compare original continuous-score policies with pooled decisions including abstention.
        start = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        scores = [2., 3., 3., None, 4., 4., 4., 0., 4., 4.]
        rows = tuple(EvaluationDay('person', start + timedelta(days=i), score, 0) for i, score in enumerate(scores))
        original = event_metrics(rows, 2., cooldown_days=3)
        pooled = event_metrics(policy_decisions(rows, 2.), .5, cooldown_days=3)
        for key in ('risk_confusion', 'risk_accuracy', 'classification_coverage', 'false_alarms', 'unknown_alarms'):
            self.assertEqual(pooled[key], original[key])
        self.assertEqual(policy_decisions(rows, 2.)[0].score, 0.)
        self.assertIsNone(policy_decisions(rows, 2.)[3].score)
        unavailable = event_metrics(policy_decisions(rows, None), .5)
        self.assertIsNone(unavailable['risk_accuracy'])
        self.assertEqual(unavailable['false_alarms'], 0)


if __name__ == '__main__':
    unittest.main()
