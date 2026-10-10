"""Candidate annotations must preserve uncertainty and stay independent of warnings."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from rhythm_dnb.research.dataset_annotations import annotate_sequence, annotate_records, daily_values, _spread


class DatasetAnnotationTests(unittest.TestCase):
    def rows(self, days=44):
        start = datetime(2020, 1, 1, 12, tzinfo=timezone.utc)
        return [{'record_id': f'row-{i}', 'participant_id': 'person', 'source_dataset': 'test-source',
                 'observed_at': (start + timedelta(days=i)).isoformat(), 'sleep_start_hour': 22.,
                 'sleep_end_hour': 6., 'exercise_hour': 10., 'exercise_minutes': 30.,
                 'meal_interval_cv': .1} for i in range(days)]

    def worsening(self, days=44):
        rows = self.rows(days)
        for i, row in enumerate(rows[28:], 28):
            row.update(sleep_start_hour=18. if i % 2 else 0., sleep_end_hour=2. if i % 2 else 8., meal_interval_cv=.8)
        return rows

    def test_midnight_and_no_exercise_are_not_fabricated_changes(self):
        self.assertLess(_spread([23.9, .1]), .11)
        row = self.rows(1)[0]
        self.assertEqual(daily_values(row), (2., .1, 10.))
        row['exercise_minutes'] = 0
        self.assertIsNone(daily_values(row)[2])

    def test_missing_and_short_sequences_are_not_normal_labels(self):
        self.assertTrue(all(r['candidate_state'] is None for r in annotate_sequence(self.rows(14))['rows']))
        rows = self.rows()
        for row in rows:
            row['meal_interval_cv'] = None
        result = annotate_sequence(rows)
        self.assertEqual(result['events'], [])
        self.assertTrue(all(r['candidate_state'] is None for r in result['rows']))

    def test_three_days_and_two_domains_required_with_day_precision(self):
        result = annotate_sequence(self.worsening())
        self.assertEqual(len(result['events']), 1)
        event = result['events'][0]
        self.assertEqual(event['onset_date'], '2020-02-04')
        self.assertEqual(event['confirmed_date'], '2020-02-06')
        self.assertEqual(result['rows'][34]['candidate_state'], 1)
        # Meal changes alone do not establish multi-domain worsening.
        one_domain = self.rows()
        for row in one_domain[28:]:
            row['meal_interval_cv'] = .8
        self.assertEqual(annotate_sequence(one_domain)['events'], [])

    def test_gaps_and_unconfirmed_runs_abstain(self):
        rows = self.worsening(37)
        rows = [r for r in rows if r['record_id'] != 'row-35']
        result = annotate_sequence(rows)
        self.assertEqual(result['events'], [])
        self.assertIsNone(result['rows'][-1]['candidate_state'])
        self.assertEqual(result['rows'][-1]['status'], 'persistence_not_confirmed')

    def test_truth_warnings_and_post_baseline_extremes_do_not_refit_thresholds(self):
        original = self.worsening()
        edited = deepcopy(original)
        for row in edited:
            row.update(rhythm_state_gt=1, warning=1, dnb_score=999., event_onset_at='1990-01-01')
        self.assertEqual(annotate_sequence(original), annotate_sequence(edited))
        before = annotate_sequence(original)['baseline']
        edited[-1]['meal_interval_cv'] = 100.
        self.assertEqual(before, annotate_sequence(edited)['baseline'])
        self.assertEqual(before['baseline_complete_days'], 28)

    def test_duplicate_days_and_mixed_people_rejected(self):
        rows = self.rows()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            annotate_sequence(rows + [rows[0]])
        rows[0]['participant_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'one source/person'):
            annotate_sequence(rows)

    def test_prevalent_diagnosis_and_reference_visit_have_no_incident_onset(self):
        row = self.rows(1)[0]
        row.update(source_dataset='DelSoM-baseline', rhythm_state_gt=1)
        annotations, _ = annotate_records([row])
        self.assertEqual(annotations[0]['reference_scope'], 'prevalent_sleep_phase_diagnosis')
        self.assertIsNone(annotations[0]['reference_event_in_followup'])
        self.assertIsNone(annotations[0]['reference_onset_at'])
        row.update(source_dataset='SIMULATED-RHYTHM', split='reference', rhythm_state_gt=0, event_onset_at=None)
        annotations, _ = annotate_records([row])
        self.assertIsNone(annotations[0]['reference_event_in_followup'])
        self.assertIsNone(annotations[0]['candidate_state'])


if __name__ == '__main__':
    unittest.main()
