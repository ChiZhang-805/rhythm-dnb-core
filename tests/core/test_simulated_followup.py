"""Check identity, clock, endpoint and censoring boundaries of authored follow-up."""

import copy
from datetime import datetime
import unittest

from rhythm_dnb.research.simulated_followup import (
    FEATURES, SETTINGS, SCENARIOS, generate_sequence, select_backgrounds,
)


def source_record(person, source='public', day='2020-01-01', **changes):
    return {'record_id': person+'-'+day, 'participant_id': person, 'source_dataset': source,
            'observed_at': day+'T12:00:00+00:00', 'split': 'train', 'sleep_start_hour': 23.,
            'sleep_end_hour': 7., 'sleep_duration_h': 8., 'exercise_hour': 17.,
            'meal_interval_cv': .1, 'exercise_minutes': 35., 'resting_hr_bpm': 60.,
            'screen_time_min': 120., **changes}


def background():
    source = [source_record('parent')]
    return select_backgrounds(source, [{'record_id': source[0]['record_id'], 'candidate_state': None}])[0]


class SimulatedFollowupTests(unittest.TestCase):
    def test_keeps_simulated_and_eligible_original_people_out_of_expansion(self):
        rows = [source_record('short'), source_record('usable'), source_record('ref', 'SIMULATED-RHYTHM')]
        annotations = [{'record_id': r['record_id'], 'candidate_state': 0 if r['participant_id']=='usable' else None} for r in rows]
        selected = select_backgrounds(rows, annotations)
        self.assertEqual([b['parent_participant_id'] for b in selected], ['short'])
        with self.assertRaises(ValueError):
            select_backgrounds(rows, annotations + annotations[:1])

    def test_original_identity_is_not_split_or_counted_twice_across_sources(self):
        rows = [source_record('same', 'phase-a'), source_record('same', 'phase-b', day='2020-03-01')]
        annotations = [{'record_id': r['record_id'], 'candidate_state': None} for r in rows]
        selected = select_backgrounds(rows, annotations)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['anchor_record_ids'], [rows[0]['record_id']])

    def test_reproducible_complete_new_timeline_without_mutating_background(self):
        b = background(); old = copy.deepcopy(b)
        rows, evidence = generate_sequence(b)
        self.assertEqual(b, old)
        self.assertEqual((rows, evidence), generate_sequence(b))
        self.assertEqual(len(rows), 84)
        self.assertEqual([r['day_index'] for r in rows], list(range(84)))
        self.assertEqual(len({r['record_id'] for r in rows}), 84)
        for r in rows:
            self.assertNotIn('observed_at', r)
            self.assertEqual(set(r['features']), set(FEATURES))
            self.assertTrue(0 <= r['features']['sleep_midpoint_h'] < 24)
            self.assertTrue(0 <= r['features']['exercise_hour'] < 24)
            self.assertTrue(all(v is not None for v in r['features'].values()))
        self.assertEqual(evidence['original_followup_outcome'], None)

    def test_stable_late_clock_is_not_itself_a_disruption_label(self):
        b = background(); b['scenario'] = 'stable'
        regular, a = generate_sequence(b)
        b['scenario'] = 'stable_late_phase'
        late, z = generate_sequence(b)
        self.assertEqual([r['candidate_state'] for r in regular], [r['candidate_state'] for r in late])
        self.assertEqual(a['event_in_followup'], z['event_in_followup'])
        for x, y in zip(regular, late):
            self.assertAlmostEqual((y['features']['sleep_midpoint_h']-x['features']['sleep_midpoint_h']) % 24, 3)

    def test_authored_truth_and_candidate_rule_are_distinct_and_ignore_warning_fields(self):
        b = background(); baseline = generate_sequence(b)
        b.update(warning=1, reference_event_in_followup=1, event_onset_at='2020-01-01')
        altered = generate_sequence(b)
        self.assertEqual(altered[0], baseline[0])
        for case in SCENARIOS:
            b['scenario'] = case
            rows, e = generate_sequence(b)
            self.assertEqual(e['event_in_followup'], int(case in ('gradual_irregularity','abrupt_irregularity','coupled_fluctuation')))
            self.assertEqual(bool(e['candidate_events']), any(r['candidate_state']==1 for r in rows))
            self.assertEqual(e['event_in_followup'], int(any(r['regime_state']==1 for r in rows)))
            if e['event_in_followup']:
                self.assertLess(e['first_onset_date'], e['first_confirmed_date'])
            else:
                self.assertIsNone(e['first_onset_date'])
                self.assertIsNone(e['first_confirmed_date'])
        b['scenario'] = 'transient_change'
        rows, e = generate_sequence(b)
        self.assertTrue(e['candidate_events'])
        self.assertEqual(e['event_in_followup'], 0)

    def test_changing_candidate_rule_does_not_rewrite_reference_outcomes(self):
        b = background(); b['scenario'] = 'gradual_irregularity'
        a, first = generate_sequence(b)
        cfg = copy.deepcopy(SETTINGS); cfg['outcome_policy']['quantile'] = .5
        z, second = generate_sequence(b, cfg)
        self.assertEqual(first['first_onset_date'], second['first_onset_date'])
        self.assertEqual([r['future_event_7d'] for r in a], [r['future_event_7d'] for r in z])
        self.assertEqual([r['features'] for r in a], [r['features'] for r in z])

    def test_future_labels_never_turn_baseline_postevent_or_censoring_into_negatives(self):
        for case in SCENARIOS:
            b = background(); b['scenario'] = case
            rows, e = generate_sequence(b)
            onset = datetime.fromisoformat(e['first_onset_date']).date() if e['first_onset_date'] else None
            for r in rows:
                d = datetime.fromisoformat(r['simulated_at']).date()
                if r['day_index'] < 28 or onset and d >= onset:
                    self.assertIsNone(r['future_event_7d'])
                elif r['future_event_7d'] == 1:
                    self.assertTrue(1 <= (onset-d).days <= 7)
                elif r['future_event_7d'] == 0:
                    self.assertLess(r['day_index'] + 9, 84)

    def test_future_days_do_not_change_earlier_measurements(self):
        b = background(); a, _ = generate_sequence(b)
        cfg = copy.deepcopy(SETTINGS); cfg['days'] = 90
        longer, _ = generate_sequence(b, cfg)
        self.assertEqual([r['features'] for r in a], [r['features'] for r in longer[:84]])


if __name__ == '__main__':
    unittest.main()
