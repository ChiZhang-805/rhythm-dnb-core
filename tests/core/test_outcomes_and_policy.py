"""Endpoint confirmation, censoring and alarm-state timing."""

from datetime import date, datetime, timedelta, timezone
from dataclasses import replace
import unittest
from rhythm_dnb.contracts import OutcomeAssessment, OutcomeEvent, AlarmState, EvaluationDay
from rhythm_dnb.outcomes.criteria import personal_anchor, fit_criteria, assess_day, weighted_quantile
from rhythm_dnb.outcomes.events import detect_events
from rhythm_dnb.outcomes.labels import future_label
from rhythm_dnb.warning.policy import advance
from rhythm_dnb.research.evaluate import event_metrics, cluster_intervals

UTC = timezone.utc


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.start = date(2025, 1, 1)
        self.t = datetime(2025, 1, 1, 12, tzinfo=UTC)
        self.domains = {'S1': 1., 'E1': 1., 'A1': .3}
        self.observed = {self.start + timedelta(days=i) for i in range(11)}

    def test_baseline_uses_valid_rolling_endpoints(self):
        values = {self.start + timedelta(days=i): self.domains for i in range(14)}
        self.assertEqual(personal_anchor(values, self.start), self.domains)
        for i in (6, 7, 8):
            values[self.start + timedelta(days=i)] = {}
        self.assertIsNone(personal_anchor(values, self.start))

    def test_person_equal_weight_and_dual_threshold(self):
        people = [{'participant_id': str(i), 'stable': True, 'evidence_id': 'independent',
                   'available_at': self.t, 'anchor': {'S1': 0., 'E1': 0., 'A1': 0.},
                   'values': [self.domains] * (1 if i else 100)} for i in range(3)]
        criteria = fit_criteria(people, self.t, minimum_people=3)
        self.assertEqual(criteria['thresholds']['S1']['absolute'], 1)
        result = assess_day('p', self.start, {'S1': 3., 'E1': 3., 'A1': .4}, self.domains, criteria, self.t)
        self.assertEqual(result.abnormal_domains, ('S1', 'E1'))
        result = assess_day('p', self.start, {'S1': 3., 'E1': 3.}, self.domains, criteria, self.t)
        self.assertFalse(result.evaluable)
        self.assertEqual(weighted_quantile([0, 10, 10], [1., .5, .5], .5), 0)

    def test_events_backdate_only_after_three_days(self):
        rows = [OutcomeAssessment('p', self.start + timedelta(days=i), ('S1', 'E1'), True, self.t + timedelta(days=i + 1)) for i in range(3)]
        self.assertEqual(detect_events(rows[:2], 'UTC'), ())
        event = detect_events(rows, 'UTC')[0]
        self.assertEqual(event.onset, datetime(2025, 1, 2, 4, tzinfo=UTC))
        self.assertEqual(event.confirmed_at, self.t + timedelta(days=3))
        self.assertEqual(detect_events([rows[0], replace(rows[1], evaluable=False, abnormal_domains=()), rows[2]], 'UTC'), ())

    def test_gaps_do_not_join_runs(self):
        rows = [OutcomeAssessment('p', self.start + timedelta(days=i), ('S1', 'A1'), True, self.t + timedelta(days=i + 1)) for i in (0, 2, 3)]
        self.assertEqual(detect_events(rows, 'UTC'), ())

    def test_exact_lead_and_horizon_boundaries(self):
        for days, expected in [(1, 1), (7, 1), (8, 0)]:
            event = OutcomeEvent('p', self.t + timedelta(days=days), self.t + timedelta(days=days + 2))
            result = future_label(self.t, [event], followup_end=self.t + timedelta(days=10), observed_days=self.observed)
            self.assertEqual(result.value, expected)
        early = OutcomeEvent('p', self.t + timedelta(hours=23), self.t + timedelta(days=3))
        self.assertIsNone(future_label(self.t, [early], followup_end=self.t + timedelta(days=10), observed_days=self.observed).value)

    def test_censor_and_gap_not_negative(self):
        self.assertIsNone(future_label(self.t, [], followup_end=self.t + timedelta(days=8), observed_days=self.observed).value)
        incomplete = self.observed - {self.start + timedelta(days=5)}
        self.assertIsNone(future_label(self.t, [], followup_end=self.t + timedelta(days=10), observed_days=incomplete).value)
        event = OutcomeEvent('p', self.t - timedelta(days=1), self.t + timedelta(days=1))
        self.assertIsNone(future_label(self.t, [event], followup_end=self.t + timedelta(days=10), observed_days=self.observed).value)


class PolicyTests(unittest.TestCase):
    def test_missing_breaks_persistence_and_returns_unknown(self):
        day = date(2025, 1, 1); state = AlarmState()
        values = []
        for i, score in enumerate([3., None, 3., 3.]):
            warning, _, state = advance(state, 'p', 'b', day + timedelta(days=i), score, 2)
            values.append(warning)
        self.assertEqual(values, [0, None, 0, 1])

    def test_cooldown_and_replay(self):
        day = date(2025, 1, 1); state = AlarmState(); alarms = []
        for i in range(10):
            warning, _, state = advance(state, 'p', 'b', day + timedelta(days=i), 3., 2.)
            if warning:
                alarms.append(i)
        self.assertEqual(alarms, [1, 8])
        with self.assertRaises(ValueError):
            advance(state, 'p', 'b', day + timedelta(days=9), 3, 2)
        with self.assertRaises(ValueError):
            advance(state, 'other', 'b', day + timedelta(days=10), 3, 2)

    def test_gap_threshold_equality_and_uncalibrated(self):
        state = AlarmState('p', 'b', date(2025, 1, 1), None, 1)
        self.assertEqual(advance(state, 'p', 'b', date(2025, 1, 3), 4, 2)[0], 0)
        self.assertEqual(advance(AlarmState(), 'p', 'b', date(2025, 1, 1), 2, 2)[0], 0)
        self.assertIsNone(advance(AlarmState(), 'p', 'b', date(2025, 1, 1), 2, None)[0])
        with self.assertRaisesRegex(ValueError, 'identities'):
            advance(replace(state, participant_id='', bundle_id=''), 'other', 'b', date(2025, 1, 2), 4, 2)

    def test_event_metrics_include_unscorable_events(self):
        t = datetime(2025, 1, 1, 12, tzinfo=UTC)
        onset = t + timedelta(days=5)
        rows = [EvaluationDay('a', t + timedelta(days=i), 5, 1, onset) for i in range(3)]
        rows += [EvaluationDay('b', t + timedelta(days=i), None, 1, onset) for i in range(3)]
        rows += [EvaluationDay('c', t + timedelta(days=i), 5, 0) for i in range(3)]
        result = event_metrics(rows, 2)
        self.assertEqual(result['event_sensitivity'], .5)
        self.assertEqual(result['false_alarms'], 1)
        self.assertEqual(result['valid_risk_days'], 6)
        self.assertEqual(result['false_alarms_per_30_days'], 5)
        self.assertEqual(cluster_intervals(rows, 2, repetitions=5)['event_sensitivity']['requested_replicates'], 5)

    def test_registered_event_with_no_forecasts_stays_in_denominator(self):
        t = datetime(2025, 1, 1, 12, tzinfo=UTC)
        event = OutcomeEvent('unobserved-person', t + timedelta(days=5), t + timedelta(days=7))
        result = event_metrics([], 2, events=[event])
        self.assertEqual(result['event_sensitivity'], 0)
        self.assertEqual(result['events'], 1)
