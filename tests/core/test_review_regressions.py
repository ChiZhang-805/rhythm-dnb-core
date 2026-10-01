"""Scientific failures reproduced during the 2026-10-01 full repository review."""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import tempfile
import unittest
import numpy as np
from rhythm_dnb.config import StudyConfig
from rhythm_dnb.contracts import Observation, Provenance, OutcomeAssessment, OutcomeEvent, EvaluationDay, AlarmState
from rhythm_dnb.provenance import build_lineage, eligible_measurement
from rhythm_dnb.workflows.prepare import prepare_day
from rhythm_dnb.warning.policy import advance
from rhythm_dnb.warning.windows import select_window
from rhythm_dnb.research.evaluate import replay, event_metrics
from rhythm_dnb.research.simulate import _panel
from rhythm_dnb.measures.panel import OBJECTIVE8
from rhythm_dnb.outcomes.events import detect_events
from rhythm_dnb.outcomes.labels import future_label
from rhythm_dnb.io.tabular import read_table
from rhythm_dnb.io.validation import validate_observations
from rhythm_dnb.research.discover import _component_weights, _components, _directional_statistic
from rhythm_dnb.dnb.classic import dnb_components

UTC = timezone.utc


class ReviewRegressions(unittest.TestCase):
    def observation(self, identity, start, end, variable='sleep_episode', value=1., unit='state'):
        # PSEUDOCODE: construct a traceable completed measurement for boundary tests.
        return Observation(identity, 'p', variable, value, unit, start, end, end, 'UTC', Provenance('observed', identity, 'abc'))

    def test_fragment_before_day_is_not_counted_twice(self):
        start = datetime(2025, 1, 1, 22, tzinfo=UTC)
        rows = [self.observation('a', start, start + timedelta(hours=4)),
                self.observation('b', start + timedelta(hours=7), start + timedelta(hours=9)),
                self.observation('main', start, start + timedelta(hours=9), 'main_sleep_period', None, 'interval')]
        totals = []
        for day in (date(2025, 1, 1), date(2025, 1, 2)):
            panel = prepare_day(rows, 'p', day, 'UTC', datetime.combine(day + timedelta(days=1), datetime.min.time(), UTC).replace(hour=12), panel_id='objective8', sleep_complete=True)
            totals.append(next(f.value for f in panel.features if f.name == 'sleep_duration_h'))
        self.assertEqual(totals, [4., 2.])
        self.assertEqual(sum(totals), 6.)

    def test_wake_interval_cannot_be_counted_as_sleep(self):
        start = datetime(2025, 1, 1, 22, tzinfo=UTC)
        row = self.observation('wake', start, start + timedelta(hours=3), value=0.)
        with self.assertRaises(ValueError):
            prepare_day([row], 'p', start.date(), 'UTC', start + timedelta(hours=14), panel_id='objective8', sleep_complete=True)

    def test_mixed_simulation_and_real_lineage_is_ineligible(self):
        p = build_lineage([Provenance('observed', 'real', 'a'), Provenance('synthetic', 'sim', 'b')], 'combined')
        self.assertFalse(eligible_measurement(p))
        self.assertFalse(eligible_measurement(p, simulation=True))

    def test_missing_target_cannot_reuse_old_rolling_score(self):
        last = date(2025, 1, 31)
        panels = [_panel('p', last - timedelta(days=i), [4, 7, 8, 20, 10, .5, 100, 60], OBJECTIVE8) for i in range(1, 28)]
        _, reason = select_window(panels, 'p', 'UTC', last, OBJECTIVE8, datetime(2025, 2, 1, 12, tzinfo=UTC), simulation=True)
        self.assertEqual(reason, 'missing_target_day')

    def test_state_rejects_method_and_timezone_switch(self):
        _, _, state = advance(AlarmState(), 'p', 'bundle', date(2025, 1, 1), 3., 2., context='single_sample:UTC')
        for context in ('rolling:UTC', 'single_sample:Asia/Shanghai'):
            with self.assertRaises(ValueError):
                advance(state, 'p', 'bundle', date(2025, 1, 2), 3., 2., context=context)

    def test_offline_replay_uses_local_forecast_schedule(self):
        zone = ZoneInfo('Pacific/Auckland')
        rows = [EvaluationDay('p', datetime(2025, 1, 1 + i, 12, tzinfo=zone).astimezone(UTC), 3., 0, timezone='Pacific/Auckland') for i in range(2)]
        self.assertEqual([alarm for _, alarm, _ in replay(rows, 2.)], [0, 1])
        with self.assertRaises(ValueError):
            replay([replace(rows[0], timezone='UTC')], 2.)

    def test_positive_labels_must_match_lead_horizon(self):
        t = datetime(2025, 1, 1, 12, tzinfo=UTC)
        for hours in (-24, 1, 169):
            with self.assertRaises(ValueError):
                event_metrics([EvaluationDay('p', t, 3., 1, t + timedelta(hours=hours))], 2.)

    def test_negative_cannot_hide_registered_future_event(self):
        t = datetime(2025, 1, 1, 12, tzinfo=UTC)
        event = OutcomeEvent('p', t + timedelta(days=3), t + timedelta(days=5))
        with self.assertRaises(ValueError):
            event_metrics([EvaluationDay('p', t, 3., 0)], 2., events=[event])

    def test_duplicate_domains_do_not_fake_multidomain_endpoint(self):
        row = OutcomeAssessment('p', date(2025, 1, 1), ('S1', 'S1'), True, datetime(2025, 1, 2, 12, tzinfo=UTC))
        with self.assertRaises(ValueError):
            detect_events([row], 'UTC', persistence=1)

    def test_each_assessment_requires_actual_completion(self):
        t = datetime(2025, 1, 2, 12, tzinfo=UTC)
        rows = [OutcomeAssessment('p', date(2025, 1, 1 + i), ('S1', 'E1'), True, t) for i in range(3)]
        with self.assertRaises(ValueError):
            detect_events(rows, 'UTC')

    def test_negative_label_coverage_includes_next_local_boundary(self):
        t = datetime(2025, 1, 1, 12, tzinfo=ZoneInfo('Pacific/Auckland'))
        days = {t.date() + timedelta(days=i) for i in range(9)}
        args = dict(followup_end=t + timedelta(days=10), timezone='Pacific/Auckland')
        self.assertEqual(future_label(t, [], observed_days=days, **args).value, 0)
        self.assertIsNone(future_label(t, [], observed_days=days - {t.date()}, **args).value)

    def test_numeric_settings_reject_boolean_and_nonfinite(self):
        for key in ('module_stability', 'threshold_quantile', 'epsilon', 'max_false_alarms_per_30_days', 'discovery_alpha'):
            for value in (True, float('nan'), float('inf')):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    StudyConfig(**{key: value})

    def test_duplicate_csv_headers_rejected_before_overwriting_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'rows.csv'; p.write_text('value,value\n1,99\n')
            with self.assertRaises(ValueError):
                read_table(p)

    def test_invalid_numeric_observations_never_coerce_to_measurements(self):
        t = datetime(2025, 1, 1, tzinfo=UTC)
        for value in (True, '12', float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                validate_observations([self.observation('r', t, t, 'resting_hr_bpm', value, 'bpm')])

    def test_vectorized_discovery_matches_independent_kernel(self):
        x = np.random.default_rng(4).normal(size=(50, 4))
        inside, outside = np.array([0, 2]), np.array([1, 3])
        weights = _component_weights([(inside, outside, np.triu_indices(2, 1))], 4)
        components = _components(x, weights)[0]
        actual = dnb_components(x, ['a', 'b', 'c', 'd'], ['a', 'c'])
        np.testing.assert_allclose(components, [actual[k] for k in ('sd_in', 'pcc_in', 'pcc_out')], rtol=1e-12)
        self.assertEqual(_directional_statistic(components[None], components[None], 1e-8)[0], 0.)

    def test_legacy_readiness_requires_affirmative_source_evidence(self):
        # PSEUDOCODE: distinguish a genuine zero from a zero with unknown or rewritten provenance.
        from rhythm_dnb.io.sources.lineage import legacy_field_kind
        row = {'water_ml': 0., 'source_dataset': 'test'}
        self.assertEqual(legacy_field_kind(row.get('provenance', {}), 'water_ml'), 'unknown')
        row['provenance'] = {'water_ml': {'source_file': 'a', 'source_key': 'water'}}
        self.assertEqual(legacy_field_kind(row['provenance'], 'water_ml'), 'source_traceable_retrospective')
        row['provenance']['water_ml']['method'] = 'manual_contextual_rewrite'
        self.assertEqual(legacy_field_kind(row['provenance'], 'water_ml'), 'constructed')

    def test_ambiguous_activity_peak_has_no_arbitrary_first_hour(self):
        # PSEUDOCODE: create repeated identical daily peaks and reject a tie-based phase estimate.
        from rhythm_dnb.measures.activity import daily_activity
        x = np.zeros(24); x[[1, 13]] = 10
        result = daily_activity(x, [60] * 24)
        self.assertIsNone(result['activity_m10_start_h'])
        self.assertIsNotNone(result['activity_ra'])

    def test_core_store_rejects_legacy_database_before_mutation(self):
        # PSEUDOCODE: point the core repository at an unrelated SQLite schema and verify read-only rejection.
        import sqlite3
        from rhythm_dnb.io.repository import RhythmRepository
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'old.sqlite'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE observations(id TEXT)')
            db.close()
            with self.assertRaises(ValueError), RhythmRepository(path).connection():
                pass

    def test_endpoint_orchestration_uses_configured_windows_and_persistence(self):
        # PSEUDOCODE: build a causal complete stable baseline and ensure no artificial endpoint is created.
        from rhythm_dnb.workflows.endpoints import EndpointDay, build_endpoint_timeline
        from rhythm_dnb.outcomes.criteria import fit_criteria
        start = date(2025, 1, 1); cutoff = datetime(2024, 12, 1, tzinfo=UTC)
        stable = [{'participant_id': str(i), 'stable': True, 'evidence_id': str(i), 'available_at': cutoff,
                   'anchor': {'S1': .1, 'E1': .1, 'A1': .1}, 'values': [{'S1': .2, 'E1': .2, 'A1': .2}]} for i in range(3)]
        criteria = fit_criteria(stable, cutoff, minimum_people=3)
        values = {'sleep_midpoint_h': 4., 'first_caloric_h': 8., 'last_caloric_h': 20., 'activity_hours': [0.] * 8 + [10.] * 10 + [0.] * 6}
        rows = [EndpointDay('p', start + timedelta(days=i), 'UTC', values, datetime(2025, 1, 2, 4, tzinfo=UTC) + timedelta(days=i), Provenance('observed', str(i), 'hash')) for i in range(25)]
        timeline = build_endpoint_timeline(rows, start, criteria, StudyConfig(), as_of=datetime(2025, 2, 1, tzinfo=UTC))
        self.assertEqual(timeline['events'], ())
        self.assertEqual(len(timeline['assessments']), 11)
        self.assertTrue(all(r.evaluable for r in timeline['assessments']))

    def test_median_baseline_is_fit_only_on_training_labels(self):
        # PSEUDOCODE: compare constant predictors under an asymmetric training distribution.
        from rhythm_dnb.text.evaluate import median_baseline, mean_baseline
        from rhythm_dnb.text.schema import METRICS
        rows = [{'scores': {key: v for key, *_ in METRICS}} for v in (0., 0., 90.)]
        median, mean = median_baseline(rows), mean_baseline(rows)
        self.assertEqual(set(median.values()), {0.})
        self.assertEqual(set(mean.values()), {30.})

    def test_different_source_receipts_preserve_real_evidence(self):
        # PSEUDOCODE: cover sheet/row, shared file and calculation lineage without promoting rewritten values.
        from rhythm_dnb.io.sources.lineage import legacy_field_kind
        cases = [
            {'source': {'source_file': 'diary.csv', 'source_sha256': 'hash'}, 'x': {'calculation': 'elapsed', 'source_line': 3}},
            {'source': {'supplements_sha256': {'a.xlsx': 'hash'}}, 'x': {'source_sheet': 'Combined', 'source_row': 2, 'source_key': 'TST'}},
            {'x': {'file': 'sleep.RData', 'source_key': 'duration'}},
            {'source': {'source_file': 'data.csv', 'source_sha256': 'hash'}, 'x': {'source_key': 'time'}}]
        for provenance in cases:
            self.assertEqual(legacy_field_kind(provenance, 'x'), 'source_traceable_retrospective')
            provenance['x']['method'] = 'manual_contextual_rewrite'
            self.assertEqual(legacy_field_kind(provenance, 'x'), 'constructed')

    def test_negative_followup_cannot_be_known_on_forecast_day(self):
        # PSEUDOCODE: reject a retrospectively attached zero with a forged early availability time.
        t = datetime(2025, 1, 1, 12, tzinfo=UTC)
        with self.assertRaises(ValueError):
            event_metrics([EvaluationDay('p', t, 3., 0, label_available_at=t)], 2.)

    def test_real_discovery_rejects_untraceable_flat_vectors(self):
        # PSEUDOCODE: fail closed before accepting constructed vectors into a real development cohort.
        from rhythm_dnb.research.simulate import simulate_cohort
        from rhythm_dnb.research.discover import discover
        c = simulate_cohort()
        with self.assertRaisesRegex(ValueError, 'source-backed'):
            discover(c['pairs'], replace(c['config'], simulation=False), {'people': [], 'features': list(OBJECTIVE8)}, c['discovery_cutoff'])

    def test_missing_scheduled_forecasts_reduce_coverage_without_becoming_zero(self):
        # PSEUDOCODE: register three monitoring days but supply one forecast; retain both missing days.
        from rhythm_dnb.contracts import MonitoringPeriod
        from rhythm_dnb.research.evaluate import include_monitoring_days, cluster_intervals
        t = datetime(2025, 1, 1, 12, tzinfo=UTC)
        periods = [MonitoringPeriod(p, t.date(), t.date() + timedelta(days=2), 'UTC') for p in ('p', 'q')]
        rows = [EvaluationDay('p', t, 3., 0)]
        completed = include_monitoring_days(rows, periods)
        self.assertEqual(sum(r.score is None and r.label is None for r in completed), 5)
        result = event_metrics(rows, 2., monitoring=periods)
        self.assertEqual(result['score_coverage'], 1 / 6)
        self.assertEqual(result['coverage_denominator'], 'registered_monitoring_days')
        self.assertEqual(cluster_intervals(rows, 2., monitoring=periods, repetitions=5)['alarm_ppv']['requested_replicates'], 5)
