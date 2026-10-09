"""Scientific failures reproduced during the 2026-10-01 full repository review."""

from dataclasses import replace
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import tempfile
import unittest
import json
import sqlite3
import numpy as np
from rhythm_dnb.config import StudyConfig
from rhythm_dnb.contracts import Observation, Provenance, OutcomeAssessment, OutcomeEvent, EvaluationDay, AlarmState
from rhythm_dnb.provenance import build_lineage, eligible_measurement
from rhythm_dnb.workflows.prepare import prepare_day
from rhythm_dnb.warning.policy import advance
from rhythm_dnb.warning.windows import select_window
from rhythm_dnb.research.evaluate import replay, event_metrics
from support.cohort import _panel
from rhythm_dnb.measures.panel import OBJECTIVE8
from rhythm_dnb.outcomes.events import detect_events
from rhythm_dnb.outcomes.labels import future_label
from rhythm_dnb.io.tabular import read_table
from rhythm_dnb.io.validation import validate_observations
from rhythm_dnb.research.discover import _component_weights, _components, _directional_statistic
from rhythm_dnb.dnb.classic import dnb_components

UTC = timezone.utc


class ReviewRegressions(unittest.TestCase):
    def test_legacy_audit_distinguishes_unbuilt_panels_from_incomplete_rows(self):
        from rhythm_dnb.research.report import audit_legacy_store
        from rhythm_dnb.measures.panel import JOINT12
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.sqlite'
            with closing(sqlite3.connect(path)) as db, db:
                db.execute('CREATE TABLE observations (record_id TEXT, participant_id TEXT, source_dataset TEXT, '
                    'event_onset_at TEXT, followup_end_at TEXT, rhythm_state_gt INTEGER, row_sha256 TEXT, '
                    'sleep_start_hour REAL, sleep_end_hour REAL, sleep_duration_h REAL, resting_hr_bpm REAL)')
                db.execute('CREATE TABLE observation_provenance (record_id TEXT, payload TEXT)')
                db.execute("INSERT INTO observations VALUES ('r', 'p', 'test-only', NULL, NULL, NULL, 'hash', 23, 7, 8, 60)")
                fields = ('sleep_start_hour', 'sleep_end_hour', 'sleep_duration_h', 'resting_hr_bpm')
                provenance = {key: {'source_file': 'test-only.csv', 'source_key': key} for key in fields}
                db.execute('INSERT INTO observation_provenance VALUES (?, ?)', ('r', json.dumps({'provenance': provenance})))
            initial = audit_legacy_store(path)
            self.assertEqual(initial['source_traceable_fields']['sleep_start_hour'], 1)
            self.assertIsNone(initial['complete_source_traceable_objective8_rows'])
            self.assertIsNone(initial['complete_source_traceable_joint12_rows'])
            self.assertIn('sleep_midpoint_h', initial['stored_panel_coverage']['objective8']['missing_columns'])
            self.assertEqual(initial['stored_panel_coverage']['objective8']['status'], 'requires_source_derivation')
            with closing(sqlite3.connect(path)) as db, db:
                for key in JOINT12:
                    if key not in fields:
                        db.execute(f'ALTER TABLE observations ADD COLUMN "{key}" REAL')
                        db.execute(f'UPDATE observations SET "{key}"=1')
                    provenance[key] = {'source_file': 'test-only.csv', 'source_key': key}
                db.execute('UPDATE observation_provenance SET payload=?', (json.dumps({'provenance': provenance}),))
            complete = audit_legacy_store(path)
            self.assertEqual(complete['complete_source_traceable_objective8_rows'], 1)
            self.assertEqual(complete['complete_source_traceable_joint12_rows'], 1)
            self.assertFalse(complete['prospective_validation_ready'])
            with closing(sqlite3.connect(path)) as db, db:
                db.execute('UPDATE observations SET first_caloric_h=NULL')
            incomplete = audit_legacy_store(path)
            self.assertEqual(incomplete['complete_source_traceable_objective8_rows'], 0)
            self.assertEqual(incomplete['complete_source_traceable_joint12_rows'], 0)
            self.assertEqual(incomplete['stored_panel_coverage']['objective8']['missing_columns'], [])

    def test_recurrent_events_cannot_inflate_calibration_participant_count(self):
        from rhythm_dnb.research.calibrate import calibrate
        from rhythm_dnb.contracts import MonitoringPeriod
        t = datetime(2025, 1, 1, 12, tzinfo=UTC)
        rows = [EvaluationDay('p', t, 1., 0, None, t + timedelta(days=10), 'UTC')]
        events = [OutcomeEvent('p', t + timedelta(days=d), t + timedelta(days=d + 2)) for d in (30, 60)]
        config = replace(StudyConfig(), calibration_min_events=2, calibration_min_negative_days=1)
        result = calibrate(rows, config, {'id': 'r', 'people': []}, {'id': 'd', 'people': [], 'modules': [[0, 1]]},
            t + timedelta(days=100), events=events, monitoring=[MonitoringPeriod('p', t.date(), t.date(), 'UTC')])
        self.assertEqual(result['status'], 'insufficient_outcomes')
        self.assertEqual(result['event_people'], 1)

    def test_endpoint_thresholds_reject_impossible_input_values(self):
        from rhythm_dnb.outcomes.criteria import fit_criteria
        t = datetime(2025, 1, 1, tzinfo=UTC)
        for value in (True, -1., 1.1, float('nan'), float('inf')):
            rows = [{'participant_id': str(i), 'anchor': {'S1': 0., 'E1': 0., 'A1': 0.},
                'stable': True, 'evidence_id': str(i), 'available_at': t,
                'values': [{'S1': 1., 'E1': 1., 'A1': value}]} for i in range(3)]
            with self.subTest(value=value), self.assertRaises(ValueError):
                fit_criteria(rows, t, minimum_people=3)

    def test_manchester_duration_uses_elapsed_time_through_dst(self):
        import pandas as pd
        from rhythm_dnb.io.sources.manchester import assemble_records, _local_datetime
        for day, start, end, duration, allowed in (
            ('2025-03-30', '00:00:00+00:00', '03:00:00+01:00', 150, False),
            ('2025-10-26', '00:00:00+01:00', '02:00:00+00:00', 180, True)):
            frame = pd.DataFrame([{'Id': 'p', 'startTime': day + 'T' + start, 'endTime': day + 'T' + end, 'sleepDuration': duration}])
            if allowed:
                self.assertEqual(assemble_records(frame, pd.DataFrame(), pd.DataFrame([{'Id': 'p'}]))[0]['sleep_duration_h'], 3.)
            else:
                with self.assertRaises(ValueError):
                    assemble_records(frame, pd.DataFrame(), pd.DataFrame([{'Id': 'p'}]))
        with self.assertRaises(ValueError):
            _local_datetime('2025-10-26T01:30:00')

    def test_legacy_schema_accepts_individually_unknown_text_scores(self):
        from rhythm_dnb.io.sources.legacy_schema import validate_record
        result = validate_record({'record_id': 'fixture', 'participant_id': 'p', 'observed_at': '2025-01-01T12:00:00+00:00',
            'source_dataset': 'test-only', 'split': 'test', 'text_anxiety_intensity': 0., 'text_model_id': 'test-only'})
        self.assertEqual(result['text_anxiety_intensity'], 0.)
        self.assertIsNone(result['text_sadness_intensity'])

    def test_measurement_identity_ignores_prose_but_preserves_calculations(self):
        from rhythm_dnb.definitions import normalized_syntax
        first = 'def calculate(x):\n    return x + 1\n'
        commented = 'def calculate(x):\n  """A clearer explanation."""\n  # Same operation\n  return x + 1\n'
        self.assertEqual(normalized_syntax(first), normalized_syntax(commented))
        self.assertNotEqual(normalized_syntax(first), normalized_syntax(first.replace('+ 1', '+ 2')))

    def test_serialized_panel_cannot_guess_measurement_identity(self):
        from rhythm_dnb.contracts import parse_panel
        from rhythm_dnb.provenance import canonical_json
        import json
        panel = _panel('p', date(2025, 1, 1), np.arange(8, dtype=float), OBJECTIVE8)
        payload = json.loads(canonical_json(panel))
        payload['features'][0].pop('measurement_id')
        with self.assertRaisesRegex(ValueError, 'measurement identity'):
            parse_panel(payload)

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

    def test_main_period_does_not_reintroduce_sleep_outside_the_day(self):
        start = datetime(2025, 1, 1, 22, tzinfo=UTC)
        rows = [self.observation('sleep', start, start + timedelta(hours=5)),
                self.observation('main', start, start + timedelta(hours=9), 'main_sleep_period', None, 'interval')]
        panel = prepare_day(rows, 'p', date(2025, 1, 2), 'UTC', datetime(2025, 1, 3, 12, tzinfo=UTC),
                            panel_id='objective8', sleep_complete=True)
        duration = next(f for f in panel.features if f.name == 'sleep_duration_h')
        self.assertIsNone(duration.value)
        self.assertIsNotNone(duration.reason)

    def test_prior_day_sleep_cannot_be_hidden_by_an_incorrect_main_boundary(self):
        start = datetime(2025, 1, 1, 22, tzinfo=UTC)
        rows = [self.observation('early', start, start + timedelta(hours=3)),
                self.observation('late', start + timedelta(hours=4), start + timedelta(hours=8)),
                self.observation('main', start + timedelta(hours=1), start + timedelta(hours=8), 'main_sleep_period', None, 'interval')]
        with self.assertRaisesRegex(ValueError, 'cut through'):
            prepare_day(rows, 'p', date(2025, 1, 2), 'UTC', datetime(2025, 1, 3, 12, tzinfo=UTC),
                        panel_id='objective8', sleep_complete=True)

    def test_mixed_simulation_and_real_lineage_is_ineligible(self):
        p = build_lineage([Provenance('observed', 'real', 'a'), Provenance('synthetic', 'sim', 'b')], 'combined')
        self.assertFalse(eligible_measurement(p))
        self.assertFalse(eligible_measurement(Provenance('observed', 'real', 'a', independent='yes')))

    def test_missing_target_cannot_reuse_old_rolling_score(self):
        last = date(2025, 1, 31)
        panels = [_panel('p', last - timedelta(days=i), [4, 7, 8, 20, 10, .5, 100, 60], OBJECTIVE8) for i in range(1, 28)]
        _, reason = select_window(panels, 'p', 'UTC', last, OBJECTIVE8, datetime(2025, 2, 1, 12, tzinfo=UTC))
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
        from support.cohort import make_cohort
        from rhythm_dnb.research.discover import discover
        c = make_cohort()
        c['pairs'] = [replace(p, stable_panel=None, pre_event_panel=None) for p in c['pairs']]
        with self.assertRaisesRegex(ValueError, 'source-backed'):
            discover(c['pairs'], c['config'], {'people': [], 'features': list(OBJECTIVE8)}, c['discovery_cutoff'])

    def test_sleep_crossing_boundary_is_counted_once_across_two_days(self):
        start = datetime(2025, 1, 1, 22, tzinfo=UTC)
        row = self.observation('overnight', start, start + timedelta(hours=9))
        panels = [prepare_day([row], 'p', date(2025, 1, day), 'UTC',
                              datetime(2025, 1, day + 1, 12, tzinfo=UTC),
                              panel_id='objective8', sleep_complete=True) for day in (1, 2)]
        durations = [next(f.value for f in p.features if f.name == 'sleep_duration_h') for p in panels]
        self.assertEqual(durations, [6., 3.])
        self.assertEqual(sum(durations), 9.)
        first = next(f for f in panels[0].features if f.name == 'sleep_duration_h')
        self.assertEqual(first.measured_until, datetime(2025, 1, 2, 4, tzinfo=UTC))
        self.assertEqual(first.available_at, row.end)

    def test_discovery_rejects_measurement_arriving_after_forecast(self):
        from support.cohort import make_cohort
        from rhythm_dnb.dnb.reference import fit_reference
        from rhythm_dnb.research.discover import discover
        c = make_cohort()
        reference = fit_reference(c['reference'], OBJECTIVE8, c['reference_cutoff'], minimum=c['config'].reference_min_people)
        p = c['pairs'][0]
        late = replace(p.pre_event_panel, features=tuple(replace(f, available_at=p.evidence_available_at) for f in p.pre_event_panel.features))
        with self.assertRaisesRegex(ValueError, 'source-backed'):
            discover([replace(p, pre_event_panel=late)] + c['pairs'][1:], c['config'], reference, c['discovery_cutoff'])

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
