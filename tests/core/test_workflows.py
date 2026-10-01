"""End-to-end synthetic fitting, frozen inference and leakage guard tests."""

from dataclasses import replace, asdict
from datetime import datetime, timedelta, timezone
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from support.cohort import make_cohort, _panel
from rhythm_dnb.workflows.develop import develop
from rhythm_dnb.workflows.validate import validate
from rhythm_dnb.bundles import load_bundle, save_bundle, check_compatibility
from rhythm_dnb.api import RhythmPredictor
from rhythm_dnb.contracts import parse_request, Provenance
from rhythm_dnb.provenance import canonical_json, fingerprint
from rhythm_dnb.research.discover import discover
from rhythm_dnb.research.calibrate import calibrate


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort = make_cohort(people_per_role=20, development_people=200)
        c = cls.cohort
        cls.bundle = develop(c['reference'], c['pairs'], c['calibration'], c['config'],
            **{k: c[k] for k in ('reference_cutoff', 'discovery_cutoff', 'calibration_cutoff', 'calibration_events', 'calibration_monitoring')})

    def test_complete_research_chain(self):
        self.assertTrue(self.bundle['discovery']['modules'])
        self.assertIsNotNone(self.bundle['calibration']['threshold'])
        result = validate(self.bundle, self.cohort['test'], bootstrap_repetitions=10,
                          evaluation_as_of=datetime(2025, 10, 1, tzinfo=timezone.utc),
                          events=self.cohort['test_events'], monitoring=self.cohort['test_monitoring'])
        self.assertEqual(result['domain'], 'source_backed')
        self.assertFalse(result['clinical_validation'])
        self.assertEqual(result['metrics']['events'], 10)

    def test_bundle_roundtrip_immutable_and_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            path = save_bundle(self.bundle, folder)
            self.assertEqual(load_bundle(path)['id'], self.bundle['id'])
            with self.assertRaises(FileExistsError):
                save_bundle(self.bundle, folder)
        bad = deepcopy(self.bundle); bad['reference']['matrix'][0][0] += 1
        with self.assertRaises(ValueError):
            check_compatibility(bad)

    def test_future_history_and_label_do_not_affect_score(self):
        request = self.cohort['test'][0].request
        predictor = RhythmPredictor(self.bundle)
        a = predictor.predict(request)
        future = replace(request.history[0], day=request.history[0].day + timedelta(days=10))
        b = predictor.predict(replace(request, history=request.history + (future,)))
        self.assertEqual(a.score, b.score)
        payload = __import__('json').loads(canonical_json(request)); payload['label'] = 1
        with self.assertRaises(TypeError):
            parse_request(payload)

    def test_reference_participant_and_early_forecast_rejected(self):
        request = self.cohort['test'][0].request
        with self.assertRaises(ValueError):
            RhythmPredictor(self.bundle).predict(replace(request, participant_id='ref-0'))
        with self.assertRaises(ValueError):
            RhythmPredictor(self.bundle).predict(replace(request, issued_at=datetime(2025, 1, 1, 12, tzinfo=timezone.utc)))

    def test_late_arrival_and_constructed_cells_abstain(self):
        request = self.cohort['test'][0].request; panel = request.history[0]
        for feature in (replace(panel.features[0], available_at=request.issued_at + timedelta(seconds=1)),
                        replace(panel.features[0], provenance=Provenance('constructed', 'rewrite', 'abc')),
                        replace(panel.features[0], measured_until=None)):
            bad = replace(panel, features=(feature,) + panel.features[1:])
            result = RhythmPredictor(self.bundle).predict(replace(request, history=(bad,)))
            self.assertIsNone(result.warning)
            self.assertIsNone(result.score)

    def test_missing_reference_dimensions_do_not_shrink_network(self):
        request = self.cohort['test'][0].request; panel = request.history[0]
        result = RhythmPredictor(self.bundle).predict(replace(request, history=(replace(panel, features=panel.features[:-1]),)))
        self.assertIsNone(result.score)

    def test_discovery_requires_all_three_conditions(self):
        c = self.cohort
        # Identical states have exactly no changes, so bootstrapping cannot invent a DNB.
        pairs = [replace(p, pre_event=p.stable,
                         pre_event_panel=_panel(p.participant_id, p.pre_event_panel.day, p.stable,
                                                self.bundle['reference']['features'])) for p in c['pairs']]
        result = discover(pairs, c['config'], self.bundle['reference'], c['discovery_cutoff'])
        self.assertEqual(result['modules'], [])

    def test_discovery_rejects_future_evidence(self):
        c = self.cohort
        pairs = [replace(p, evidence_available_at=datetime(2030, 1, 1, tzinfo=timezone.utc)) for p in c['pairs']]
        with self.assertRaises(ValueError):
            discover(pairs, c['config'], self.bundle['reference'], c['discovery_cutoff'])

    def test_test_labels_cannot_be_known_early(self):
        with self.assertRaises(ValueError):
            validate(self.bundle, self.cohort['test'], bootstrap_repetitions=0,
                     evaluation_as_of=datetime(2025, 8, 1, tzinfo=timezone.utc),
                     events=self.cohort['test_events'], monitoring=self.cohort['test_monitoring'])

    def test_rolling_needs_own_calibration(self):
        request = self.cohort['test'][0].request
        result = RhythmPredictor(self.bundle).predict(request, method='rolling')
        self.assertIsNone(result.warning)
        self.assertIn('insufficient_complete_days', result.reasons)

    def test_reference_draw_is_independent_per_person(self):
        reference = self.bundle['reference']
        self.assertEqual(len(reference['people']), len(set(reference['people'])))
        self.assertEqual(len(reference['matrix']), 60)
