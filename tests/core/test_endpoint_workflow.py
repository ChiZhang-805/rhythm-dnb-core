"""Offline labeling preserves frozen definitions, personal baselines and source availability."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from rhythm_dnb.cli import main, _cases
from rhythm_dnb.config import StudyConfig
from rhythm_dnb.contracts import PredictionRequest, Provenance
from rhythm_dnb.outcomes.criteria import fit_criteria
from rhythm_dnb.provenance import canonical_json
from rhythm_dnb.workflows.endpoints import EndpointDay, build_endpoint_timeline, label_timeline, parse_timeline

UTC = timezone.utc


class EndpointWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.config = StudyConfig()
        self.start = date(2025, 1, 1)
        self.as_of = datetime(2025, 2, 10, tzinfo=UTC)
        self.issued = datetime(2025, 1, 15, 12, tzinfo=UTC)
        cutoff = datetime(2024, 12, 1, tzinfo=UTC)
        stable = [{'participant_id': str(i), 'stable': True, 'evidence_id': str(i), 'available_at': cutoff,
                   'anchor': {'S1': .1, 'E1': .1, 'A1': .1}, 'values': [{'S1': .2, 'E1': .2, 'A1': .2}]} for i in range(3)]
        self.criteria = fit_criteria(stable, cutoff, minimum_people=3)
        values = {'sleep_midpoint_h': 4., 'first_caloric_h': 8., 'last_caloric_h': 20.,
                  'activity_hours': [0.] * 8 + [10.] * 10 + [0.] * 6}
        self.rows = [EndpointDay('p', self.start + timedelta(days=i), 'UTC', dict(values),
                     datetime(2025, 1, 2, 4, tzinfo=UTC) + timedelta(days=i),
                     Provenance('observed', str(i), 'software-fixture')) for i in range(35)]

    def timeline(self, rows=None):
        return build_endpoint_timeline(self.rows if rows is None else rows, self.start,
                                       self.criteria, self.config, as_of=self.as_of)

    def label(self, timeline, **options):
        return label_timeline(self.issued, timeline, self.config,
                              **{'followup_end': self.as_of, 'as_of': self.as_of, **options})

    def test_roundtrip_negative_and_confirmed_positive(self):
        timeline = self.timeline()
        restored = parse_timeline(json.loads(canonical_json(timeline)))
        self.assertEqual(restored, timeline)
        self.assertEqual(self.label(restored).value, 0)
        changed = [replace(row, values={**row.values, 'sleep_midpoint_h': 8., 'first_caloric_h': 12.})
                   if i >= 16 and i % 2 == 0 else row for i, row in enumerate(self.rows)]
        positive = self.timeline(changed)
        self.assertEqual(self.label(positive).value, 1)
        self.assertEqual(self.label(positive).onset, positive['events'][0].onset)
        self.assertIsNone(self.label(positive, followup_end=self.issued).value)

    def test_rejects_changed_protocol_foreign_people_and_inconsistent_events(self):
        timeline = self.timeline()
        for config in (replace(self.config, persistence_days=4), replace(self.config, horizon_days=5),
                       replace(self.config, min_lead_hours=48), replace(self.config, outcome_window=8),
                       replace(self.config, threshold_quantile=.9)):
            with self.assertRaisesRegex(ValueError, 'protocol'):
                label_timeline(self.issued, timeline, config, followup_end=self.as_of, as_of=self.as_of)
        bad = deepcopy(timeline)
        bad['protocol']['minimum'] = 5
        with self.assertRaisesRegex(ValueError, 'protocol'):
            self.label(bad)
        bad = deepcopy(timeline)
        bad['assessments'] = (replace(bad['assessments'][0], participant_id='foreign'),) + bad['assessments'][1:]
        with self.assertRaisesRegex(ValueError, 'participants'):
            self.label(bad)
        from rhythm_dnb.contracts import OutcomeEvent
        bad = deepcopy(timeline)
        bad['events'] = (OutcomeEvent('p', self.issued + timedelta(days=2), self.issued + timedelta(days=4)),)
        with self.assertRaisesRegex(ValueError, 'confirmed daily assessments'):
            self.label(bad)
        old = json.loads(canonical_json(timeline))
        old.pop('protocol')
        with self.assertRaisesRegex(ValueError, 'rebuild legacy'):
            parse_timeline(old)

    def test_full_baseline_must_end_even_if_last_baseline_day_is_missing(self):
        timeline = self.timeline([row for i, row in enumerate(self.rows) if i != 13])
        self.assertIsNotNone(timeline['anchor'])
        self.assertEqual(timeline['baseline_available_at'], datetime(2025, 1, 15, 4, tzinfo=UTC))
        early = label_timeline(datetime(2025, 1, 14, 12, tzinfo=UTC), timeline, self.config,
                               followup_end=self.as_of, as_of=self.as_of)
        self.assertEqual(early.reason, 'personal_endpoint_baseline_unavailable')

    def test_later_snapshot_cannot_be_labeled_as_if_known_earlier(self):
        with self.assertRaisesRegex(ValueError, 'requested label cutoff'):
            self.label(self.timeline(), as_of=self.as_of - timedelta(days=1))

    def test_label_cli_output_is_a_complete_case_and_checks_person_and_zone(self):
        request = PredictionRequest('p', self.issued, 'UTC', ())
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            for name, value in (('timeline', self.timeline()), ('study', asdict(self.config)), ('request', request)):
                (root / (name + '.json')).write_text(canonical_json(value), encoding='utf-8')
            command = ['label', '--timeline', str(root / 'timeline.json'), '--study', str(root / 'study.json'),
                       '--request', str(root / 'request.json'), '--as-of', self.as_of.isoformat(),
                       '--followup-end', self.as_of.isoformat(), '--output', str(root / 'case.json')]
            self.assertEqual(main(command), 0)
            case = _cases([json.loads((root / 'case.json').read_text(encoding='utf-8'))])[0]
            self.assertEqual(case.request, request)
            self.assertEqual(case.label.value, 0)
            self.assertEqual(case.label_available_at, self.as_of)
            self.assertEqual(case.outcome_protocol_id, self.timeline()['protocol_id'])
            for different in (replace(request, participant_id='foreign'), replace(request, timezone='Asia/Shanghai')):
                (root / 'request.json').write_text(canonical_json(different), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'same person and timezone'):
                    main(command)
