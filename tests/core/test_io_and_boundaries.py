"""Storage, acquisition, source-parser parity and architectural dependency tests."""

from dataclasses import replace
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
import importlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from rhythm_dnb.io.repository import RhythmRepository
from rhythm_dnb.io.splits import make_splits, validate_splits
from rhythm_dnb.io.sources.hospital import read_observations
from rhythm_dnb.provenance import canonical_json, build_lineage, eligible_measurement
from rhythm_dnb.contracts import Observation, Provenance
from rhythm_dnb.research.simulate import _panel
from rhythm_dnb.measures.panel import OBJECTIVE8
from rhythm_dnb.workflows.prepare import prepare_day


class IOTests(unittest.TestCase):
    def test_repository_read_as_of_and_unique_forecasts(self):
        panel = _panel('p', date(2025, 1, 1), [4, 7, 8, 20, 10, .5, 100, 60], OBJECTIVE8)
        with tempfile.TemporaryDirectory() as directory:
            repo = RhythmRepository.create(Path(directory) / 'new.sqlite')
            repo.append_panel(panel)
            before = repo.read_features_as_of('p', datetime(2025, 1, 1, tzinfo=timezone.utc))
            after = repo.read_features_as_of('p', datetime(2025, 1, 3, tzinfo=timezone.utc))
            self.assertEqual(before[0].features, ())
            self.assertEqual(after[0], panel)
            with self.assertRaises(sqlite3.IntegrityError):
                repo.append_panel(panel)
            with self.assertRaises(FileExistsError):
                RhythmRepository.create(repo.path)

    def test_hospital_requires_aware_times_and_explicit_map(self):
        row = {'observation_id': '1', 'participant_id': 'p', 'variable': 'resting_hr_bpm', 'value': 60,
               'unit': 'bpm', 'start': '2025-01-01T10:00:00+00:00', 'end': '2025-01-01T10:00:00+00:00',
               'available_at': '2025-01-01T10:00:00+00:00'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.json'; path.write_text(json.dumps([row]))
            result = read_observations(path, {k: k for k in row}, timezone='UTC', source_id='hospital')
            self.assertEqual(result[0].value, 60)
            self.assertTrue(eligible_measurement(result[0].provenance))
            row['start'] = '2025-01-01T10:00:00'; path.write_text(json.dumps([row]))
            with self.assertRaises(ValueError):
                read_observations(path, {k: k for k in row}, timezone='UTC', source_id='hospital')

    def test_prepare_requires_complete_sleep_and_eating_logs(self):
        start = datetime(2025, 1, 1, 22, tzinfo=timezone.utc); end = start + timedelta(hours=4)
        provenance = Provenance('observed', 'sensor', 'abc')
        rows = [Observation('sleep', 'p', 'sleep_episode', 1., 'state', start, end, end, 'UTC', provenance),
                Observation('food', 'p', 'caloric_event', 500., 'kcal', start, start, start, 'UTC', provenance)]
        kwargs = dict(participant_id='p', day=date(2025, 1, 1), zone='UTC', issued_at=end.replace(hour=12), panel_id='objective8')
        a = prepare_day(rows, **kwargs)
        b = prepare_day(rows, **kwargs, sleep_complete=True, eating_complete=True)
        self.assertIsNone(a.features[0].value)
        self.assertEqual(b.features[0].value, 0)
        self.assertEqual(b.features[1].value, 4)
        self.assertEqual(b.features[2].value, 22)

    def test_lineage_never_promotes_constructed_values(self):
        lineage = build_lineage([Provenance('observed', 'a', 'x'), Provenance('constructed', 'b', 'y')], 'mean')
        self.assertEqual(lineage.kind, 'constructed')
        self.assertFalse(eligible_measurement(lineage))
        self.assertFalse(eligible_measurement(Provenance('observed', 'a')))

    def test_splits_and_template_leakage(self):
        splits = make_splits([str(i) for i in range(40)])
        self.assertEqual(sum(map(len, splits['partitions'].values())), 40)
        with self.assertRaises(ValueError):
            validate_splits([{'split': 'train', 'group_id': 'a'}, {'split': 'test', 'group_id': 'a'}])

    def test_import_has_no_training_or_legacy_dependency(self):
        code = "import sys; from rhythm_dnb.api import RhythmPredictor; assert not any(n.startswith(('torch','rhythm_dnb.research','rhythm_dnb.outcomes','rhythm_dnb.text.train','rhythm_pipeline','text_distillation')) for n in sys.modules)"
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)

    def test_source_parser_numeric_parity(self):
        fixture = json.loads((Path(__file__).resolve().parents[1] / 'fixtures/migration.json').read_text(encoding='utf-8'))
        for name in ('pmdata', 'lifesnaps', 'sleep_diary', 'rest', 'delsom', 'manchester'):
            new = importlib.import_module('rhythm_dnb.io.sources.' + name)
            self.assertTrue(callable(new.build_records))
        for row in fixture['parsers']:
            module = importlib.import_module('rhythm_dnb.io.sources.' + row['source'])
            with self.subTest(source=row['source'], args=row['args']):
                self.assertEqual(getattr(module, row['function'])(*row['args'], **row['kwargs']), row['expected'])

    def test_mechanical_legacy_schema_parity(self):
        from rhythm_dnb.io.sources.legacy_schema import validate_record as new
        fixture = json.loads((Path(__file__).resolve().parents[1] / 'fixtures/migration.json').read_text(encoding='utf-8'))['schema']
        self.assertEqual(new(fixture['input']), fixture['expected'])
