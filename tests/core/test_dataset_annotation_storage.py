"""Adding labels must not revise original observations or remove their provenance."""

from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest

from rhythm_dnb.provenance import file_hash, fingerprint
from rhythm_dnb.research.dataset_annotations import POLICY
from rhythm_dnb.research.experiment_data import save_json
from tools.annotate_research_database import apply


class AnnotationStorageTests(unittest.TestCase):
    def proposal(self, root, *, record_id='one'):
        database = root / 'dataset.sqlite'
        with closing(sqlite3.connect(database)) as db, db:
            db.execute('CREATE TABLE observations(record_id TEXT PRIMARY KEY,value REAL,label INTEGER)')
            db.execute('INSERT INTO observations VALUES (?,?,?)', ('one', 17.25, None))
            db.execute('CREATE TABLE observation_provenance(record_id TEXT,payload TEXT)')
            db.execute('INSERT INTO observation_provenance VALUES (?,?)', ('one', 'original-source'))
        output = root / 'proposal'; output.mkdir()
        plan = {'created_at': '2026-10-10T00:00:00+00:00', 'source_sha256': file_hash(database), 'policy': POLICY}
        plan['id'] = fingerprint(plan)
        row = {'record_id': record_id, 'is_simulated_cohort': 0, 'contains_constructed_values': 1,
               'reference_scope': 'not_provided', 'reference_event_in_followup': None,
               'reference_onset_at': None, 'candidate_state': None, 'candidate_onset_date': None,
               'candidate_confirmed_date': None, 'status': 'baseline_only', 'evidence': {'reason': 'short'}}
        for name, value in [('plan.json', plan), ('annotations.json', [row]),
                            ('sequence-evidence.json', []), ('warnings.json', [])]:
            save_json(output / name, value)
        files = {p.name: file_hash(p) for p in output.iterdir()}
        save_json(output / 'manifest.json', {'files': files, 'id': fingerprint(files)})
        return database, output, plan

    def test_preserves_all_original_tables_and_saves_exact_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            database, output, plan = self.proposal(Path(folder))
            receipt = apply(database, output)
            self.assertTrue(receipt['all_original_tables_unchanged'])
            self.assertEqual(file_hash(output / 'rhythm.before.sqlite'), plan['source_sha256'])
            with closing(sqlite3.connect(database)) as db, db:
                self.assertEqual(db.execute('SELECT * FROM observations').fetchall(), [('one', 17.25, None)])
                self.assertEqual(db.execute('SELECT * FROM observation_provenance').fetchall(), [('one', 'original-source')])
                self.assertEqual(db.execute('SELECT candidate_state,annotation_status FROM research_labeled_observations').fetchall(), [(None, 'baseline_only')])

    def test_unknown_record_rolls_back_the_entire_annotation_transaction(self):
        with tempfile.TemporaryDirectory() as folder:
            database, output, _ = self.proposal(Path(folder), record_id='not-present')
            with self.assertRaises(sqlite3.IntegrityError):
                apply(database, output)
            with closing(sqlite3.connect(database)) as db, db:
                self.assertEqual(db.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE 'research_%'").fetchone()[0], 0)
                self.assertEqual(db.execute('SELECT * FROM observations').fetchall(), [('one', 17.25, None)])

    def test_stale_database_or_changed_proposal_cannot_be_applied(self):
        with tempfile.TemporaryDirectory() as folder:
            database, output, _ = self.proposal(Path(folder))
            with closing(sqlite3.connect(database)) as db, db:
                db.execute('UPDATE observations SET value=18')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                apply(database, output)
            self.assertFalse((output / 'rhythm.before.sqlite').exists())


if __name__ == '__main__':
    unittest.main()
