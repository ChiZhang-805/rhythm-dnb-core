"""Exercise disjoint resume accounting and rejection of corrupted score caches."""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from rhythm_dnb.provenance import canonical_json, file_hash, fingerprint
from rhythm_dnb.research.experiment_data import save_json
from rhythm_dnb.text.schema import CATEGORIES
from rhythm_dnb.research.expanded_results import merge_scores, append_database


class ExpandedCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.packet = self.root/'packet'; self.packet.mkdir()
        self.download = self.root/'download'; self.download.mkdir()
        self.tasks = {}; bindings = []; cached = {}
        categories = ('emotion', 'sleep', 'diet', 'social')
        for person in ('old', 'new'):
            for category in categories:
                text = person+'-'+category
                key = fingerprint([category, text])
                values = dict.fromkeys(CATEGORIES[category][1], 31.)
                if person == 'old':
                    cached[key] = values
                else:
                    self.tasks[key] = dict(category=category, text=text)
                bindings.append(dict(record_id=person, participant_id=person, category=category, task_id=key))
        self.profile = dict(checkpoint_id='chosen', batch_size=1)
        self.profile['id'] = fingerprint(self.profile)
        previous = dict(model_id='chosen', predictions=cached, profile=self.profile)
        previous['id'] = fingerprint(previous)
        inventory = dict(tasks=self.tasks, tasks_id=fingerprint(self.tasks))
        for name, data in [('cached-text-scores', previous), ('text-bindings', bindings), ('text-tasks', inventory)]:
            save_json(self.packet/(name+'.json'), data)
        plan = dict(text_model_id='chosen', original_records=2, files={p.name: file_hash(p) for p in self.packet.iterdir()})
        plan['id'] = fingerprint(plan); save_json(self.packet/'plan.json', plan)
        switch = {'files': {}}
        for folder in ('text-output', 'text-batched-output'):
            (self.download/folder).mkdir()
        for shard in range(2):
            assigned = [key for i, key in enumerate(sorted(self.tasks)) if i % 2 == shard]
            base_binding = dict(tasks_id=inventory['tasks_id'], model_id='chosen', shard=shard, shards=2, precision='fp32')
            old = self.download/'text-output'/f'shard-{shard}.sqlite'
            self.write_cache(old, base_binding, self.profile, assigned[:1])
            old_hash = file_hash(old)
            switch['files'][old.name] = dict(sha256=old_hash, committed_tasks=1)
            binding = dict(base_binding, skipped_cache_sha256=old_hash, skipped_tasks=1, batch_size=16)
            profile = dict(checkpoint_id='chosen', batch_size=16); profile['id'] = fingerprint(profile)
            self.write_cache(self.download/'text-batched-output'/old.name, binding, profile, assigned[1:])
            save_json(self.download/'text-batched-output'/f'shard-{shard}-complete.json', dict(binding, count=1, profile=profile))
            save_json(self.download/'text-batched-output'/f'shard-{shard}-batch-check.json',
                      dict(maximum_absolute_difference=.00001, batch_size=16))
        save_json(self.download/'batch-switch.json', switch)
        save_json(self.download/'text-batched-output/exit.json', dict(shard_0=0, shard_1=0))

    def write_cache(self, path, binding, profile, keys):
        with closing(sqlite3.connect(path)) as db, db:
            db.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT)')
            db.execute('CREATE TABLE predictions (task_id TEXT PRIMARY KEY,estimates TEXT)')
            db.executemany('INSERT INTO metadata VALUES (?,?)', [('binding', canonical_json(binding)), ('profile', canonical_json(profile))])
            for key in keys:
                values = dict.fromkeys(CATEGORIES[self.tasks[key]['category']][1], 32.)
                db.execute('INSERT INTO predictions VALUES (?,?)', (key, canonical_json(values)))

    def test_merge_preserves_both_numerical_profiles_and_every_task(self):
        scores, bindings, _ = merge_scores(self.packet, self.download)
        self.assertEqual(scores['counts']['texts'], 8)
        self.assertEqual(scores['counts']['records'], 2)
        self.assertEqual(len(scores['profiles']), 2)
        self.assertEqual(set(scores['predictions']), {r['task_id'] for r in bindings})
        self.assertTrue(scores['experimental_uncalibrated'])

    def test_corrupted_preserved_cache_fails_without_replacing_outputs(self):
        with closing(sqlite3.connect(self.download/'text-output/shard-0.sqlite')) as db, db:
            db.execute("UPDATE predictions SET estimates='{}'")
        with self.assertRaisesRegex(ValueError, 'Preserved partial cache changed'):
            merge_scores(self.packet, self.download)

    def test_overlap_between_resumed_and_completed_tasks_is_rejected(self):
        old = self.download/'text-output/shard-0.sqlite'
        with closing(sqlite3.connect(old)) as db:
            key = db.execute('SELECT task_id FROM predictions').fetchone()[0]
        with closing(sqlite3.connect(self.download/'text-batched-output/shard-0.sqlite')) as db, db:
            db.execute('UPDATE predictions SET task_id=?', (key,))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            merge_scores(self.packet, self.download)

    def test_failed_worker_does_not_produce_a_complete_result(self):
        path = self.download/'text-batched-output/exit.json'
        path.write_text('{"shard_0":0,"shard_1":1}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'not completed'):
            merge_scores(self.packet, self.download)

    def database_inputs(self):
        scores, bindings, plan = merge_scores(self.packet, self.download)
        database = self.root/'rhythm.sqlite'
        with closing(sqlite3.connect(database)) as db, db:
            db.execute('CREATE TABLE observations (record_id TEXT PRIMARY KEY,payload TEXT)')
            db.execute('CREATE TABLE research_followup_days (record_id TEXT PRIMARY KEY,payload TEXT)')
            db.executemany('INSERT INTO observations VALUES (?,?)', [('old','original'),('new','original')])
            db.execute("INSERT INTO research_followup_days VALUES ('followup','unchanged outcome')")
        plan['source_database_sha256'] = file_hash(database)
        evaluation = self.root/'evaluation'; evaluation.mkdir()
        issued = '2000-01-29T12:00:00+00:00'
        save_json(evaluation/'result.json', dict(packet_id=plan['id'], methods={'personal_dnb': dict(threshold=2.)}))
        save_json(evaluation/'personal_dnb-test-alerts.json', [dict(participant_id='person',issued_at=issued.replace('T',' '),score=1.,warning=False,status='below_threshold')])
        files = {p.name: file_hash(p) for p in evaluation.iterdir()}
        save_json(evaluation/'archive-manifest.json', dict(files=files,id=fingerprint(files)))
        output = self.root/'merged'; output.mkdir()
        measurements = [dict(participant_id='person', issued_at=issued, record_id='followup')]
        return database, output, scores, bindings, plan, evaluation, measurements

    def test_database_append_preserves_inputs_and_has_verified_recovery_copy(self):
        args = self.database_inputs()
        append_database(*args)
        database, output, _, _, plan, _, _ = args
        self.assertEqual(file_hash(output/'rhythm.before-results.sqlite'), plan['source_database_sha256'])
        with closing(sqlite3.connect(database)) as db:
            self.assertEqual(db.execute('SELECT * FROM observations ORDER BY record_id').fetchall(),
                             [('new','original'),('old','original')])
            self.assertEqual(db.execute('SELECT * FROM research_followup_days').fetchall(), [('followup','unchanged outcome')])
            self.assertEqual(db.execute('SELECT count(*) FROM research_text_scores').fetchone()[0],8)
            self.assertEqual(db.execute('SELECT count(*) FROM research_expanded_forecasts').fetchone()[0],1)

    def test_bad_reference_rolls_back_new_tables_and_preserves_the_source_database(self):
        args = list(self.database_inputs())
        args[3][0]['record_id'] = 'absent-source-record'
        with self.assertRaises(sqlite3.IntegrityError):
            append_database(*args)
        with closing(sqlite3.connect(args[0])) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertEqual(tables, {'observations','research_followup_days'})


if __name__ == '__main__':
    unittest.main()
