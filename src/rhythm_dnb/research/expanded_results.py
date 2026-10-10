"""Verify downloaded scores and forecasts, then append experiment outputs without changing input data."""

from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sqlite3

from rhythm_dnb.provenance import canonical_json, file_hash, fingerprint
from rhythm_dnb.research.experiment_data import save_json
from rhythm_dnb.text.schema import CATEGORIES
from rhythm_dnb.timebase import instant


def read_json(path):
    # PSEUDOCODE: read an existing artifact without rewriting it.
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_archive(directory):
    # PSEUDOCODE: accept only contained files whose bytes match the completed experiment manifest.
    directory = Path(directory).resolve()
    manifest = read_json(directory/'archive-manifest.json')
    if fingerprint(manifest['files']) != manifest['id']:
        raise ValueError('Archive manifest changed.')
    for name, digest in manifest['files'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or file_hash(path) != digest:
            raise ValueError('Invalid archived file: ' + name)
    return manifest


def merge_scores(packet, downloaded):
    # PSEUDOCODE: validate each disjoint shard and its skipped cache; combine all original texts exactly once.
    packet, downloaded = Path(packet), Path(downloaded)
    plan = read_json(packet/'plan.json')
    if fingerprint({k: v for k, v in plan.items() if k != 'id'}) != plan['id']:
        raise ValueError('Packet identity changed.')
    for name, digest in plan['files'].items():
        if Path(name).name != name or file_hash(packet/name) != digest:
            raise ValueError('Packet file changed: '+name)
    inventory = read_json(packet/'text-tasks.json')
    cached = read_json(packet/'cached-text-scores.json')
    bindings = read_json(packet/'text-bindings.json')
    if fingerprint(inventory['tasks']) != inventory['tasks_id']:
        raise ValueError('Text inventory changed.')
    if (fingerprint({k: v for k, v in cached.items() if k != 'id'}) != cached['id']
            or cached['model_id'] != plan['text_model_id']):
        raise ValueError('Old text scores changed.')
    scores = dict(cached['predictions'])
    old_profile = cached['profile']
    if fingerprint({k: v for k, v in old_profile.items() if k != 'id'}) != old_profile['id']:
        raise ValueError('Preserved inference profile changed.')
    profiles = {old_profile['id']: old_profile}
    profile_ids = {key: old_profile['id'] for key in scores}
    categories = {}; identities = set()
    for row in bindings:
        identity = (row['record_id'], row['category'])
        if identity in identities:
            raise ValueError('Duplicate text binding.')
        identities.add(identity)
        if row['task_id'] in categories and categories[row['task_id']] != row['category']:
            raise ValueError('Conflicting task categories.')
        categories[row['task_id']] = row['category']
    for key, task in inventory['tasks'].items():
        if fingerprint([task['category'], task['text']]) != key or categories[key] != task['category']:
            raise ValueError('Text identity/category mismatch.')
    if set(scores) & set(inventory['tasks']) or set(scores) | set(inventory['tasks']) != set(categories):
        raise ValueError('Cached and unfinished inventories are not disjoint and complete.')
    exits = read_json(downloaded/'text-batched-output/exit.json')
    if exits != {'shard_0': 0, 'shard_1': 0}:
        raise ValueError('Scoring has not completed successfully.')
    switch = read_json(downloaded/'batch-switch.json')
    shard_counts = {}
    for shard in range(2):
        assigned = {key for i, key in enumerate(sorted(inventory['tasks'])) if i % 2 == shard}
        seen = set(); previous_hash = None
        for folder in ('text-output', 'text-batched-output'):
            path = downloaded/folder/f'shard-{shard}.sqlite'
            with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)) as db:
                if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                    raise ValueError('Corrupt inference cache.')
                metadata = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM metadata')}
                rows = list(db.execute('SELECT task_id,estimates FROM predictions'))
            binding, profile = metadata['binding'], metadata['profile']
            expected = dict(tasks_id=inventory['tasks_id'], model_id=plan['text_model_id'],
                            shard=shard, shards=2, precision='fp32')
            if any(binding[k] != v for k, v in expected.items()):
                raise ValueError('Shard belongs to another inference plan.')
            if (fingerprint({k: v for k, v in profile.items() if k != 'id'}) != profile['id']
                    or profile['checkpoint_id'] != plan['text_model_id']):
                raise ValueError('Invalid inference profile.')
            if folder == 'text-output':
                previous_hash = file_hash(path)
                saved = switch['files'][path.name]
                if saved != {'sha256': previous_hash, 'committed_tasks': len(rows)}:
                    raise ValueError('Preserved partial cache changed.')
            else:
                if binding['skipped_cache_sha256'] != previous_hash or binding['skipped_tasks'] != len(seen):
                    raise ValueError('Resumed shard skipped a different cache.')
                receipt = read_json(downloaded/folder/f'shard-{shard}-complete.json')
                if receipt['count'] != len(rows) or receipt['profile'] != profile:
                    raise ValueError('Completion receipt does not match the shard.')
                if any(receipt[k] != v for k, v in binding.items()):
                    raise ValueError('Completion binding changed.')
                check = read_json(downloaded/folder/f'shard-{shard}-batch-check.json')
                if check['maximum_absolute_difference'] > .001 or check['batch_size'] != profile['batch_size']:
                    raise ValueError('Batch numerical equivalence failed.')
            profiles[profile['id']] = profile
            for key, value in rows:
                if key not in assigned or key in scores:
                    raise ValueError('Duplicate or incorrectly assigned prediction.')
                scores[key] = json.loads(value); profile_ids[key] = profile['id']; seen.add(key)
            shard_counts[f'{folder}/{shard}'] = len(rows)
        if seen != assigned:
            raise ValueError('Incomplete shard coverage.')
    for key, values in scores.items():
        if set(values) != set(CATEGORIES[categories[key]][1]) or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100 for v in values.values()):
            raise ValueError('Invalid metric values.')
    by_record = Counter(row['record_id'] for row in bindings)
    if len(by_record) != plan['original_records'] or set(by_record.values()) != {4}:
        raise ValueError('Each original record must have four category results.')
    result = {'packet_id': plan['id'], 'model_id': plan['text_model_id'],
              'predictions': scores, 'task_profiles': profile_ids, 'profiles': profiles,
              'counts': {'texts': len(scores), 'records': len(by_record), 'cached': len(cached['predictions']),
                         'new': len(inventory['tasks']), 'shards': shard_counts},
              'experimental_uncalibrated': True, 'followup_text_invented': False}
    result['id'] = fingerprint(result)
    return result, bindings, plan


def append_database(database, output, scores, bindings, plan, evaluation, measurements):
    # PSEUDOCODE: verify the original snapshot -> byte-verified backup -> transactionally append result tables only.
    database, output, evaluation = Path(database), Path(output), Path(evaluation)
    if file_hash(database) != plan['source_database_sha256']:
        raise ValueError('Database changed after freezing; reconcile before appending results.')
    backup = output/'rhythm.before-results.sqlite'
    if backup.exists():
        raise ValueError('Refusing to overwrite the recovery copy.')
    shutil.copy2(database, backup)
    if file_hash(backup) != plan['source_database_sha256']:
        raise ValueError('Database backup failed verification.')
    manifest = verify_archive(evaluation)
    result = read_json(evaluation/'result.json')
    if result['packet_id'] != plan['id']:
        raise ValueError('Warning experiment belongs to a different packet.')
    now = datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(database)) as db:
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('BEGIN IMMEDIATE')
        if file_hash(database) != plan['source_database_sha256']:
            raise ValueError('Database changed before acquiring the write transaction.')
        original_tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        row_counts = {name: db.execute('SELECT count(*) FROM "'+name.replace('"', '""')+'"').fetchone()[0] for name in original_tables}
        new_tables = {'research_text_scores', 'research_text_bindings', 'research_expanded_forecasts', 'research_expanded_runs'}
        if new_tables & set(original_tables):
            raise ValueError('Expanded result tables already exist; do not silently replace a prior experiment.')
        with db:
            db.execute('CREATE TABLE research_expanded_runs (run_id TEXT PRIMARY KEY,created_at TEXT NOT NULL,payload TEXT NOT NULL)')
            db.execute('CREATE TABLE research_text_scores (run_id TEXT NOT NULL REFERENCES research_expanded_runs(run_id),task_id TEXT NOT NULL,profile_id TEXT NOT NULL,estimates TEXT NOT NULL,PRIMARY KEY(run_id,task_id))')
            db.execute('CREATE TABLE research_text_bindings (run_id TEXT NOT NULL,record_id TEXT NOT NULL REFERENCES observations(record_id),category TEXT NOT NULL,task_id TEXT NOT NULL,PRIMARY KEY(run_id,record_id,category),FOREIGN KEY(run_id,task_id) REFERENCES research_text_scores(run_id,task_id))')
            db.execute('CREATE TABLE research_expanded_forecasts (run_id TEXT NOT NULL REFERENCES research_expanded_runs(run_id),record_id TEXT NOT NULL REFERENCES research_followup_days(record_id),method TEXT NOT NULL,issued_at TEXT NOT NULL,score REAL,threshold REAL NOT NULL,warning INTEGER NOT NULL,status TEXT NOT NULL,PRIMARY KEY(run_id,record_id,method))')
            db.execute('INSERT INTO research_expanded_runs VALUES (?,?,?)', (plan['id'], now, canonical_json({
                'plan': plan, 'text_result_id': scores['id'], 'profiles': scores['profiles'],
                'warning_result': result, 'warning_archive_id': manifest['id']})))
            db.executemany('INSERT INTO research_text_scores VALUES (?,?,?,?)',
                [(plan['id'], key, scores['task_profiles'][key], canonical_json(value)) for key, value in scores['predictions'].items()])
            db.executemany('INSERT INTO research_text_bindings VALUES (?,?,?,?)',
                [(plan['id'], r['record_id'], r['category'], r['task_id']) for r in bindings])
            identities = {(r['participant_id'], instant(r['issued_at'])): r['record_id'] for r in measurements}
            for method, item in result['methods'].items():
                alerts = read_json(evaluation/(method+'-test-alerts.json'))
                db.executemany('INSERT INTO research_expanded_forecasts VALUES (?,?,?,?,?,?,?,?)', [
                    (plan['id'], identities[(r['participant_id'], instant(r['issued_at']))], method, r['issued_at'],
                     r['score'], item['threshold'], int(r['warning']), r['status']) for r in alerts])
            if db.execute('PRAGMA foreign_key_check').fetchall():
                raise ValueError('Result table references are invalid.')
            for name, count in row_counts.items():
                if db.execute('SELECT count(*) FROM "'+name.replace('"', '""')+'"').fetchone()[0] != count:
                    raise ValueError('Existing table row count changed.')
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Database integrity check failed after append.')
    save_json(output/'database-receipt.json', {'database': str(database.resolve()), 'before_sha256': file_hash(backup),
        'after_sha256': file_hash(database), 'backup': str(backup.resolve()), 'new_tables': sorted(new_tables),
        'existing_rows_preserved': row_counts, 'input_and_outcome_tables_modified': False})
