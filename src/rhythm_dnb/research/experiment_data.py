"""Read-only, traceable preparation of the existing authored longitudinal experiment."""

from collections import Counter, defaultdict
from contextlib import closing
from datetime import timedelta
import json
import math
from pathlib import Path
import sqlite3

from ..provenance import canonical_json, file_hash, fingerprint
from ..timebase import instant
from ..text.expansion import template_identity, screen_weak_absence, MENTIONS
from ..text.experiment import PURPOSE, REFERENCE, prepare_development
from ..text.schema import CATEGORIES

OBJECTIVE = ('sleep_midpoint_h', 'sleep_duration_h', 'meal_interval_cv',
             'exercise_minutes', 'resting_hr_bpm', 'screen_time_min')
TEXT = ('text_anxiety_intensity', 'text_sadness_intensity',
        'text_post_sleep_fatigue', 'text_meal_irregularity_intensity')
DOMAIN = 'authored_simulation_not_clinical_validation'


def read_snapshot(database):
    # PSEUDOCODE: hold a read transaction -> join the saved originals -> verify immutable bytes.
    database = Path(database).resolve()
    before = file_hash(database)
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok' or db.execute('PRAGMA foreign_key_check').fetchone():
            raise ValueError('Source database integrity check failed.')
        rows = [dict(r) for r in db.execute('SELECT o.*, r.payload AS raw, p.payload AS provenance '
                 'FROM observations o JOIN raw_inputs r USING(record_id) '
                 'JOIN observation_provenance p USING(record_id) ORDER BY o.record_id')]
        if len(rows) != db.execute('SELECT count(*) FROM observations').fetchone()[0]:
            raise ValueError('Missing or duplicate raw/provenance records.')
    if file_hash(database) != before:
        raise ValueError('Source changed while exporting; retry from a stable snapshot.')
    return rows, before


def cohort_rows(source):
    # PSEUDOCODE: retain only explicitly simulated sequences -> derive clock midpoint -> preserve outcomes separately.
    rows, corrections, seen, roles = [], [], set(), {}
    for original in source:
        if original['source_dataset'] != 'SIMULATED-RHYTHM':
            continue
        if original['label_source'] != 'synthetic_scenario_not_clinical_ground_truth':
            raise ValueError('Unexpected outcome provenance in simulation cohort.')
        raw = json.loads(original['raw'])
        observed = instant(original['observed_at'])
        start, end = instant(raw['sleep_start_time']), instant(raw['sleep_end_time'])
        if not start < end <= observed or end - start > timedelta(days=1):
            raise ValueError('Invalid or future sleep interval: ' + original['record_id'])
        middle = start + (end - start) / 2
        values = {k: original[k] for k in OBJECTIVE if k != 'sleep_midpoint_h'}
        values['sleep_midpoint_h'] = (middle.hour + middle.minute / 60 + middle.second / 3600 + middle.microsecond / 3.6e9)
        if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError('Incomplete or invalid experimental feature values.')
        person, role = original['participant_id'], original['split']
        if role not in ('reference', 'train', 'validation', 'test') or roles.setdefault(person, role) != role:
            raise ValueError('Participant crosses cohort roles.')
        key = (person, observed.date())
        if key in seen:
            raise ValueError('Duplicate person/day; do not silently average.')
        seen.add(key)
        # Midnight-to-midnight rows are conservatively issued at noon on the following day.
        issued = (observed + timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
        texts = {c: raw[c + '_description_text'].strip() for c in ('emotion', 'diet', 'sleep', 'social')}
        if any(not text for text in texts.values()):
            raise ValueError('Empty cohort description.')
        rows.append({'record_id': original['record_id'], 'participant_id': person, 'split': role,
                     'observed_at': observed.isoformat(), 'issued_at': issued.isoformat(),
                     'features': values, 'texts': texts,
                     'outcome': {k: original[k] for k in ('event_onset_at', 'followup_end_at', 'label_observed_at')},
                     'domain': DOMAIN})
        corrections.append({'record_id': original['record_id'], 'field': 'sleep_midpoint_h',
                            'method': 'midpoint_of_aware_sleep_interval_UTC', 'value': values['sleep_midpoint_h']})
    if set(roles.values()) != {'reference', 'train', 'validation', 'test'}:
        raise ValueError('All four independent cohort roles are required.')
    return rows, corrections


def split_text_groups(rows, seed, validation_fraction=.2):
    # PSEUDOCODE: union shared identities and numeric templates -> assign entire connected families by a seeded hash.
    parent = list(range(len(rows)))
    def find(i):
        # PSEUDOCODE: compress paths to the identity component root.
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        keys = [(k, row[k]) for k in ('group_id', 'source_record_id', 'participant_id', 'family_id') if row.get(k)]
        keys.append(('template', template_identity(row['text'])))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[find(i)].append(row)
    result = []
    for group in components.values():
        identity = min(r['example_id'] for r in group)
        number = int(fingerprint([seed, identity])[:16], 16) / 2 ** 64
        role = 'validation' if number < validation_fraction else 'train'
        result.extend({**r, 'split': role} for r in group)
    return sorted(result, key=lambda r: r['example_id'])


def prepare_text(source, cohort, corpus, master_database, seed):
    # PSEUDOCODE: refresh existing development texts -> exclude the whole DNB cohort -> freeze person/family-separated text roles.
    import re
    records = {r['record_id']: r for r in source}
    excluded_people = {'rhythm:' + r['participant_id'] for r in cohort}
    excluded_templates = {template_identity(t) for r in cohort for t in r['texts'].values()}
    payload = json.loads(Path(corpus).read_text(encoding='utf-8'))
    prepare_development(payload)
    original_rows = list(payload['rows'])
    master_database = Path(master_database).resolve()
    master_hash = file_hash(master_database)
    with closing(sqlite3.connect(master_database.as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        for r in db.execute("SELECT * FROM chinese_examples WHERE category='stress' AND split IN ('train','validation')"):
            if r['example_id'] not in {item['example_id'] for item in original_rows}:
                original_rows.append({k: r[k] for k in ('example_id', 'group_id', 'source_record_id', 'text', 'category', 'origin', 'split')} |
                                     {'scores': json.loads(r['scores']), 'annotation_source': REFERENCE})
    kept, changes, exclusions = [], [], []
    for original in original_rows:
        row = dict(original)
        rid = row.get('source_record_id') or ''
        if rid.startswith('rhythm:'):
            saved = records.get(rid[7:])
            if saved is None or row.get('participant_id') != 'rhythm:' + saved['participant_id']:
                raise ValueError('Text identity differs from current database.')
            raw, provenance = json.loads(saved['raw']), json.loads(saved['provenance'])['provenance']
            text = raw[row['category'] + '_description_text'].strip()
            if row.get('label_states') or row.get('scope_targets'):
                raise ValueError('Reviewed/scope annotations require explicit reconciliation before source refresh.')
            keys = CATEGORIES[row['category']][1]
            if any(provenance.get('text_' + k, {}).get('method') != 'manual_contextual_rewrite' for k in keys):
                raise ValueError('Unexpected text reference origin.')
            scores = {k: saved['text_' + k] if re.search(MENTIONS[k], text) else None for k in keys}
            row.update(text=text, scores=scores, observed_at=saved['observed_at'],
                       temporal_basis='source_timestamp_for_authored_text_not_observed_patient_text')
            if text != original['text'] or scores != original['scores']:
                changes.append({'example_id': row['example_id'], 'old_hash': fingerprint(original), 'new_hash': fingerprint(row)})
        reason = ('DNB_participant' if row.get('participant_id') in excluded_people else
                  'DNB_text_template' if template_identity(row['text']) in excluded_templates else None)
        if reason:
            exclusions.append({'example_id': row['example_id'], 'reason': reason})
        else:
            kept.append(row)
    kept, masked = screen_weak_absence(kept)
    kept = split_text_groups(kept, seed)
    partitions = {role: [r for r in kept if r['split'] == role] for role in ('train', 'validation')}
    ordered = partitions['train'] + partitions['validation']
    manifest = {'purpose': PURPOSE, 'reference': REFERENCE, 'id': fingerprint(kept),
                'development_id': fingerprint(ordered), 'counts': {k: len(v) for k, v in partitions.items()},
                'allow_missing': True, 'independent_human_gold': False, 'independent_people_verified': False,
                'test_usage': 'text_validation_is_development_only; DNB_test_is_separate',
                'initial_checkpoint_id': None, 'initialization_required': 'pinned_base_only',
                'excluded_dnb_people': sorted(excluded_people),
                'excluded_dnb_templates_id': fingerprint(sorted(excluded_templates))}
    development = {'manifest': manifest, 'rows': ordered}
    prepare_development(development)
    coverage = {role: {key: sum(r['scores'].get(key) is not None for r in rows) for _, keys in CATEGORIES.values() for key in keys}
                for role, rows in partitions.items()}
    if any(not count for counts in coverage.values() for count in counts.values()):
        raise ValueError('A text metric has no training or validation references: ' + str(coverage))
    if file_hash(master_database) != master_hash:
        raise ValueError('Master database changed during preparation.')
    return development, {'source_refresh': changes, 'exclusions': exclusions, 'masked_references': masked,
                         'coverage': coverage, 'master_sha256': master_hash,
                         'validation_fraction': .2, 'partition_unit': 'connected_person_source_family_numeric_template'}


def save_json(path, payload):
    # PSEUDOCODE: refuse overwrite -> write canonical portable scientific evidence.
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(canonical_json(payload))


def prepare_packet(database, corpus, master_database, output, protocol):
    # PSEUDOCODE: create a new portable packet containing source receipts, outcome separation and a leakage-free development corpus.
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory; existing evidence is preserved.')
    source, digest = read_snapshot(database)
    cohort, corrections = cohort_rows(source)
    development, audit = prepare_text(source, cohort, corpus, master_database, protocol['seed'])
    output.mkdir(parents=True)
    for role in ('reference', 'train', 'validation', 'test'):
        selected = [r for r in cohort if r['split'] == role]
        save_json(output / (role + '.json'), selected)
    save_json(output / 'text-development.json', development)
    save_json(output / 'protocol.json', protocol)
    save_json(output / 'data-audit.json', {'domain': DOMAIN, 'source_sha256': digest, 'source_rows': len(source),
        'source_counts': dict(Counter(r['source_dataset'] for r in source)),
        'cohort_rows': len(cohort), 'derived_fields': corrections, 'text': audit,
        'limitations': ['existing_authored_outcomes_not_independent_clinical_events',
                       'historical_test_previously_inspected_not_new_prospective_validation',
                       'source_followup_does_not_replace_daily_outcome_coverage',
                       'text_evidence_uncalibrated; joint_branch_is_exploratory'], 'source_unchanged': True})
    files = {p.name: file_hash(p) for p in sorted(output.glob('*.json'))}
    manifest = {'domain': DOMAIN, 'files': files, 'id': fingerprint(files),
                'row_counts': dict(Counter(r['split'] for r in cohort)),
                'text_counts': development['manifest']['counts'], 'source_sha256': digest}
    save_json(output / 'manifest.json', manifest)
    return manifest


def load_packet(directory, *, names=None):
    # PSEUDOCODE: verify every sealed file before returning any experiment inputs.
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['domain'] != DOMAIN or fingerprint(manifest['files']) != manifest['id']:
        raise ValueError('Invalid experiment packet identity.')
    for name, digest in manifest['files'].items():
        if Path(name).name != name or file_hash(directory / name) != digest:
            raise ValueError('Experiment packet changed: ' + name)
    return {name[:-5]: json.loads((directory / name).read_text(encoding='utf-8')) for name in manifest['files']
            if names is None or name[:-5] in names}, manifest
