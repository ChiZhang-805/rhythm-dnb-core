"""Explicit experiments on authored/model-scored references, with a sealed final test set."""

from collections import Counter
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3

from ..io.splits import validate_splits
from ..provenance import canonical_json, fingerprint
from .schema import CATEGORIES, validate_input
from .labels import validate_labels, has_supervision

PURPOSE = 'experimental_semantic_regression'
REFERENCE = 'model_semantic_reference_not_independent_human_gold'
ORIGINS = {'authored_simulation', 'translated_source', 'observed', 'translated_observed'}


def validate_rows(rows, roles, *, categories=None, allow_missing=False):
    # PSEUDOCODE: keep source identities and labels -> reject malformed rows and cross-role leakage.
    rows = [dict(row) for row in rows]
    ids, texts = set(), {}
    for row in rows:
        category, text = validate_input(row['category'], row['text'])
        row.update(category=category, text=text)
        for key in ('example_id', 'group_id'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError('Experiment requires original example and source-group identities.')
        if row['example_id'] in ids or row['split'] not in roles or row['origin'] not in ORIGINS:
            raise ValueError('Duplicate identity, unapproved source, or unexpected partition.')
        ids.add(row['example_id'])
        if row.get('annotation_source') != REFERENCE:
            raise ValueError('Experimental reference provenance must remain explicit.')
        scores = row.get('scores')
        if not isinstance(scores, dict) or set(scores) != set(CATEGORIES[category][1]) or any(
                not (allow_missing and value is None) and
                (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100)
                for value in scores.values()) or not has_supervision(row):
            raise ValueError('Experiment needs complete, finite category references on the 0-100 scale.')
        validate_labels(row)
        from .annotations import validate_scope_targets
        validate_scope_targets(row)
        if row.get('observed_at') is not None:
            from ..timebase import instant
            instant(row['observed_at'])
        key = fingerprint(''.join(text.split()))
        if key in texts and texts[key] != row['split']:
            raise ValueError('Duplicate text crosses experimental partitions.')
        texts[key] = row['split']
    validate_splits(rows, group_keys=('group_id', 'source_record_id', 'participant_id', 'family_id'))
    rows.sort(key=lambda row: row['example_id'])
    partitions = {role: [row for row in rows if row['split'] == role] for role in roles}
    if any({row['category'] for row in items} != set((categories or {}).get(role, CATEGORIES))
           for role, items in partitions.items()):
        raise ValueError('Experimental partition does not match its declared category coverage.')
    return rows, partitions


def export_corpus(database, output_dir):
    # PSEUDOCODE: read a consistent SQLite snapshot -> preserve authored labels/splits -> seal test separately.
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute('BEGIN')
        raw = list(connection.execute('SELECT * FROM chinese_examples ORDER BY example_id'))
        people = {row['record_id']: (row['dataset'], row['subject_id']) for row in
                  connection.execute('SELECT record_id, dataset, subject_id FROM records WHERE subject_id IS NOT NULL')}
    rows = []
    for raw_row in raw:
        row = {key: raw_row[key] for key in ('example_id', 'source_record_id', 'category', 'text', 'origin', 'split', 'group_id')}
        row.update(scores=json.loads(raw_row['scores']), annotation_source=REFERENCE)
        if row['source_record_id'] in people:
            dataset, person = people[row['source_record_id']]
            row['participant_id'] = dataset + ':' + person
        rows.append(row)
    rows, partitions = validate_rows(rows, ('train', 'validation', 'test'))
    development = partitions['train'] + partitions['validation']
    manifest = {'purpose': PURPOSE, 'reference': REFERENCE, 'id': fingerprint(rows),
                'development_id': fingerprint(development), 'test_id': fingerprint(partitions['test']),
                'counts': {role: len(items) for role, items in partitions.items()},
                'origins': dict(Counter(row['origin'] for row in rows)),
                'independent_human_gold': False, 'independent_people_verified': False,
                'split_unit': 'original_source_groups_and_known_participants', 'evidence_calibration': None}
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    for name, items in (('development', development), ('test', partitions['test'])):
        (output / (name + '.json')).write_text(canonical_json({'manifest': manifest, 'rows': items}), encoding='utf-8')
    (output / 'manifest.json').write_text(canonical_json(manifest), encoding='utf-8')
    return {**manifest, 'directory': str(output.resolve())}


def prepare_development(payload):
    # PSEUDOCODE: accept only sealed train/validation rows; never load test labels in the training process.
    rows, partitions = validate_rows(payload['rows'], ('train', 'validation'),
        categories=payload['manifest'].get('categories'), allow_missing=payload['manifest'].get('allow_missing', False))
    ordered = partitions['train'] + partitions['validation']
    manifest = payload['manifest']
    if manifest.get('purpose') != PURPOSE or manifest.get('reference') != REFERENCE or fingerprint(ordered) != manifest.get('development_id'):
        raise ValueError('Development corpus differs from the sealed experimental snapshot.')
    if any(manifest['counts'][role] != len(items) for role, items in partitions.items()):
        raise ValueError('Development partition counts differ from the snapshot.')
    groups = {key: sorted({row[key] for row in rows if row.get(key)}) for key in ('group_id', 'source_record_id', 'participant_id', 'family_id')}
    return partitions, {**manifest, 'development_groups': groups,
                        'development_text_hashes': sorted({fingerprint(''.join(row['text'].split())) for row in rows})}


def train_experiment(payload, base_path, output_dir, config, *, save_resume_state=False, resume_state=None, initialize_from=None):
    # PSEUDOCODE: verify the experimental snapshot and base -> share the normal optimizer -> select by validation only.
    from .config import validate_config
    from .weights import check_base
    from .runtime import TrainingRuntime
    from .train import _train
    config = validate_config(config)
    partitions, manifest = prepare_development(payload)
    base = Path(base_path).resolve()
    identity = check_base(base, config)
    with TrainingRuntime(config) as runtime:
        return _train(partitions, manifest, base, identity, Path(output_dir).resolve(), config, runtime,
                      development_only=True, experimental=True, save_resume_state=save_resume_state,
                      resume_state=resume_state, initialize_from=initialize_from)


def validate_holdout(payload, manifest, *, regression_only=False):
    # PSEUDOCODE: match the sealed test snapshot and reject identities/text that appeared during development.
    corpus = manifest['dataset']
    if corpus.get('test_usage', '').startswith('development_regression') and not regression_only:
        raise ValueError('This test was already exposed; use an explicit regression-only evaluation or a fresh sealed test.')
    rows, _ = validate_rows(payload['rows'], ('test',), categories=corpus.get('categories'),
                            allow_missing=corpus.get('allow_missing', False))
    if (payload['manifest'].get('id') != corpus['id'] or payload['manifest'].get('purpose') != PURPOSE
            or fingerprint(rows) != corpus['test_id'] or len(rows) != corpus['counts']['test']):
        raise ValueError('Test corpus differs from the precommitted held-out snapshot.')
    ancestor = manifest.get('initialization') or {}
    for key, values in corpus['development_groups'].items():
        values = set(values) | set(ancestor.get('exposure_groups', {}).get(key, []))
        if set(values) & {row[key] for row in rows if row.get(key)}:
            raise ValueError('Test identity overlaps development: ' + key)
    if (set(corpus['development_text_hashes']) | set(ancestor.get('exposure_text_hashes', []))) & {fingerprint(''.join(row['text'].split())) for row in rows}:
        raise ValueError('Test text overlaps development.')
    return rows


def evaluate_experiment(payload, checkpoint, base_path, output_dir, device='auto', *, regression_only=False):
    # PSEUDOCODE: verify the chosen checkpoint/test seal -> evaluate once -> preserve scores and honest reference errors.
    from .checkpoint import inspect_checkpoint, load_checkpoint, save_checkpoint
    from .train import predict_rows
    from .evaluate import report
    from ..provenance import file_hash
    manifest, _ = inspect_checkpoint(checkpoint, allow_experimental=True)
    if manifest['purpose'] != PURPOSE:
        raise ValueError('Use the standard evaluation flow for a primary research checkpoint.')
    rows = validate_holdout(payload, manifest, regression_only=regression_only)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    model, tokenizer, loaded, _ = load_checkpoint(checkpoint, base_path=base_path, device=device, allow_experimental=True)
    model.eval()
    predictions = predict_rows(model, tokenizer, rows, loaded['config'], next(model.heads.parameters()).device)
    evaluation = report(rows, predictions, loaded['mean_baseline'], loaded['median_baseline'])
    evaluation.update(reference=REFERENCE, evidence_trained=loaded.get('evidence_trained', False), evidence_calibrated=False, eligible_for_primary_dnb=False)
    metadata = {key: value for key, value in loaded.items() if key not in ('files', 'status', 'storage', 'contract', 'config', 'purpose')}
    final_path = save_checkpoint(output / 'model', model, tokenizer, loaded['config'], metadata, purpose=PURPOSE)
    identity = file_hash(final_path / 'manifest.json')
    # Experimental evidence remains uncalibrated even after auxiliary training; do not expose it as acceptance probability.
    points = {'identity': identity, 'reference': REFERENCE,
              'rows': [{'example_id': row['example_id'], 'category': row['category'], 'truth': row['scores'],
                        'scores': prediction['scores']} for row, prediction in zip(rows, predictions)]}
    result = {'best_checkpoint': str(final_path.resolve()), 'identity': identity, 'purpose': PURPOSE,
              'evaluation_usage': 'development_regression_only' if regression_only else 'sealed_test',
              'selected_epoch': loaded['epoch'], 'test_used_for_selection': False, 'test': evaluation,
              'corpus': loaded['dataset'], 'execution': loaded['execution']}
    (output / 'test-predictions.json').write_text(canonical_json(points), encoding='utf-8')
    (output / 'result.json').write_text(canonical_json(result), encoding='utf-8')
    return result


def seal_experiment(rows, previously_exposed, output_dir, *, initial_checkpoint_id=None):
    # PSEUDOCODE: validate proposed new roles -> reject all earlier exposure from held-out roles -> seal separately.
    from .expansion import template_identity
    rows, partitions = validate_rows(rows, ('train', 'validation', 'test'), allow_missing=True)
    group_keys = ('example_id', 'group_id', 'source_record_id', 'participant_id', 'family_id')
    previous = {key: {r[key] for r in previously_exposed if r.get(key)} for key in group_keys}
    templates = {template_identity(r['text']) for r in previously_exposed}
    for role in ('validation', 'test'):
        for row in partitions[role]:
            if any(row.get(key) in values for key, values in previous.items()) or template_identity(row['text']) in templates:
                raise ValueError('Fresh holdout overlaps an earlier training, validation or inspected test record: ' + row['example_id'])
    # Keep new paraphrase families and numeric templates together, including across the new train/validation roles.
    assigned = {}
    for row in rows:
        key = template_identity(row['text'])
        if key in assigned and assigned[key] != row['split']:
            raise ValueError('A numeric/text template crosses the new partitions.')
        assigned[key] = row['split']
    from .review import coverage_report
    coverage = coverage_report(rows)
    for role in ('train', 'validation', 'test'):
        missing = [key for key, item in coverage['partitions'][role]['metrics'].items() if not item['labeled']]
        if missing:
            raise ValueError(role + ' lacks numeric references for: ' + ', '.join(missing))
    development = partitions['train'] + partitions['validation']
    manifest = {'purpose': PURPOSE, 'reference': REFERENCE, 'id': fingerprint(rows),
        'development_id': fingerprint(development), 'test_id': fingerprint(partitions['test']),
        'counts': {k: len(v) for k, v in partitions.items()}, 'allow_missing': True,
        'initial_checkpoint_id': initial_checkpoint_id, 'exposure_inventory_id': fingerprint(previously_exposed),
        'origins': dict(Counter(r['origin'] for r in rows)), 'independent_human_gold': False,
        'independent_people_verified': False, 'test_usage': 'sealed', 'fresh_sealed_test': True,
        'known_limitations': ['freshness_is_relative_to_supplied_exposure_inventory', 'semantic_paraphrases_require_family_review']}
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    for name, items in (('development', development), ('test', partitions['test'])):
        (output / (name + '.json')).write_text(canonical_json({'manifest': manifest, 'rows': items}), encoding='utf-8')
    (output / 'manifest.json').write_text(canonical_json(manifest), encoding='utf-8')
    (output / 'coverage.json').write_text(canonical_json(coverage), encoding='utf-8')
    return manifest
