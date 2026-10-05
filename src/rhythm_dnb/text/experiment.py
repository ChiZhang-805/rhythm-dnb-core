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

PURPOSE = 'experimental_semantic_regression'
REFERENCE = 'model_semantic_reference_not_independent_human_gold'
ORIGINS = {'authored_simulation', 'translated_source', 'observed', 'translated_observed'}


def validate_rows(rows, roles):
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
                type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100
                for value in scores.values()):
            raise ValueError('Experiment needs complete, finite category references on the 0-100 scale.')
        key = fingerprint(''.join(text.split()))
        if key in texts and texts[key] != row['split']:
            raise ValueError('Duplicate text crosses experimental partitions.')
        texts[key] = row['split']
    validate_splits(rows, group_keys=('group_id', 'source_record_id', 'participant_id'))
    rows.sort(key=lambda row: row['example_id'])
    partitions = {role: [row for row in rows if row['split'] == role] for role in roles}
    if any({row['category'] for row in items} != set(CATEGORIES) for items in partitions.values()):
        raise ValueError('Every experimental partition must contain all five categories.')
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
    rows, partitions = validate_rows(payload['rows'], ('train', 'validation'))
    ordered = partitions['train'] + partitions['validation']
    manifest = payload['manifest']
    if manifest.get('purpose') != PURPOSE or manifest.get('reference') != REFERENCE or fingerprint(ordered) != manifest.get('development_id'):
        raise ValueError('Development corpus differs from the sealed experimental snapshot.')
    if any(manifest['counts'][role] != len(items) for role, items in partitions.items()):
        raise ValueError('Development partition counts differ from the snapshot.')
    groups = {key: sorted({row[key] for row in rows if row.get(key)}) for key in ('group_id', 'source_record_id', 'participant_id')}
    return partitions, {**manifest, 'development_groups': groups,
                        'development_text_hashes': sorted({fingerprint(''.join(row['text'].split())) for row in rows})}


def train_experiment(payload, base_path, output_dir, config, *, save_resume_state=False, resume_state=None):
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
                      development_only=True, experimental=True, save_resume_state=save_resume_state, resume_state=resume_state)


def validate_holdout(payload, manifest):
    # PSEUDOCODE: match the sealed test snapshot and reject identities/text that appeared during development.
    rows, _ = validate_rows(payload['rows'], ('test',))
    corpus = manifest['dataset']
    if (payload['manifest'].get('id') != corpus['id'] or payload['manifest'].get('purpose') != PURPOSE
            or fingerprint(rows) != corpus['test_id'] or len(rows) != corpus['counts']['test']):
        raise ValueError('Test corpus differs from the precommitted held-out snapshot.')
    for key, values in corpus['development_groups'].items():
        if set(values) & {row[key] for row in rows if row.get(key)}:
            raise ValueError('Test identity overlaps development: ' + key)
    if set(corpus['development_text_hashes']) & {fingerprint(''.join(row['text'].split())) for row in rows}:
        raise ValueError('Test text overlaps development.')
    return rows


def evaluate_experiment(payload, checkpoint, base_path, output_dir, device='auto'):
    # PSEUDOCODE: verify the chosen checkpoint/test seal -> evaluate once -> preserve scores and honest reference errors.
    from .checkpoint import inspect_checkpoint, load_checkpoint, save_checkpoint
    from .train import predict_rows
    from .evaluate import report
    from ..provenance import file_hash
    manifest, _ = inspect_checkpoint(checkpoint, allow_experimental=True)
    if manifest['purpose'] != PURPOSE:
        raise ValueError('Use the standard evaluation flow for a primary research checkpoint.')
    rows = validate_holdout(payload, manifest)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    model, tokenizer, loaded, _ = load_checkpoint(checkpoint, base_path=base_path, device=device, allow_experimental=True)
    model.eval()
    predictions = predict_rows(model, tokenizer, rows, loaded['config'], next(model.heads.parameters()).device)
    evaluation = report(rows, predictions, loaded['mean_baseline'], loaded['median_baseline'])
    evaluation.update(reference=REFERENCE, evidence_trained=False, evidence_calibrated=False, eligible_for_primary_dnb=False)
    metadata = {key: value for key, value in loaded.items() if key not in ('files', 'status', 'storage', 'contract', 'config', 'purpose')}
    final_path = save_checkpoint(output / 'model', model, tokenizer, loaded['config'], metadata, purpose=PURPOSE)
    identity = file_hash(final_path / 'manifest.json')
    # Raw evidence-head outputs are untrained in this experiment and are deliberately excluded from deliverables.
    points = {'identity': identity, 'reference': REFERENCE,
              'rows': [{'example_id': row['example_id'], 'category': row['category'], 'truth': row['scores'],
                        'scores': prediction['scores']} for row, prediction in zip(rows, predictions)]}
    result = {'best_checkpoint': str(final_path.resolve()), 'identity': identity, 'purpose': PURPOSE,
              'selected_epoch': loaded['epoch'], 'test_used_for_selection': False, 'test': evaluation,
              'corpus': loaded['dataset'], 'execution': loaded['execution']}
    (output / 'test-predictions.json').write_text(canonical_json(points), encoding='utf-8')
    (output / 'result.json').write_text(canonical_json(result), encoding='utf-8')
    return result
