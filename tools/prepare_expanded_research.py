"""Freeze existing follow-up measurements, answers and unfinished text tasks before evaluation."""

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from rhythm_dnb.provenance import file_hash, fingerprint
from rhythm_dnb.research.experiment_data import save_json
from rhythm_dnb.research.simulated_followup import FEATURES, SETTINGS


def prepare(database, predictions, output):
    # PSEUDOCODE: read one immutable snapshot; preserve all identities, labels and already assigned splits.
    database, predictions, output = map(Path, (database, predictions, output))
    before = file_hash(database)
    cached = json.loads(predictions.read_text(encoding='utf-8'))
    if fingerprint({k: v for k, v in cached.items() if k != 'id'}) != cached['id']:
        raise ValueError('Preserved text scores changed.')
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        sequences = [dict(r) for r in db.execute('SELECT * FROM research_followup_sequences ORDER BY sequence_id')]
        followup = [dict(r) for r in db.execute('SELECT * FROM research_followup_days ORDER BY sequence_id,day_index')]
        originals = [dict(r) for r in db.execute('SELECT o.record_id,o.participant_id,o.source_dataset,r.payload '
                                               'FROM observations o JOIN raw_inputs r USING(record_id) ORDER BY o.record_id')]
        if len(originals) != db.execute('SELECT count(*) FROM observations').fetchone()[0]:
            raise ValueError('Missing source records.')
    people = [r['parent_participant_id'] for r in sequences]
    if len(people) != len(set(people)) or {r['split'] for r in sequences} != {'train', 'calibration', 'test'}:
        raise ValueError('Follow-up parent identity or split mismatch.')
    by_sequence = {r['sequence_id']: r for r in sequences}
    measurements, answers = [], []
    for row in followup:
        seq = by_sequence[row['sequence_id']]
        values = json.loads(row['features_json'])
        if set(values) != set(FEATURES):
            raise ValueError('Unexpected objective measurements.')
        measurements.append({'record_id': row['record_id'], 'participant_id': row['sequence_id'],
            'parent_participant_id': seq['parent_participant_id'], 'split': seq['split'],
            'day_index': row['day_index'], 'observed_at': row['simulated_at'],
            'issued_at': row['simulated_at'], 'features': values})
        answers.append({k: row[k] for k in ('record_id', 'future_event_7d', 'future_label_status', 'regime_state')})
    tasks, bindings = {}, []
    for row in originals:
        raw = json.loads(row.pop('payload'))
        for category in ('emotion', 'sleep', 'diet', 'social'):
            text = raw[category + '_description_text'].strip()
            if not text:
                raise ValueError('Empty source text.')
            key = fingerprint([category, text]); tasks[key] = {'category': category, 'text': text}
            bindings.append({**row, 'category': category, 'task_id': key})
    if not set(cached['predictions']) <= set(tasks):
        raise ValueError('Preserved predictions have no matching original text.')
    remaining = {key: value for key, value in tasks.items() if key not in cached['predictions']}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('measurements', measurements), ('answers', answers), ('sequences', sequences),
                        ('text-bindings', bindings), ('cached-text-scores', cached),
                        ('text-tasks', {'tasks': remaining, 'tasks_id': fingerprint(remaining)})]:
        save_json(output / (name + '.json'), value)
    root = Path(__file__).resolve().parents[1]
    temporal = json.loads((root / 'configs/temporal_warning.json').read_text(encoding='utf-8'))
    config = {'seed': SETTINGS['seed'], 'baseline_days': SETTINGS['baseline_days'],
              'history_windows': [7, 14, 28], 'rolling_windows': [7, 14, 28],
              'horizon_days': 7, 'min_lead_hours': 24, 'confirmation_days': 2,
              'consecutive': 2, 'cooldown_days': 7, 'max_false_alarms_per_30_days': 1.,
              'epsilon': 1e-8, 'cluster_bootstrap_repetitions': 2000, 'temporal': temporal,
              'primary_method': 'personal_dnb', 'secondary_method': 'history_dnb',
              'methods': ['personal_dnb', 'rolling_dnb', 'mean_deviation', 'history_control', 'history_dnb'],
              'threshold_selection': 'calibration_balanced_accuracy_with_preserved_alarm_budget',
              'primary_metric': 'risk_balanced_accuracy',
              'parameter_basis': 'Preserve authored baseline, horizon, existing 7/14/28-day window grid and '
                  'alarm policy; reuse the same grouped model capacity grid for matched arms. '
                  'Select windows/smoothing on development only, thresholds on calibration only.',
              'dnb_interpretation': 'Exploratory personal stable-day reference and rolling DNB scores; '
                  'not a claim that classic population-module discovery passed.',
              'no_new_text_for_followup': True, 'no_outcome_relabelling': True}
    save_json(output / 'config.json', config)
    files = {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_database_sha256': before,
            'source_database': str(database.resolve()), 'domain': SETTINGS['domain'],
            'followup_plan_ids': sorted({r['run_id'] for r in sequences}), 'files': files,
            'split_counts': dict(Counter(r['split'] for r in sequences)),
            'parent_people': len(people), 'followup_records': len(measurements),
            'original_records': len(originals), 'total_records': len(originals)+len(measurements),
            'new_independent_real_people': 0, 'cached_text_tasks': len(cached['predictions']),
            'pending_text_tasks': len(remaining), 'text_model_id': cached['model_id'],
            'followup_without_text': len(measurements), 'source_database_unchanged': True,
            'eligibility': 'Original histories lacking independent rhythm endpoints remain quantification-only; '
                 'all 245 authored follow-up sequences participate in the existing train/calibration/test split.',
            'test_results_inspected_before_freeze': False}
    plan['id'] = fingerprint(plan)
    if file_hash(database) != before:
        raise ValueError('Database changed during preparation.')
    save_json(output / 'plan.json', plan)
    print(json.dumps({k: v for k, v in plan.items() if k != 'files'}, ensure_ascii=False))


def main():
    # PSEUDOCODE: freeze into a new folder; never replace earlier experiment packets.
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('database', 'predictions', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    prepare(args.database, args.predictions, args.output)


if __name__ == '__main__':
    main()
