"""Export one compact experiment sheet without changing source data or saved results."""

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from itertools import zip_longest
import json
import math
from pathlib import Path
import shutil
import sqlite3

from openpyxl import Workbook, load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from rhythm_dnb.provenance import file_hash, fingerprint
from rhythm_dnb.research.dataset_annotations import daily_values
from rhythm_dnb.research.experiment_data import load_packet, save_json
from rhythm_dnb.research.simulated_followup import FEATURES
from rhythm_dnb.text.schema import CATEGORIES, FORMAL_CATEGORIES, METRICS
from rhythm_dnb.timebase import instant


TEXT_METRICS = [(key, label) for key, category, label, _ in METRICS if category in FORMAL_CATEGORIES]
METHODS = {
    'reference_robust_dnb': 'DNB',
    'reference_robust_deviation': '简单对照',
    'history_robust_dnb': '联合DNB',
    'history_robust_control': '联合对照',
}
# The source group and parent ID are needed to prevent mixing cohorts or leaking a person across splits.
COLUMNS = [
    ('record_id', '记录编号', 'identity'),
    ('participant_id', '序列编号', 'identity'),
    ('parent_participant_id', '人物编号', 'identity'),
    ('source_dataset', '数据分组', 'identity'),
    ('split', '训练或测试分组', 'identity'),
    ('observed_at', '记录时间', 'identity'),
    ('sleep_start_time', '入睡时间', 'input'),
    ('sleep_end_time', '起床时间', 'input'),
    ('emotion_description_text', '情绪文本', 'input'),
    ('sleep_description_text', '睡眠文本', 'input'),
    ('diet_description_text', '饮食文本', 'input'),
    ('social_description_text', '社交文本', 'input'),
    ('sleep_midpoint_h', '睡眠中点（小时）', 'input'),
    ('sleep_duration_h', '睡眠时长（小时）', 'input'),
    ('meal_interval_cv', '餐间隔不规则程度', 'input'),
    ('exercise_hour', '运动时刻（小时）', 'input'),
    ('exercise_minutes', '运动时长（分钟）', 'input'),
    ('resting_hr_bpm', '静息心率', 'input'),
    ('screen_time_min', '屏幕使用（分钟）', 'input'),
] + [('text_' + key, '参考·' + label, 'reference') for key, label in TEXT_METRICS] + [
    ('model_' + key, '模型·' + label, 'model') for key, label in TEXT_METRICS
] + [
    ('rhythm_state', '当日是否紊乱', 'truth'),
    ('event_in_followup', '随访期间是否紊乱', 'truth'),
    ('event_onset_at', '紊乱开始时间', 'truth'),
    ('event_confirmed_at', '紊乱确认时间', 'truth'),
    ('followup_end_at', '随访结束时间', 'truth'),
    ('label_observed_at', '参考答案可获得时间', 'truth'),
    ('issued_at', '预测时点', 'truth'),
    ('future_event_7d', '未来7天是否开始紊乱', 'truth'),
] + [item for method, label in METHODS.items() for item in (
    (method + '_score', label + '分数', 'warning'),
    (method + '_warning', label + '预警', 'warning'),
)]
COLORS = dict(identity='334155', input='275D72', reference='627447',
              model='326E63', truth='8A6538', warning='675685')


def read_json(path):
    # PSEUDOCODE: read an existing immutable artifact; no inference or label generation.
    return json.loads(Path(path).read_text(encoding='utf-8'))


def unique(rows, key):
    # PSEUDOCODE: reject duplicate identities instead of silently selecting a result.
    result = {}
    for row in rows:
        identity = key(row)
        if identity in result:
            raise ValueError('Duplicate identity: ' + str(identity))
        result[identity] = row
    return result


def saved_results(packet, predictions, decisions):
    # PSEUDOCODE: verify historical hashes and exact text bindings, without requiring today's code identity.
    data, manifest = load_packet(packet)
    if data['protocol']['horizon_days'] != 7:
        raise ValueError('This projection requires a seven-day forecast endpoint.')
    rows = unique([r for role in ('reference', 'train', 'validation', 'test') for r in data[role]],
                  lambda r: r['record_id'])
    tasks = {fingerprint([category, text]): {'category': category, 'text': text}
             for row in rows.values() for category, text in row['texts'].items()}
    result = read_json(predictions)
    if (fingerprint({k: v for k, v in result.items() if k != 'id'}) != result['id']
            or result['packet_id'] != manifest['id'] or result['tasks_id'] != fingerprint(tasks)
            or result['checkpoint_dataset_id'] != data['text-development']['manifest']['id']
            or set(result['predictions']) != set(tasks)):
        raise ValueError('Text predictions differ from the sealed packet.')
    for key, values in result['predictions'].items():
        if set(values) != set(CATEGORIES[tasks[key]['category']][1]) or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100 for v in values.values()):
            raise ValueError('Invalid text prediction.')
    archive = read_json(Path(decisions).parent / 'archive-manifest.json')
    if file_hash(decisions) != archive['files'][Path(decisions).name]:
        raise ValueError('Saved forecast answers changed.')
    answers = unique(read_json(decisions), lambda r: (r['participant_id'], instant(r['issued_at'])))
    return rows, result, answers, manifest['id']


def project_database(database, packet_rows, predictions, answers):
    # PSEUDOCODE: join only matching saved records; keep unknowns blank and follow-up branches distinct.
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Database integrity check failed.')
        for table in ('research_annotation_runs', 'research_followup_runs'):
            if db.execute('SELECT count(*) FROM ' + table).fetchone()[0] != 1:
                raise ValueError('Select one reconciled run before exporting: ' + table)
        annotations = unique([dict(r) for r in db.execute('SELECT * FROM research_outcome_annotations')],
                             lambda r: r['record_id'])
        warnings = unique([dict(r) for r in db.execute('SELECT * FROM research_warning_annotations')],
                          lambda r: (r['record_id'], r['method']))
        originals = [dict(r) for r in db.execute(
            'SELECT o.*, r.payload AS raw FROM observations o JOIN raw_inputs r USING(record_id) ORDER BY o.record_id')]
        if len(originals) != db.execute('SELECT count(*) FROM observations').fetchone()[0]:
            raise ValueError('Missing raw inputs.')
        followup = [dict(r) for r in db.execute(
            'SELECT d.*, s.parent_participant_id, s.split, s.event_in_followup, s.first_onset_date, '
            's.first_confirmed_date, s.followup_end_date FROM research_followup_days d '
            'JOIN research_followup_sequences s USING(sequence_id) ORDER BY d.record_id')]
        if len(followup) != db.execute('SELECT count(*) FROM research_followup_days').fetchone()[0]:
            raise ValueError('Missing follow-up sequence.')
    rows, matched_text, matched_answers, matched_warnings = [], set(), set(), set()
    for original in originals:
        row = dict(original)
        raw = json.loads(row.pop('raw'))
        row.update({k: raw.get(k) for k in ('sleep_start_time', 'sleep_end_time',
                   'emotion_description_text', 'sleep_description_text', 'diet_description_text', 'social_description_text')})
        row['parent_participant_id'] = row['participant_id']
        row['sleep_midpoint_h'] = daily_values(original)[0]
        annotation = annotations[row['record_id']]
        # A prevalent sleep-phase diagnosis is not the rhythm-warning outcome.
        row['rhythm_state'] = original['rhythm_state_gt'] if annotation['reference_scope'] == 'scenario' else None
        row['event_in_followup'] = annotation['reference_event_in_followup']
        row['event_onset_at'] = annotation['reference_onset_at']
        sealed = packet_rows.get(row['record_id'])
        if sealed:
            if (sealed['participant_id'] != row['participant_id'] or sealed['split'] != row['split']
                    or instant(sealed['observed_at']) != instant(row['observed_at'])):
                raise ValueError('Record identity differs from the saved packet.')
            for key, value in sealed['features'].items():
                if key != 'sleep_midpoint_h' and row[key] != value:
                    raise ValueError('Sealed objective input changed: ' + key)
            row['sleep_midpoint_h'] = sealed['features']['sleep_midpoint_h']
            row['issued_at'] = sealed['issued_at']
            for category, text in sealed['texts'].items():
                if raw[category + '_description_text'].strip() != text:
                    raise ValueError('Text changed since inference: ' + row['record_id'])
                row.update({'model_' + k: v for k, v in predictions[fingerprint([category, text])].items()})
            matched_text.add(row['record_id'])
        for method in METHODS:
            identity = (row['record_id'], method)
            warning = warnings.get(identity)
            if warning:
                if not sealed or instant(warning['issued_at']) != instant(row['issued_at']):
                    raise ValueError('Warning cutoff differs from the saved input record.')
                row[method + '_score'], row[method + '_warning'] = warning['score'], warning['warning']
                matched_warnings.add(identity)
        answer_key = (row['participant_id'], instant(row['issued_at'])) if sealed else None
        if answer_key in answers:
            answer = answers[answer_key]
            if answer['label'] not in (0, 1):
                raise ValueError('Invalid saved future label.')
            row['future_event_7d'] = answer['label']
            matched_answers.add(answer_key)
        rows.append(row)
    for original in followup:
        values = json.loads(original['features_json'])
        if set(values) != set(FEATURES):
            raise ValueError('Unexpected follow-up inputs.')
        rows.append({**values, 'record_id': original['record_id'], 'participant_id': original['sequence_id'],
                     'parent_participant_id': original['parent_participant_id'], 'split': original['split'],
                     'source_dataset': 'SIMULATED-FOLLOWUP', 'observed_at': original['simulated_at'],
                     'issued_at': original['simulated_at'], 'rhythm_state': original['regime_state'],
                     'event_in_followup': original['event_in_followup'], 'event_onset_at': original['first_onset_date'],
                     'event_confirmed_at': original['first_confirmed_date'], 'followup_end_at': original['followup_end_date'],
                     'future_event_7d': original['future_event_7d']})
    unique(rows, lambda r: r['record_id'])
    if matched_text != set(packet_rows) or matched_answers != set(answers) or matched_warnings != set(warnings):
        raise ValueError('Not every saved result was joined exactly once.')
    return rows, {'original_rows': len(originals), 'followup_rows': len(followup),
                  'text_prediction_rows': len(matched_text), 'forecast_answer_rows': len(matched_answers),
                  'warning_values': len(matched_warnings)}


def write_workbook(rows, path):
    # PSEUDOCODE: export one visible, filtered sheet; display decimals without changing stored values.
    wb = Workbook(write_only=True)
    ws = wb.create_sheet('实验数据')
    ws.freeze_panes = 'D2'
    ws.sheet_view.zoomScale = 85
    ws.row_dimensions[1].height = 42
    ws.auto_filter.ref = f'A1:{get_column_letter(len(COLUMNS))}{len(rows) + 1}'
    header = []
    for i, (key, label, group) in enumerate(COLUMNS, 1):
        cell = WriteOnlyCell(ws, label)
        cell.font = Font(name='Microsoft YaHei', bold=True, color='FFFFFF', size=10)
        cell.fill = PatternFill('solid', fgColor=COLORS[group])
        cell.alignment = Alignment(vertical='center', wrap_text=True)
        header.append(cell)
        width = 45 if key.endswith('_text') else 27 if 'time' in key or key.endswith('_at') else 21
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.append(header)
    for row in rows:
        cells = []
        for key, _, _ in COLUMNS:
            value = row.get(key)
            cell = WriteOnlyCell(ws, value)
            if isinstance(value, str):
                if len(value) > 32767:
                    raise ValueError('Excel would truncate an input text.')
                cell.data_type = 's'  # Never interpret input text as an Excel formula.
            elif type(value) is float:
                if not math.isfinite(value):
                    raise ValueError('Nonfinite experimental value.')
                cell.number_format = '0.00'
            cells.append(cell)
        ws.append(cells)
    wb.save(path)


def verify_workbook(path, rows):
    # PSEUDOCODE: compare every exported cell, including blank/zero distinctions and float precision.
    wb = load_workbook(path, read_only=True, data_only=False)
    try:
        if wb.sheetnames != ['实验数据']:
            raise ValueError('Unexpected worksheets.')
        values = wb.active.iter_rows(values_only=True, max_col=len(COLUMNS))
        if next(values) != tuple(label for _, label, _ in COLUMNS):
            raise ValueError('Unexpected column headers.')
        seen = set()
        for expected, actual in zip_longest(rows, values):
            if expected is None or actual is None or len(actual) != len(COLUMNS):
                raise ValueError('Exported row count or width differs.')
            if actual[0] in seen:
                raise ValueError('Duplicate exported record.')
            seen.add(actual[0])
            for (key, _, _), value in zip(COLUMNS, actual):
                wanted = expected.get(key)
                if type(wanted) is float and type(value) in (int, float):
                    equal = math.isclose(wanted, value, rel_tol=1e-14, abs_tol=1e-12)
                else:
                    equal = wanted == value or wanted == '' and value is None
                if not equal:
                    raise ValueError(f'Export changed {expected["record_id"]}: {key}')
        return {'rows': len(seen), 'columns': len(COLUMNS), 'cells_checked': len(seen) * len(COLUMNS)}
    finally:
        wb.close()


def verify_existing_inputs(path, rows):
    # PSEUDOCODE: detect spreadsheet-only edits before replacing the view with database values.
    expected = {r['record_id']: r for r in rows}
    keys = {k for k, _, group in COLUMNS if group in ('identity', 'input', 'reference')}
    names = {label: key for key, label, _ in COLUMNS}
    wb = load_workbook(path, read_only=True)
    try:
        sheet = wb.worksheets[0]
        values = sheet.iter_rows(values_only=True)
        headers = [names.get(str(v), str(v).split('\n')[-1]) for v in next(values)]
        legacy = 'text_model_id' in headers
        if 'record_id' not in headers:
            raise ValueError('Unrecognized workbook; refusing to overwrite a different view.')
        checked = 0
        for actual in values:
            source = dict(zip(headers, actual))
            target = expected[source['record_id']]
            for key in keys.intersection(source):
                value, wanted = source[key], target.get(key)
                equal = (math.isclose(wanted, value, rel_tol=1e-14, abs_tol=1e-12)
                         if type(wanted) is float and type(value) in (int, float)
                         else wanted == value or wanted == '' and value is None)
                if not equal and not (legacy and value == legacy_display(key, wanted)):
                    raise ValueError('Workbook-only edit requires reconciliation: ' + source['record_id'] + '/' + key)
                checked += 1
        return checked
    finally:
        wb.close()


def legacy_display(key, value):
    # PSEUDOCODE: recognize the old view's minute/integer formatting, but export full database precision.
    if value is None:
        return None
    if key in ('sleep_start_time', 'sleep_end_time'):
        return datetime.fromisoformat(value.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
    if key == 'sleep_duration_h':
        hours, minutes = divmod(max(0, int(round(value * 60))), 60)
        return f'{hours}小时{minutes}分钟' if hours and minutes else f'{hours}小时' if hours else f'{minutes}分钟'
    if key == 'exercise_hour':
        minutes = min(1439, int(round((value % 24) * 60)))
        return f'{minutes // 60:02d}:{minutes % 60:02d}'
    if key in ('exercise_minutes', 'resting_hr_bpm', 'screen_time_min') or key in {'text_' + k for k, _ in TEXT_METRICS}:
        return int(round(value))
    if key == 'meal_interval_cv':
        return round(value, 2)
    return value


def main():
    # PSEUDOCODE: build and verify a fresh projection -> back up the old view -> atomically replace it.
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('database', 'workbook', 'archive', 'packet', 'text-predictions', 'decisions'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    paths = {name: getattr(args, name).resolve() for name in
             ('database', 'workbook', 'archive', 'packet', 'text_predictions', 'decisions')}
    if len(set(paths.values())) != len(paths):
        raise ValueError('Input and output paths must be distinct.')
    args.archive.mkdir(parents=True, exist_ok=True)
    if any(args.archive.iterdir()):
        raise ValueError('Use an empty archive directory; existing evidence must not be overwritten.')
    before = {key: file_hash(paths[key]) for key in ('database', 'workbook', 'text_predictions', 'decisions')}
    old = load_workbook(args.workbook, read_only=True)
    try:
        old_sheets = old.sheetnames
        old_columns = [str(v).split('\n')[-1] for v in next(old.worksheets[0].values)]
    finally:
        old.close()
    packet_rows, result, answers, packet_id = saved_results(args.packet, args.text_predictions, args.decisions)
    rows, counts = project_database(args.database, packet_rows, result['predictions'], answers)
    print(json.dumps({'stage': 'source_join_verified', **counts}), flush=True)
    existing_checked = verify_existing_inputs(args.workbook, rows)
    staged = args.archive / 'quantified_data.xlsx'
    write_workbook(rows, staged)
    checked = verify_workbook(staged, rows)
    if any(file_hash(paths[key]) != digest for key, digest in before.items()):
        raise ValueError('An input changed during export; original workbook not replaced.')
    backup = args.archive / 'quantified_data.before.xlsx'
    shutil.copy2(args.workbook, backup)
    if file_hash(backup) != before['workbook']:
        raise ValueError('Backup verification failed.')
    digest = file_hash(staged)
    receipt = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_paths': paths,
               'before_sha256': before, 'output_sha256': digest, 'packet_id': packet_id,
               'text_prediction_id': result['id'], 'counts': counts, 'verification': checked,
               'existing_workbook_cells_checked': existing_checked,
               'old_sheets': old_sheets, 'new_sheets': ['实验数据'],
               'columns': [{'key': k, 'label': label, 'group': g} for k, label, g in COLUMNS],
               'removed_original_columns': [k for k in old_columns if k not in {c[0] for c in COLUMNS}],
               'cohorts': dict(Counter(r['source_dataset'] for r in rows)),
               'missing_values_preserved': True, 'source_database_changed': False,
               'prevalent_sleep_diagnosis_not_exported_as_rhythm_truth': True,
               'midpoint_derivation': 'sealed packet for its cohort; stored circular sleep clocks for other original rows',
               'historical_code_and_experiments_changed': False, 'new_model_inference': False,
               'all_source_values_retained_in_database': True}
    save_json(args.archive / 'verification.json', receipt)
    try:
        staged.replace(args.workbook)
    except PermissionError:
        print(json.dumps({'status': 'workbook_locked', 'verified_staged_file': str(staged),
                          'original_unchanged': file_hash(args.workbook) == before['workbook']}, ensure_ascii=False))
        return 2
    if file_hash(args.workbook) != digest or file_hash(args.database) != before['database']:
        raise ValueError('Installed workbook or database hash mismatch.')
    save_json(args.archive / 'installed.json', {'workbook': paths['workbook'], 'sha256': digest,
                                              'database_unchanged': True, **checked})
    print(json.dumps({'status': 'installed', 'workbook': str(args.workbook), **checked}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
