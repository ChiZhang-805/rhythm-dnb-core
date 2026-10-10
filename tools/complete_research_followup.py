"""Prepare, verify and append explicit simulated follow-up in the existing database."""

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import sqlite3
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

from rhythm_dnb.provenance import canonical_json, file_hash, fingerprint
from rhythm_dnb.research.experiment_data import read_snapshot, save_json
from rhythm_dnb.research.simulated_followup import (
    SETTINGS, FEATURES, SCENARIOS, SCENARIO_NAMES, generate_sequence, select_backgrounds,
)
from tools.annotate_research_database import _original_table_hashes


def read_json(path):
    # PSEUDOCODE: load the exact saved experiment artifact.
    return json.loads(Path(path).read_text(encoding='utf-8'))


def annotations_from_database(database):
    # PSEUDOCODE: require one explicit original annotation run; never multiply source rows across annotation versions.
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        runs = db.execute('SELECT run_id FROM research_annotation_runs').fetchall()
        if len(runs) != 1:
            raise ValueError('Select and reconcile the annotation run explicitly before expansion.')
        rows = [dict(r) for r in db.execute('SELECT * FROM research_outcome_annotations WHERE run_id=?', (runs[0][0],))]
    return rows, runs[0][0]


def prepare(database, output):
    # PSEUDOCODE: freeze eligibility, identities, split and generation settings before authoring any follow-up.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source, digest = read_snapshot(database)
    annotations, run_id = annotations_from_database(database)
    backgrounds = select_backgrounds(source, annotations)
    if len(backgrounds) < 30 or file_hash(database) != digest:
        raise ValueError('Insufficient eligible backgrounds or concurrent source changes.')
    root = Path(__file__).resolve().parents[1]
    implementation = {name: file_hash(root / name) for name in (
        'src/rhythm_dnb/research/simulated_followup.py',
        'src/rhythm_dnb/research/dataset_annotations.py', 'tools/complete_research_followup.py')}
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': digest,
            'database': str(Path(database).resolve()), 'source_annotation_run': run_id,
            'settings': SETTINGS, 'backgrounds': backgrounds, 'implementation': implementation,
            'original_records': len(source), 'original_people': len({r['participant_id'] for r in source}),
            'original_data_changed': False, 'new_independent_real_people': 0,
            'evaluation_scope': 'data completeness and authored regime truth, candidate-rule quality audit; no new DNB accuracy result',
            'split_unit': 'original_parent_participant_id',
            'original_unknown_outcomes_remain_unknown': True}
    plan['id'] = fingerprint(plan)
    save_json(output / 'plan.json', plan)
    sequences = []
    states, future_states, scenarios = Counter(), Counter(), Counter()
    with gzip.open(output / 'followup.jsonl.gz', 'xt', encoding='utf-8', newline='\n') as stream:
        for i, background in enumerate(backgrounds):
            rows, evidence = generate_sequence(background)
            sequences.append(evidence)
            for row in rows:
                stream.write(canonical_json(row) + '\n')
                states[row['candidate_status']] += 1
                future_states[row['future_label_status']] += 1
            scenarios[(background['scenario'], evidence['event_in_followup'])] += 1
            if (i + 1) % 25 == 0:
                print(json.dumps({'prepared_sequences': i+1, 'total': len(backgrounds)}), flush=True)
    save_json(output / 'sequences.json', sequences)
    summary = {'plan_id': plan['id'], 'simulated_records': len(backgrounds) * SETTINGS['days'],
               'simulated_trajectories': len(backgrounds), 'parent_identities': len(backgrounds),
               'new_independent_real_people': 0, 'days_per_trajectory': SETTINGS['days'],
               'splits': dict(Counter(b['split'] for b in backgrounds)),
               'event_positive_trajectories': sum(s['event_in_followup'] for s in sequences),
               'event_negative_trajectories': sum(not s['event_in_followup'] for s in sequences),
               'candidate_positive_trajectories': sum(bool(s['candidate_events']) for s in sequences),
               'candidate_positive_without_authored_event': sum(bool(s['candidate_events']) and not s['event_in_followup'] for s in sequences),
               'candidate_rule_used_as_reference_truth': False,
               'candidate_statuses': dict(states), 'future_label_statuses': dict(future_states),
               'scenario_outcomes': [{'scenario': k[0], 'event': k[1], 'trajectories': v}
                                     for k, v in sorted(scenarios.items())],
               'original_records_preserved': len(source), 'new_text_predictions': 0,
               'new_warning_predictions': 0, 'new_clinical_labels': 0}
    save_json(output / 'summary.json', summary)
    files = {name: file_hash(output/name) for name in ('plan.json','sequences.json','followup.jsonl.gz','summary.json')}
    save_json(output / 'manifest.json', {'files': files, 'id': fingerprint(files)})
    validate(output)
    if file_hash(database) != digest:
        raise ValueError('Source changed during preparation; proposal retained, not applied.')
    return summary


def validate(output):
    # PSEUDOCODE: verify hashes and independently regenerate every row under the frozen plan before allowing application.
    output = Path(output)
    manifest, plan = read_json(output / 'manifest.json'), read_json(output / 'plan.json')
    if fingerprint(manifest['files']) != manifest['id']:
        raise ValueError('Bad proposal manifest.')
    for name, digest in manifest['files'].items():
        if Path(name).name != name or file_hash(output / name) != digest:
            raise ValueError('Changed proposal artifact: ' + name)
    if fingerprint({k: v for k, v in plan.items() if k != 'id'}) != plan['id'] or plan['settings'] != SETTINGS:
        raise ValueError('Plan or generation settings changed.')
    root = Path(__file__).resolve().parents[1]
    if any(file_hash(root / name) != digest for name, digest in plan['implementation'].items()):
        raise ValueError('Implementation changed after freezing the plan.')
    backgrounds = plan['backgrounds']
    if (len({b['parent_participant_id'] for b in backgrounds}) != len(backgrounds) or
            len({b['sequence_id'] for b in backgrounds}) != len(backgrounds)):
        raise ValueError('Repeated background identity or synthetic trajectory.')
    sequences = read_json(output / 'sequences.json')
    if len(sequences) != len(backgrounds):
        raise ValueError('Incomplete trajectory metadata.')
    count = 0
    with gzip.open(output / 'followup.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for background, saved_evidence in zip(backgrounds, sequences):
            rows, evidence = generate_sequence(background, plan['settings'])
            if evidence != saved_evidence:
                raise ValueError('Sequence label or lineage mismatch.')
            for expected in rows:
                actual = json.loads(next(stream))
                if actual != expected:
                    raise ValueError('Follow-up row or outcome differs from the frozen generator.')
                count += 1
        if next(stream, None) is not None:
            raise ValueError('Unplanned extra records.')
    if count != len(backgrounds) * plan['settings']['days']:
        raise ValueError('Incomplete follow-up.')
    return plan, sequences, count


def apply(database, output):
    # PSEUDOCODE: lock source -> exact recovery snapshot -> append new tables -> hash all previous tables -> commit.
    database, output = Path(database).resolve(), Path(output)
    plan, sequences, count = validate(output)
    if file_hash(database) != plan['source_sha256']:
        raise ValueError('Database changed; preserve the proposal and prepare against the new snapshot.')
    backup = output / 'rhythm.before.sqlite'
    if backup.exists():
        raise FileExistsError('A recovery snapshot already exists; inspect before retrying.')
    with closing(sqlite3.connect(database)) as db, db:
        if db.execute('PRAGMA journal_mode').fetchone()[0] != 'delete':
            raise ValueError('Require a checkpointed DELETE journal database.')
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('BEGIN IMMEDIATE')
        if file_hash(database) != plan['source_sha256']:
            raise ValueError('Source changed before acquiring the write lock.')
        shutil.copy2(database, backup)
        if file_hash(backup) != plan['source_sha256']:
            raise ValueError('Recovery snapshot hash mismatch.')
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        original = _original_table_hashes(db, tables)
        db.execute('CREATE TABLE research_followup_runs (run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, '
                   'source_sha256 TEXT NOT NULL, plan_json TEXT NOT NULL)')
        db.execute('CREATE TABLE research_followup_sequences (sequence_id TEXT PRIMARY KEY, '
                   'run_id TEXT NOT NULL REFERENCES research_followup_runs(run_id), parent_participant_id TEXT NOT NULL, '
                   'split TEXT NOT NULL, scenario TEXT NOT NULL, event_in_followup INTEGER NOT NULL CHECK(event_in_followup IN (0,1)), '
                   'first_onset_date TEXT, first_confirmed_date TEXT, followup_end_date TEXT NOT NULL, evidence_json TEXT NOT NULL, '
                   'UNIQUE(run_id,parent_participant_id), CHECK((event_in_followup=0 AND first_onset_date IS NULL AND first_confirmed_date IS NULL) '
                   'OR (event_in_followup=1 AND first_onset_date IS NOT NULL AND first_confirmed_date IS NOT NULL)))')
        db.execute('CREATE TABLE research_followup_anchors (sequence_id TEXT NOT NULL REFERENCES research_followup_sequences(sequence_id), '
                   'record_id TEXT NOT NULL REFERENCES observations(record_id), PRIMARY KEY(sequence_id,record_id))')
        db.execute('CREATE TABLE research_followup_days (record_id TEXT PRIMARY KEY, '
                   'sequence_id TEXT NOT NULL REFERENCES research_followup_sequences(sequence_id), day_index INTEGER NOT NULL, '
                   'simulated_at TEXT NOT NULL, features_json TEXT NOT NULL, candidate_state INTEGER CHECK(candidate_state IN (0,1)), '
                   'candidate_status TEXT NOT NULL, candidate_onset_date TEXT, candidate_confirmed_date TEXT, '
                   'future_event_7d INTEGER CHECK(future_event_7d IN (0,1)), future_label_status TEXT NOT NULL, '
                   'regime_state INTEGER NOT NULL CHECK(regime_state IN (0,1)), '
                   'UNIQUE(sequence_id,day_index), UNIQUE(sequence_id,simulated_at))')
        db.execute('INSERT INTO research_followup_runs VALUES (?,?,?,?)',
                   (plan['id'], plan['created_at'], plan['source_sha256'], canonical_json(plan)))
        for s in sequences:
            values = [s[k] for k in ('sequence_id','parent_participant_id','split','scenario','event_in_followup',
                                    'first_onset_date','first_confirmed_date','followup_end_date')]
            db.execute('INSERT INTO research_followup_sequences VALUES (?,?,?,?,?,?,?,?,?,?)',
                       [values[0], plan['id'], *values[1:], canonical_json(s)])
            for rid in s['anchor_record_ids']:
                actual = db.execute('SELECT participant_id FROM observations WHERE record_id=?', (rid,)).fetchone()
                if actual is None or actual[0] != s['parent_participant_id']:
                    raise ValueError('Anchor belongs to a different source identity.')
                db.execute('INSERT INTO research_followup_anchors VALUES (?,?)', (s['sequence_id'], rid))
        with gzip.open(output / 'followup.jsonl.gz', 'rt', encoding='utf-8') as stream:
            for line in stream:
                r = json.loads(line)
                db.execute('INSERT INTO research_followup_days VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                           [r['record_id'], r['participant_id'], r['day_index'], r['simulated_at'], canonical_json(r['features']),
                            *[r[k] for k in ('candidate_state','candidate_status','candidate_onset_date',
                                            'candidate_confirmed_date','future_event_7d','future_label_status','regime_state')]])
        db.execute('CREATE VIEW research_all_record_status AS '
            'SELECT o.record_id, o.participant_id, o.participant_id AS parent_participant_id, o.source_dataset, '
            'o.observed_at AS original_timestamp, NULL AS simulated_timestamp, a.is_simulated_cohort, '
            'a.reference_event_in_followup AS event_in_followup, a.reference_onset_at AS first_onset, '
            'a.candidate_state, a.status AS annotation_status, a.reference_scope AS label_source, '
            'NULL AS future_event_7d FROM observations o JOIN research_outcome_annotations a USING(record_id) '
            'UNION ALL SELECT d.record_id, d.sequence_id, s.parent_participant_id, \'SIMULATED-FOLLOWUP\', '
            'NULL, d.simulated_at, 1, s.event_in_followup, s.first_onset_date, d.candidate_state, '
            'd.candidate_status, \'authored_sustained_multi_domain_regime\', d.future_event_7d '
            'FROM research_followup_days d JOIN research_followup_sequences s USING(sequence_id)')
        if original != _original_table_hashes(db, tables):
            raise ValueError('An existing table changed; rolling back all new records.')
        if db.execute('PRAGMA foreign_key_check').fetchone() or db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Database integrity check failed.')
        if db.execute('SELECT count(*) FROM research_all_record_status').fetchone()[0] != plan['original_records'] + count:
            raise ValueError('Unified view is incomplete or duplicates original rows.')
    receipt = {'plan_id': plan['id'], 'database': str(database), 'backup': str(backup),
               'before_sha256': plan['source_sha256'], 'after_sha256': file_hash(database),
               'original_tables': original, 'all_original_tables_unchanged': True, 'added_records': count,
               'unified_records': plan['original_records'] + count, 'new_independent_real_people': 0}
    save_json(output / 'database-verification.json', receipt)
    return receipt


def sheet_xml(rows):
    # PSEUDOCODE: build plain spreadsheet cells without altering existing sheets or importing styles.
    from openpyxl.utils import get_column_letter
    result = []; width = None
    for i, row in enumerate(rows, 1):
        width = len(row) if width is None else width
        if len(row) != width:
            raise ValueError('Worksheet row width differs.')
        cells = []
        for j, value in enumerate(row, 1):
            if value is None:
                continue
            ref = f'{get_column_letter(j)}{i}'
            if isinstance(value, str):
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>')
            else:
                cells.append(f'<c r="{ref}" t="n"><v>{value}</v></c>')
        result.append(f'<row r="{i}">'+''.join(cells)+'</row>')
    end = f'{get_column_letter(width)}{len(result)}'
    return ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="A1:{end}"/><sheetViews><sheetView workbookViewId="0">'
            '<pane xSplit="1" ySplit="1" topLeftCell="B2" state="frozen"/></sheetView></sheetViews>'
            f'<cols><col min="1" max="{width}" width="26" customWidth="1"/></cols>'
            '<sheetData>'+''.join(result)+f'</sheetData><autoFilter ref="A1:{end}"/></worksheet>').encode('utf-8')


def workbook(database, workbook_path, output):
    # PSEUDOCODE: add simulation and completion sheets in the same workbook, preserving old worksheet XML byte-for-byte.
    database, workbook_path, output = Path(database), Path(workbook_path), Path(output)
    receipt = read_json(output / 'database-verification.json')
    if file_hash(database) != receipt['after_sha256']:
        raise ValueError('Database changed after application.')
    plan, sequences, count = validate(output)
    before = file_hash(workbook_path)
    backup = output / 'quantified_data.before.xlsx'
    if backup.exists():
        raise FileExistsError('Workbook recovery snapshot exists; inspect before retrying.')
    shutil.copy2(workbook_path, backup)
    with ZipFile(workbook_path) as z:
        entries = {n: z.read(n) for n in z.namelist()}
    main = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    package = 'http://schemas.openxmlformats.org/package/2006/relationships'
    types = 'http://schemas.openxmlformats.org/package/2006/content-types'
    wb = ET.fromstring(entries['xl/workbook.xml'])
    sheets = wb.find('{'+main+'}sheets')
    relations = ET.fromstring(entries['xl/_rels/workbook.xml.rels'])
    content = ET.fromstring(entries['[Content_Types].xml'])
    overview = [['来源人物', '模拟序列', '场景', '用途划分', '随访天数', '模拟期间是否发生（场景答案）',
                 '模拟起始日期', '模拟确认日期', '模拟随访结束日', '参考原记录数量', '旧规则是否标出候选事件']]
    for s in sequences:
        overview.append([s['parent_participant_id'], s['sequence_id'], SCENARIO_NAMES[s['scenario']], s['split'],
                         s['simulated_days'], s['event_in_followup'], s['first_onset_date'], s['first_confirmed_date'],
                         s['followup_end_date'], len(s['anchor_record_ids']), int(bool(s['candidate_events']))])
    daily = [['记录编号', '来源人物', '模拟序列', '模拟第几天（0起）', '模拟时间',
              '睡眠中点（小时）', '睡眠时长（小时）', '餐间隔不规则程度', '运动时刻（小时）',
              '运动分钟', '静息心率', '屏幕分钟', '当日候选状态', '当日状态说明',
              '未来7天是否开始（场景答案）', '未来标签说明', '当日场景状态']]
    with gzip.open(output / 'followup.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            daily.append([r['record_id'], r['parent_participant_id'], r['participant_id'], r['day_index'],
                          r['simulated_at'], *[r['features'][k] for k in FEATURES], r['candidate_state'],
                          r['candidate_status'], r['future_event_7d'], r['future_label_status'], r['regime_state']])
    combined = [['人物编号', '原记录数', '原数据结局情况', '原参考事件（0/1）', '规则候选阳性记录数',
                 '原数据可判定记录数', '新增模拟随访天数', '新增模拟事件（0/1）', '模拟起始日', '说明']]
    by_parent = {s['parent_participant_id']: s for s in sequences}
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        people = db.execute('SELECT o.participant_id,count(*),max(a.reference_event_in_followup),'
            'sum(a.candidate_state=1),sum(a.candidate_state IS NOT NULL),min(o.split), '
            'max(o.source_dataset=\'SIMULATED-RHYTHM\') '
            'FROM observations o JOIN research_outcome_annotations a USING(record_id) GROUP BY o.participant_id').fetchall()
    for person, n, truth, positives, evaluable, role, simulated in people:
        s = by_parent.get(person)
        status = ('已有模拟随访答案' if truth is not None else '稳定参考，不是随访阴性' if simulated and role == 'reference'
                  else '已有部分窗口候选标注' if evaluable else '原始随访结局未知，已补模拟分支')
        combined.append([person, n, status, truth, positives or 0, evaluable,
                         s['simulated_days'] if s else 0, s['event_in_followup'] if s else None,
                         s['first_onset_date'] if s else None,
                         '模拟日期为相对日历；不改变原始结局，不增加真实人数' if s else '沿用原数据与标注'])
    additions = [('补齐概览', combined), ('模拟随访结局', overview), ('模拟随访记录', daily)]
    old_xml = {n: payload for n, payload in entries.items() if n.startswith('xl/worksheets/')}
    for title, rows in additions:
        if any(s.attrib['name'] == title for s in sheets):
            raise ValueError('Supplementary worksheet already exists.')
        number = max(int(s.attrib['sheetId']) for s in sheets) + 1
        relation_id = 'rIdFollowup' + str(number)
        target = f'worksheets/sheet{number}.xml'
        if 'xl/' + target in entries or any(r.attrib['Id'] == relation_id for r in relations):
            raise ValueError('Workbook relationship collision.')
        ET.SubElement(sheets, '{'+main+'}sheet', {'name': title, 'sheetId': str(number), '{'+rel+'}id': relation_id})
        ET.SubElement(relations, '{'+package+'}Relationship', {'Id': relation_id, 'Type': rel+'/worksheet', 'Target': target})
        ET.SubElement(content, '{'+types+'}Override', {'PartName': '/xl/'+target,
                       'ContentType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml'})
        entries['xl/'+target] = sheet_xml(rows)
    entries['xl/workbook.xml'] = ET.tostring(wb, encoding='utf-8')
    entries['xl/_rels/workbook.xml.rels'] = ET.tostring(relations, encoding='utf-8')
    entries['[Content_Types].xml'] = ET.tostring(content, encoding='utf-8')
    pending = output / 'quantified_data.pending.xlsx'
    with ZipFile(pending, 'x', ZIP_DEFLATED) as z:
        for name, payload in entries.items():
            z.writestr(name, payload)
    from openpyxl import load_workbook
    check = load_workbook(pending, read_only=True)
    dimensions = {name: [check[name].max_row, check[name].max_column] for name, _ in additions}
    check.close()
    with ZipFile(pending) as z:
        if any(z.read(n) != value for n, value in old_xml.items()):
            raise ValueError('An existing worksheet was modified.')
    if dimensions['模拟随访记录'][0] != count+1 or file_hash(workbook_path) != before or file_hash(database) != receipt['after_sha256']:
        raise ValueError('Wrong export size or concurrent source/workbook changes.')
    pending.replace(workbook_path)
    report = {'workbook': str(workbook_path), 'before_sha256': before, 'after_sha256': file_hash(workbook_path),
              'original_worksheet_xml_unchanged': True, 'new_sheets': dimensions, 'plan_id': plan['id']}
    save_json(output / 'workbook-verification.json', report)
    return report


def main():
    # PSEUDOCODE: make data preparation, reviewed application and spreadsheet export individually auditable.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare','validate','apply','workbook'))
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workbook', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        result = prepare(args.database, args.output)
    elif args.stage == 'validate':
        plan, _, count = validate(args.output)
        result = {'plan_id': plan['id'], 'verified_records': count}
    elif args.stage == 'apply':
        result = apply(args.database, args.output)
    else:
        if args.workbook is None:
            parser.error('workbook stage requires --workbook')
        result = workbook(args.database.resolve(), args.workbook, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
