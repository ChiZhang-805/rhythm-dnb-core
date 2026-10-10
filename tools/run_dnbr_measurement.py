"""Isolate text measurement effects on warning performance without changing people or outcomes."""

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np

from rhythm_dnb.measures.scaling import fit_scaler
from rhythm_dnb.provenance import file_hash, fingerprint
from rhythm_dnb.research.experiment import policy_settings
from rhythm_dnb.research.experiment_data import load_packet, save_json, OBJECTIVE, TEXT
from rhythm_dnb.research.experiment_gpu import attach_predictions, implementation_id
from rhythm_dnb.research.evaluate import event_metrics
from tools.run_dnbr_ablation import paired_accuracy_interval
from tools.run_dnbr_cross_validation import ROTATIONS, run_fold, validate_people, verify_parent
from tools.run_dnbr_warning import read_json


ROLES = ('reference', 'train', 'validation', 'test')
CONDITIONS = ('model_text', 'authored_reference_text', 'objective_only')
LABELS = ('模型文本分数', '数据集编写时的参考分数', '仅客观指标')


def verify_source_row(packet_row, source):
    # PSEUDOCODE: require identical identities, texts, times and outcomes before accepting existing reference scores.
    if (source['source_dataset'] != 'SIMULATED-RHYTHM' or
            source['label_source'] != 'synthetic_scenario_not_clinical_ground_truth'):
        raise ValueError('Reference scores do not belong to the authored cohort.')
    raw = json.loads(source['raw'])
    for key in ('record_id', 'participant_id', 'split', 'observed_at'):
        if source[key] != packet_row[key]:
            raise ValueError('Source identity/time differs: ' + key)
    if any(source[k] != v for k, v in packet_row['outcome'].items()):
        raise ValueError('Source outcomes differ from frozen packet.')
    if any(raw[category + '_description_text'].strip() != text for category, text in packet_row['texts'].items()):
        raise ValueError('Source text differs from frozen packet.')
    provenance = json.loads(source['provenance'])['provenance']
    if any(provenance.get(k, {}).get('method') != 'manual_contextual_rewrite' for k in TEXT):
        raise ValueError('Reference score provenance changed.')
    scores = {k: source[k] for k in TEXT}
    if any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 100 for v in scores.values()):
        raise ValueError('Missing or invalid authored reference score.')
    return scores


def export_references(args):
    # PSEUDOCODE: read the unchanged database, verify every record, export only traceable existing references.
    data, manifest = load_packet(args.packet)
    database, output = Path(args.database).resolve(), Path(args.output)
    if output.exists():
        raise FileExistsError('Do not replace an existing reference export.')
    if file_hash(database) != manifest['source_sha256']:
        raise ValueError('Database differs from frozen source; reconcile before proceeding.')
    references = {}
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        for role in ROLES:
            for row in data[role]:
                source = db.execute('SELECT o.*, r.payload AS raw, p.payload AS provenance FROM observations o '
                                    'JOIN raw_inputs r USING(record_id) JOIN observation_provenance p USING(record_id) '
                                    'WHERE o.record_id=?', (row['record_id'],)).fetchone()
                if source is None:
                    raise ValueError('Reference record missing: ' + row['record_id'])
                references[row['record_id']] = verify_source_row(row, dict(source))
    if file_hash(database) != manifest['source_sha256']:
        raise ValueError('Source changed during export.')
    payload = {'packet_id': manifest['id'], 'source_sha256': manifest['source_sha256'],
               'domain': manifest['domain'], 'independent_human_gold': False,
               'purpose': 'authored_reference_counterfactual_not_deployable_predictions', 'scores': references}
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, {**payload, 'id': fingerprint(payload)})
    print('Verified reference records: ' + str(len(references)), flush=True)


def attach_references(data, manifest, path):
    # PSEUDOCODE: verify counterfactual references, preserve metadata, and change only the four text measurements.
    payload = read_json(path)
    if (payload['id'] != fingerprint({k: v for k, v in payload.items() if k != 'id'}) or
            payload['packet_id'] != manifest['id'] or payload['source_sha256'] != manifest['source_sha256'] or
            payload['domain'] != manifest['domain'] or payload['independent_human_gold'] is not False or
            payload['purpose'] != 'authored_reference_counterfactual_not_deployable_predictions'):
        raise ValueError('Authored reference export identity mismatch.')
    expected = {r['record_id'] for role in ROLES for r in data[role]}
    if set(payload['scores']) != expected:
        raise ValueError('Missing or extra authored reference records.')
    result = deepcopy(data)
    for role in ROLES:
        for row in result[role]:
            scores = payload['scores'][row['record_id']]
            if set(scores) != set(TEXT) or any(type(v) not in (int, float) or not np.isfinite(v) or
                                              not 0 <= v <= 100 for v in scores.values()):
                raise ValueError('Invalid authored reference measurements.')
            row['features'].update(scores)
    return result, payload['id']


def measurement_errors(model, reference):
    # PSEUDOCODE: describe score errors and the reference distribution without selecting or dropping any records.
    report = {}
    for role in ROLES:
        report[role] = {}
        for name in TEXT:
            a = np.array([r['features'][name] for r in model[role]])
            b = np.array([r['features'][name] for r in reference[role]])
            report[role][name] = {'n': len(a), 'mae': float(np.abs(a-b).mean()),
                                  'bias': float((a-b).mean()), 'prediction_sd': float(a.std(ddof=1)),
                                  'reference_sd': float(b.std(ddof=1))}
    return report


def write_report(result, output):
    # PSEUDOCODE: distinguish counterfactual scores from deployable predictions and show all frozen comparisons.
    lines = ['# 文本量化误差对预警的影响', '',
             '同一批 180 名模拟人物、900 次判断；没有修改结局、人物划分或报警政策。',
             '参考分数是数据编写时的赋分，不是独立人工真值，也不是可部署的模型输出。', '',
             '| 输入 | 算法 | 正确率 | 漏报事件 / 90 | 误报提醒 | 提前天数中位数 | 覆盖率 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for condition, label in zip(CONDITIONS, LABELS):
        for method, title in (('dnbr_sdnb', 'DNB'), ('reference_deviation', '平均偏离程度')):
            m = result['conditions'][condition]['methods'][method]['test']
            accuracy = f'{m["risk_accuracy"]:.2%}' if m['risk_accuracy'] is not None else '不可计算'
            lines.append(f'| {label} | {title} | {accuracy} | {m["events"] - m["detected_events"]} | '
                         f'{m["false_alarms"]} | {m["median_lead_days"]} | {m["classification_coverage"]:.1%} |')
    lines += ['', '每天是否超阈值和经过连续两天确认、七天冷却后的提醒分开统计。',
              '每种输入重新在开发人群找组、校准人群定阈值；测试人群不参与选择。',
              '原模型输入必须精确重现上一轮结果，否则实验停止。完整模块及逐条预测保存在各输入目录。',
              '参考分数实验只排查测量环节，不是可实现的性能上限；仅客观面板同时改变指标数量。',
              '历史模拟数据已经查看；不能以本轮最高成绩替换主模型，不能证明真实人群或临床正确率。']
    (output/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def evaluate(args):
    # PSEUDOCODE: freeze input ablations -> reproduce current route -> test all alternatives -> paired audit and archive.
    output, parent = Path(args.output).resolve(), Path(args.parent).resolve()
    if output.exists():
        raise FileExistsError('Preserve completed experiments; use a fresh output directory.')
    verify_parent(parent)
    previous, old_plan = read_json(parent/'result.json'), read_json(parent/'plan.json')
    data, manifest = load_packet(args.packet)
    people = validate_people(data)
    inference = attach_predictions(data, manifest, args.predictions)
    if manifest['id'] != old_plan['packet_id'] or inference != old_plan['inference_id']:
        raise ValueError('Inputs differ from preserved experiment.')
    reference, reference_id = attach_references(data, manifest, args.references)
    settings, discovery, protocol = old_plan['settings'], old_plan['discovery'], data['protocol']
    output.mkdir(parents=True)
    source = output/'source'; source.mkdir()
    for name in ('run_dnbr_measurement.py', 'run_dnbr_cross_validation.py', 'run_dnbr_warning.py',
                 'run_dnbr_ablation.py', 'export_dnbr.py', 'run_dnbr.R', 'score_dnbr.R'):
        shutil.copy2(Path(__file__).with_name(name), source/name)
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'packet_id': manifest['id'],
            'inference_id': inference, 'reference_id': reference_id, 'domain': manifest['domain'],
            'implementation_id': implementation_id(), 'conditions': CONDITIONS, 'rotations': ROTATIONS,
            'people': people, 'protocol': protocol, 'settings': settings, 'discovery': discovery,
            'primary_comparison': 'authored_reference_text_minus_model_text_DNB',
            'secondary_comparison': 'objective_only_minus_model_text_DNB',
            'historical_test_previously_inspected': True, 'method_selection_on_test': False,
            'reference_input_not_deployable': True,
            'source': {p.name: file_hash(p) for p in source.iterdir()}}
    save_json(output/'plan.json', {**plan, 'id': fingerprint(plan)})
    save_json(output/'measurement-errors.json', measurement_errors(data, reference))
    results, all_pooled = {}, {}
    for condition in CONDITIONS:
        condition_data = reference if condition == 'authored_reference_text' else data
        features = list(OBJECTIVE) + ([] if condition == 'objective_only' else list(TEXT))
        folder = output/condition; folder.mkdir()
        scaler = fit_scaler([[r['features'][k] for k in features] for r in condition_data['reference']], features)
        save_json(folder/'scaler.json', scaler)
        folds, pooled = [], defaultdict(lambda: [[], [], []])
        for number, roles in enumerate(ROTATIONS, 1):
            fold, parts = run_fold(condition_data, manifest, inference, roles, settings, discovery, scaler,
                                   args, folder/f'fold-{number}', features=features)
            if condition == 'model_text':
                for method, item in fold['methods'].items():
                    prior = previous['folds'][number-1]['methods'][method]
                    if item['threshold'] != prior['threshold'] or item['test'] != prior['test']:
                        raise ValueError('Model-input route failed exact reproduction: ' + method)
            folds.append(fold)
            for method, triple in parts.items():
                for destination, values in zip(pooled[method], triple):
                    destination.extend(values)
            print('Completed ' + condition + ' rotation ' + str(number), flush=True)
        methods = {}
        for method, (days, events, periods) in pooled.items():
            ids = [(d.participant_id, d.issued_at) for d in days]
            expected_people = set().union(*(set(people[k]) for k in ROLES if k != 'reference'))
            if (len(ids) != len(set(ids)) or {p.participant_id for p in periods} != expected_people or
                    len(periods) != len(expected_people)):
                raise ValueError('Pooled forecasts repeat or omit participants.')
            metrics = event_metrics(days, .5, events=events, monitoring=periods, **policy_settings(protocol))
            metrics.update(roc_auc=None, average_precision=None)
            methods[method] = {'test': metrics}
        results[condition] = {'features': features, 'folds': folds, 'methods': methods}
        all_pooled[condition] = pooled
    intervals = {}
    for condition in CONDITIONS:
        for method in ('dnbr_sdnb', 'reference_deviation'):
            intervals[condition + ':' + method] = paired_accuracy_interval(all_pooled[condition][method][0],
                all_pooled['model_text'][method][0], protocol['seed'], protocol['interval_repetitions'])
    result = {'plan_id': fingerprint(plan), 'conditions': results, 'paired_difference_from_model_text': intervals,
              'clinical_accuracy_established': False, 'reference_scores_are_human_gold': False}
    save_json(output/'result.json', result)
    write_report(result, output)
    files = {p.relative_to(output).as_posix(): file_hash(p) for p in output.rglob('*') if p.is_file()}
    save_json(output/'archive-manifest.json', {'files': files, 'id': fingerprint(files)})
    print('Complete: ' + str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    for name in ('database', 'packet', 'output'):
        export.add_argument('--'+name, required=True)
    compute = sub.add_parser('evaluate')
    for name in ('packet', 'predictions', 'references', 'parent', 'output', 'library'):
        compute.add_argument('--'+name, required=True)
    compute.add_argument('--rscript', default='Rscript')
    args = parser.parse_args()
    (export_references if args.command == 'export' else evaluate)(args)
