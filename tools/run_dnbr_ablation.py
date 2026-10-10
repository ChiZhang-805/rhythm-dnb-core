"""Separate amplitude from network evidence on the preserved, previously inspected rhythm cohort."""

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np

from rhythm_dnb.provenance import file_hash, fingerprint
from rhythm_dnb.research.experiment import choose_threshold, forecast_rows, policy_settings
from rhythm_dnb.research.experiment_data import load_packet, save_json
from rhythm_dnb.research.experiment_gpu import attach_predictions, implementation_id
from rhythm_dnb.research.evaluate import event_metrics
from tools.run_dnbr_cross_validation import (ROTATIONS, execute_r, policy_decisions, validate_people,
                                             verify_parent)
from tools.run_dnbr_warning import read_json, audit_components


METHODS = ('dnbr_sdnb', 'module_amplitude', 'network_ratio', 'normalized_dnb', 'reference_deviation')
LABELS = ('原 DNB', '仅指标组偏离', '仅网络关联比值', '关联标准化 DNB', '平均偏离程度法')


def matrix(path):
    # PSEUDOCODE: read the sealed feature-by-record matrix, preserving identities and orientation.
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.reader(stream))
    features, records = [r[0] for r in rows[1:]], rows[0][1:]
    values = np.asarray([r[1:] for r in rows[1:]], dtype=float).T
    if len(set(features)) != len(features) or len(set(records)) != len(records) or not np.isfinite(values).all():
        raise ValueError('Ambiguous or nonfinite scoring matrix.')
    return features, records, values


def audit_scores(inputs, scored, rows):
    # PSEUDOCODE: independently audit every R component, then derive the prespecified ablations.
    config = read_json(inputs / 'manifest.json')
    features, _, reference = matrix(inputs / 'reference.csv')
    target_features, records, targets = matrix(inputs / 'targets.csv')
    if features != target_features or records != [r['record_id'] for r in rows]:
        raise ValueError('Score/input alignment error.')
    modules = [{'module': m} for m in config['modules']]
    audit = audit_components(scored / 'components.csv', rows, reference, targets, features, modules, config['epsilon'])
    base = np.corrcoef(reference, rowvar=False)
    denominator = 1 - base**2
    np.fill_diagonal(denominator, 1)
    if np.any(denominator <= config['epsilon']):
        raise ValueError('Near-perfect reference correlation.')
    with (scored / 'ablation-components.csv').open(encoding='utf-8', newline='') as stream:
        parts = list(csv.DictReader(stream))
    lookup = {(r['record_id'], int(r['module_id'])): r for r in parts}
    if len(lookup) != len(parts) or len(parts) != len(rows) * len(modules):
        raise ValueError('Missing or repeated ablation component.')
    scores = {name: {} for name in METHODS}
    maximum = 0.
    for record, target in zip(records, targets):
        delta = np.corrcoef(np.vstack([reference, target]), rowvar=False) - base
        z = delta * (len(reference) - 1) / denominator
        np.fill_diagonal(z, 0)
        collected = defaultdict(list)
        for number, module in enumerate(config['modules'], 1):
            part = lookup[(record, number)]
            inside = [features.index(k) for k in module]
            outside = [i for i in range(len(features)) if i not in inside]
            zi = float(np.abs(z[np.ix_(inside, inside)]).mean())
            zo = float(np.abs(z[np.ix_(inside, outside)]).mean())
            expected = {'z_in': zi, 'z_out': zo, 'z_score':
                        float(part['sED_in']) * zi / zo if zo > config['epsilon'] else None}
            for key, value in expected.items():
                observed = float(part[key]) if part[key] else None
                if value is None:
                    if observed is not None:
                        raise ValueError('Undefined Z ratio must abstain.')
                else:
                    np.testing.assert_allclose(observed, value, atol=1e-10, rtol=1e-9)
                    maximum = max(maximum, abs(observed - value))
            valid = part['valid'].upper() == 'TRUE'
            collected['dnbr_sdnb'].append(float(part['score']) if valid else None)
            collected['module_amplitude'].append(float(part['sED_in']))
            collected['network_ratio'].append(float(part['sPCC_in']) / float(part['sPCC_out']) if valid else None)
            collected['normalized_dnb'].append(float(part['z_score']) if part['z_score'] else None)
        for name in METHODS[:-1]:
            values = collected[name]
            scores[name][record] = max(values) if values and all(v is not None for v in values) else None
        scores['reference_deviation'][record] = float(np.abs(target).mean())
    return scores, {**audit, 'z_maximum_absolute_error': maximum, 'z_components_checked': len(parts)}


def paired_accuracy_interval(left, right, seed, repetitions):
    # PSEUDOCODE: resample the same people for both methods; estimate balanced-accuracy differences.
    keys = lambda rows: [(d.participant_id, d.issued_at) for d in rows]
    if keys(left) != keys(right) or len(set(keys(left))) != len(left):
        raise ValueError('Paired predictions must have unique matching people and dates.')
    if any(d.score is None or d.label is None for d in (*left, *right)):
        return {'percentile_95_difference': None, 'reason': 'Unequal/partial coverage; no selective paired estimate.'}
    counts = defaultdict(lambda: np.zeros((2, 4)))
    for a, b in zip(left, right):
        if a.label != b.label or a.label not in (0, 1) or a.score not in (0., 1.) or b.score not in (0., 1.):
            raise ValueError('Paired accuracy requires full coverage and matching known labels.')
        for i, d in enumerate((a, b)):
            counts[d.participant_id][i, d.label * 2 + int(d.score)] += 1
    values = np.stack(list(counts.values()))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        sample = values[rng.integers(len(values), size=len(values))].sum(axis=0)
        if np.any(sample[:, :2].sum(axis=1) == 0) or np.any(sample[:, 2:].sum(axis=1) == 0):
            continue
        accuracy = .5 * (sample[:, 0] / sample[:, :2].sum(axis=1) + sample[:, 3] / sample[:, 2:].sum(axis=1))
        differences.append(float(accuracy[0] - accuracy[1]))
    return {'percentile_95_difference': np.quantile(differences, [.025, .975]).tolist() if differences else None,
            'valid_replicates': len(differences), 'requested_replicates': repetitions,
            'scope': 'Paired people, conditional on already fitted folds; not refitting or a fresh test.'}


def summary(result, output):
    # PSEUDOCODE: report every prespecified method, keeping ablations distinct from DNB methods.
    lines = ['# DNB 网络贡献检验', '', '同一批 180 名模拟人物、900 次判断；校准人群选阈值，测试人群核验。',
             '既有测试已看过，本轮是探索性诊断；不替换原实验，不声称真实人群有效。', '',
             '| 方法 | 答对 / 总数 | 正确率 | 漏报事件 | 误报提醒 | 提前天数中位数 |',
             '|---|---:|---:|---:|---:|---:|']
    for key, label in zip(METHODS, LABELS):
        m = result['methods'][key]['test']; c = m['risk_confusion']
        accuracy = f'{m["risk_accuracy"]:.2%}' if m['risk_accuracy'] is not None else '不可计算'
        lines.append(f'| {label} | {c["tp"] + c["tn"]}/{m["classified_days"]} | {accuracy} | '
                     f'{m["events"] - m["detected_events"]}/{m["events"]} | {m["false_alarms"]} | {m["median_lead_days"]} |')
    lines += ['', '“仅指标组偏离”和“仅网络关联比值”是拆分对照，不是完整 DNB。',
              '关联标准化使用 Liu 等 2017 年式 (5) 的 Z，保留式 (6) 结构、原指标组和原预警政策。',
              '唯一新增完整候选在计算前指定为关联标准化 DNB；没有从测试成绩里挑选赢家。',
              '配对置信区间和逐日结果见 result.json；作者 landscape 图的适配审计见各折 eligibility 文件。']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(args):
    # PSEUDOCODE: verify parent -> freeze diagnostic plan -> recalibrate variants -> audit all held-out predictions.
    parent, output = Path(args.parent).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output; preserve previous experiments.')
    verify_parent(parent)
    previous = read_json(parent / 'result.json'); old_plan = read_json(parent / 'plan.json')
    data, manifest = load_packet(args.packet)
    validate_people(data)
    inference = attach_predictions(data, manifest, args.predictions)
    if manifest['id'] != old_plan['packet_id'] or inference != old_plan['inference_id']:
        raise ValueError('Ablation inputs differ from the preserved experiment.')
    protocol = data['protocol']
    output.mkdir(parents=True); source = output / 'source'; source.mkdir()
    for name in ('run_dnbr_ablation.py', 'score_dnbr_ablation.R', 'score_dnbr.R',
                 'run_dnbr_cross_validation.py', 'run_dnbr_warning.py', 'export_dnbr.py'):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'domain': manifest['domain'],
            'packet_id': manifest['id'], 'inference_id': inference, 'implementation_id': implementation_id(),
            'parent_manifest_sha256': file_hash(parent / 'archive-manifest.json'), 'rotations': ROTATIONS,
            'protocol': protocol, 'methods': METHODS, 'primary_candidate': 'normalized_dnb',
            'method_selection_on_test': False, 'historical_data_previously_inspected': True,
            'formula': 'Z=delta_PCC*(n-1)/(1-reference_PCC^2); amplitude*mean_abs_Z_in/mean_abs_Z_out',
            'aggregation': 'max over original frozen modules; all components required; K-squared internal pairs',
            'epsilon': protocol['epsilon'], 'missing_or_degenerate': 'abstain, never zero risk',
            'source': {p.name: file_hash(p) for p in source.iterdir()},
            'paper': 'https://doi.org/10.1371/journal.pcbi.1005633',
            'landscape_audit': 'Author reference gate 0.01/p^2 and >=3 neighbors; no lowered gate.'}
    save_json(output / 'plan.json', {**plan, 'id': fingerprint(plan)})
    pooled = defaultdict(lambda: [[], [], []]); folds = []
    for number, roles in enumerate(ROTATIONS, 1):
        old_fold, fold = parent / f'fold-{number}', output / f'fold-{number}'
        fold.mkdir(); frozen = {}; methods = {}
        for role, role_name in zip(('validation', 'test'), roles[1:]):
            scored = fold / (role + '-scores')
            inputs = old_fold / (role + '-input')
            execute_r([args.rscript, str(Path(__file__).with_name('score_dnbr_ablation.R')), str(inputs),
                       str(scored), args.library], fold / (role + '-R.log'))
            scores, audit = audit_scores(inputs, scored, data[role_name])
            save_json(fold / (role + '-audit.json'), audit)
            for name, values in scores.items():
                days, events, periods, reasons = forecast_rows(data[role_name], values, protocol)
                if role == 'validation':
                    frozen[name] = choose_threshold(days, events, periods, protocol)
                else:
                    threshold = frozen[name]['threshold']
                    metrics = event_metrics(days, threshold, events=events, monitoring=periods, **policy_settings(protocol))
                    methods[name] = {'test': metrics, 'threshold': threshold, 'label_reasons': reasons}
                    save_json(fold / (name + '-forecasts.json'), [asdict(d) for d in days])
                    for target, part in zip(pooled[name], (policy_decisions(days, threshold), events, periods)):
                        target.extend(part)
            if role == 'validation':
                save_json(fold / 'frozen-policy.json', {'created_at': datetime.now(timezone.utc).isoformat(),
                          'methods': frozen, 'test_used': False})
        for name in ('dnbr_sdnb', 'reference_deviation'):
            old = previous['folds'][number - 1]['methods'][name]
            if methods[name]['test'] != old['test'] or methods[name]['threshold'] != old['threshold']:
                raise ValueError('Original DNB/baseline did not reproduce.')
        folds.append({'roles': roles, 'methods': methods})
        save_json(fold / 'result.json', folds[-1])
        print(f'Completed fold {number}', flush=True)
    methods = {}
    for name, (days, events, periods) in pooled.items():
        if len({p.participant_id for p in periods}) != len(periods):
            raise ValueError('A participant was tested more than once.')
        metrics = event_metrics(days, .5, events=events, monitoring=periods, **policy_settings(protocol))
        metrics.update(roc_auc=None, average_precision=None)
        by_day = defaultdict(list)
        for d in days:
            by_day[d.issued_at.isoformat()].append(d)
        methods[name] = {'test': metrics, 'by_date': {day: {
            'correct': sum(d.score is not None and d.label == d.score for d in sequence),
            'classified': sum(d.score is not None and d.label is not None for d in sequence)}
            for day, sequence in by_day.items()}}
        save_json(output / (name + '-pooled-decisions.json'), [asdict(d) for d in days])
    comparisons = {}
    for name in METHODS[:-1]:
        comparisons[name] = paired_accuracy_interval(pooled[name][0], pooled['reference_deviation'][0],
                                                    protocol['seed'], protocol['interval_repetitions'])
    result = {'plan_id': fingerprint(plan), 'domain': manifest['domain'], 'methods': methods, 'folds': folds,
              'paired_difference_from_reference': comparisons, 'clinical_accuracy_established': False}
    save_json(output / 'result.json', result); summary(result, output)
    files = {p.relative_to(output).as_posix(): file_hash(p) for p in output.rglob('*') if p.is_file()}
    save_json(output / 'archive-manifest.json', {'files': files, 'id': fingerprint(files)})
    print('Complete: ' + str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('parent', 'packet', 'predictions', 'output', 'library'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--rscript', default='Rscript')
    run(parser.parse_args())
