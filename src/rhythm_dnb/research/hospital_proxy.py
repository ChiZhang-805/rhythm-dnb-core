"""Compare held-out exploratory sDNB scores to later recorded timing changes, without clinical labels."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ..provenance import file_hash
from .hospital_indicators import _csv, _json
from .reference_stability import network_changes, personal_components


TARGET_FIELDS = ('sleep_start_time', 'sleep_end_time', 'breakfast_time', 'lunch_time', 'dinner_time')
METHODS = ('sdnb_exploratory', 'mean_deviation')


def check_plan(plan):
    # PSEUDOCODE: lock the implemented proxy, outcome-independent splits and all sensitivity comparisons.
    if plan.get('purpose') != 'exploratory_next_observation_timing_shift_not_clinical_warning_validation':
        raise ValueError('This workflow does not validate clinical rhythm-disruption warnings.')
    from ..io.sources.hospital_workbook import FIELDS, META, CLOCKS
    features = plan['features']
    forbidden = set(META) | set(CLOCKS) | {'exercise_type'}
    if len(features) < 3 or len(set(features)) != len(features) or not set(features) <= set(FIELDS) - forbidden:
        raise ValueError('Use unique numeric predictors disjoint from proxy outcome clock fields.')
    if plan['folds'] != 5 or type(plan['seed']) is not int or plan['seed'] < 0:
        raise ValueError('This pilot uses five deterministic person folds.')
    if plan['alert_quantiles'] != [.8, .9, .95] or plan['primary_alert_quantile'] != .9:
        raise ValueError('Keep the declared primary and sensitivity screening budgets.')
    if plan['shift_minutes'] != [30, 60, 90] or plan['primary_shift_minutes'] != 60:
        raise ValueError('Keep all declared timing-change cutoffs.')
    if plan['recording_month_day_range'] != ['08-31', '09-20']:
        raise ValueError('Verify the source observation period before changing date handling.')
    if type(plan['bootstrap_replicates']) is not int or plan['bootstrap_replicates'] < 100:
        raise ValueError('Declare at least 100 participant resamples.')
    if type(plan['epsilon']) not in (int, float) or not np.isfinite(plan['epsilon']) or plan['epsilon'] <= 0:
        raise ValueError('Invalid numerical zero tolerance.')


def period_offset(row):
    # PSEUDOCODE: keep only explicit month/day in the documented period; never recover a lost date or invent a year.
    if row['date']['status'] != 'year_missing' or not row['date'].get('month_day'):
        return None
    month, day = map(int, row['date']['month_day'].split('-'))
    return 0 if (month, day) == (8, 31) else day if month == 9 and 1 <= day <= 20 else None


def circular_difference(a, b):
    # PSEUDOCODE: measure shortest clock displacement across midnight in minutes.
    return abs((a - b + 720) % 1440 - 720)


def proxy_change(current, following):
    """Later timing change is neither a confirmed onset nor a claim of worsening."""
    # PSEUDOCODE: require ordered dates and clock evidence -> calculate two domain shifts without filling gaps.
    base = {'proxy_status': 'unavailable_date', 'gap_days': None, 'sleep_shift_min': None,
            'meal_shift_min': None, 'joint_shift_min': None}
    first, second = period_offset(current), period_offset(following)
    if first is None or second is None:
        return base
    if second <= first:
        return {**base, 'proxy_status': 'nonincreasing_date'}
    base['gap_days'] = second - first
    clocks = [[row['values'].get(f) for f in TARGET_FIELDS] for row in (current, following)]
    if any(v is None or not np.isfinite(v) or not 0 <= v < 1440 for values in clocks for v in values):
        return {**base, 'proxy_status': 'missing_clock_evidence'}
    midpoints = []
    for start, end, *_ in clocks:
        duration = (end - start) % 1440
        if duration == 0:
            return {**base, 'proxy_status': 'ambiguous_sleep_interval'}
        midpoints.append((start + duration / 2) % 1440)
    sleep = circular_difference(*midpoints)
    meals = float(np.mean([circular_difference(a, b) for a, b in zip(clocks[0][2:], clocks[1][2:])]))
    return {**base, 'proxy_status': 'observed_timing_shift', 'sleep_shift_min': sleep,
            'meal_shift_min': meals, 'joint_shift_min': min(sleep, meals)}


def group_rows(rows):
    # PSEUDOCODE: keep each person's source order explicit and reject duplicate or missing occasions.
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['participant_id']].append(row)
    for person, records in grouped.items():
        records.sort(key=lambda r: r['occasion'])
        if [r['occasion'] for r in records] != [1, 2, 3, 4]:
            raise ValueError(f'{person}: this protocol expects four source recording occasions.')
    return grouped


def score_vector(row, features):
    # PSEUDOCODE: retain incomplete predictors as unavailable rather than imputing a healthy score.
    values = [row['values'].get(f) for f in features]
    return None if any(v is None or not np.isfinite(v) for v in values) else np.asarray(values, dtype=float)


def held_out_scores(rows, plan, *, scored_occasions=(1, 2, 3), calibration_occasions=(1, 2, 3)):
    """All scores use only current numeric inputs; thresholds never use proxy outcomes."""
    # PSEUDOCODE: split people -> fit first-record background -> calibrate on separate people -> score held-out people.
    check_plan(plan)
    for occasions in (scored_occasions, calibration_occasions):
        if not occasions or len(set(occasions)) != len(occasions) or not set(occasions) <= {1, 2, 3}:
            raise ValueError('Declare unique prediction/calibration occasions before the final record.')
    grouped = group_rows(rows)
    people = np.asarray(sorted(grouped))
    order = np.random.default_rng(plan['seed']).permutation(len(people))
    folds = [people[x].tolist() for x in np.array_split(order, plan['folds'])]
    features = sorted(plan['features']); predictions, lineage = [], []
    for fold in range(plan['folds']):
        test, calibration = folds[fold], folds[(fold+1) % plan['folds']]
        reference_people = sorted(set(people.tolist()) - set(test) - set(calibration))
        complete_reference = [(p, score_vector(grouped[p][0], features)) for p in reference_people]
        complete_reference = [(p, x) for p, x in complete_reference if x is not None]
        if len(complete_reference) < 9:
            raise ValueError('Insufficient independent reference people for the sDNB arithmetic.')
        reference = np.asarray([v for _, v in complete_reference]); center = reference.mean(0); scale = reference.std(0, ddof=1)
        if np.any(scale <= plan['epsilon']):
            raise ValueError('Constant reference feature; do not select a new panel after seeing results.')
        normalized_reference = (reference - center) / scale

        def score(row):
            # PSEUDOCODE: compute the unchanged sDNB branch formula and same-feature deviation control.
            vector = score_vector(row, features)
            if vector is None:
                return {'scores': dict.fromkeys(METHODS), 'module': [], 'status': 'missing_current_predictor'}
            dev, delta = network_changes(normalized_reference, ((vector-center)/scale)[None, :])
            parts = personal_components(dev[0], delta[0], plan['epsilon'])
            return {'scores': {'sdnb_exploratory': parts['score'] if parts else None,
                               'mean_deviation': float(dev[0].mean())},
                    'module': [features[i] for i in parts['module']] if parts else [],
                    'status': 'scored' if parts else 'undefined_sdnb_ratio'}

        calibration_rows = [r for p in calibration for r in grouped[p] if r['occasion'] in calibration_occasions]
        calibrated = [(r, score(r)) for r in calibration_rows]
        common = [(r, s) for r, s in calibrated if all(s['scores'][m] is not None for m in METHODS)]
        if len({r['participant_id'] for r, _ in common}) < 2:
            raise ValueError('Insufficient calibration people for even an exploratory percentile.')
        thresholds = {m: {str(q): float(np.quantile([s['scores'][m] for _, s in common], q, method='linear'))
                           for q in plan['alert_quantiles']} for m in METHODS}
        lineage.append({'fold': fold, 'test_people': test, 'calibration_people': calibration,
            'reference_people': [p for p, _ in complete_reference], 'background_is_confirmed_stable': False,
            'reference_records': [grouped[p][0]['record_id'] for p, _ in complete_reference],
            'features': features, 'center': center.tolist(), 'scale': scale.tolist(),
            'calibration_records': [r['record_id'] for r, _ in common], 'thresholds': thresholds})
        for person in test:
            for current in (r for r in grouped[person] if r['occasion'] in scored_occasions):
                scored = score(current)
                predictions.append({'record_id': current['record_id'], 'participant_id': person,
                    'occasion': current['occasion'], 'fold': fold, **scored,
                    'alerts': {m: {str(q): int(scored['scores'][m] > thresholds[m][str(q)])
                                   if scored['scores'][m] is not None else None for q in plan['alert_quantiles']} for m in METHODS}})
    return sorted(predictions, key=lambda r: (r['participant_id'], r['occasion'])), lineage


def metrics(rows, method, cutoff, quantile):
    # PSEUDOCODE: compare eligible held-out alerts with the declared proxy, never relabel unknowns as negatives.
    from sklearn.metrics import roc_auc_score
    valid = [r for r in rows if r['joint_shift_min'] is not None and all(r['scores'][m] is not None for m in METHODS)]
    n = len(valid)
    if not n:
        return {'n': 0, 'accuracy': None, 'balanced_accuracy': None, 'spearman': None,
                'sensitivity': None, 'specificity': None, 'roc_auc': None, 'always_no_change_accuracy': None}
    y = np.asarray([r['joint_shift_min'] >= cutoff for r in valid]); predicted = np.asarray([r['alerts'][method][str(quantile)] for r in valid], dtype=bool)
    tn, fp, fn, tp = (int(np.sum(~y & ~predicted)), int(np.sum(~y & predicted)), int(np.sum(y & ~predicted)), int(np.sum(y & predicted)))
    sensitivity = tp/(tp+fn) if tp+fn else None; specificity = tn/(tn+fp) if tn+fp else None
    scores = [r['scores'][method] for r in valid]; changes = [r['joint_shift_min'] for r in valid]
    rho = float(spearmanr(scores, changes).statistic) if len(set(scores)) > 1 and len(set(changes)) > 1 else None
    return {'n': n, 'people': len({r['participant_id'] for r in valid}), 'positives': int(y.sum()),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': (tp+tn)/n,
        'balanced_accuracy': (sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
        'sensitivity': sensitivity, 'specificity': specificity, 'spearman': rho,
        'roc_auc': float(roc_auc_score(y, scores)) if len(set(y.tolist())) == 2 else None,
        'always_no_change_accuracy': float((~y).mean())}


def evaluate_predictions(rows, plan):
    # PSEUDOCODE: retain all cutoff/budget comparisons and cluster-resample the fixed primary predictions in paired draws.
    comparisons = [{ 'method': m, 'shift_minutes': cutoff, 'alert_quantile': q, **metrics(rows, m, cutoff, q)}
                   for cutoff in plan['shift_minutes'] for q in plan['alert_quantiles'] for m in METHODS]
    cutoff, q = plan['primary_shift_minutes'], plan['primary_alert_quantile']
    primary = {m: metrics(rows, m, cutoff, q) for m in METHODS}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['participant_id']].append(row)
    people = sorted(grouped); rng = np.random.default_rng(plan['seed'] + 1)
    keys = ('accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'spearman')
    samples = {m: {k: [] for k in keys} for m in (*METHODS, 'sdnb_minus_deviation')}
    for _ in range(plan['bootstrap_replicates']):
        draw = [r for p in rng.choice(people, len(people), replace=True) for r in grouped[p]]
        values = {m: metrics(draw, m, cutoff, q) for m in METHODS}
        for key in keys:
            for m in METHODS:
                value = values[m].get(key)
                if value is not None and np.isfinite(value):
                    samples[m][key].append(value)
            if all(values[m].get(key) is not None for m in METHODS):
                samples['sdnb_minus_deviation'][key].append(values[METHODS[0]][key] - values[METHODS[1]][key])
    intervals = {m: {k: {'valid_draws': len(values), 'total_draws': plan['bootstrap_replicates'],
                        'interval': np.quantile(values, [.025, .975]).tolist() if len(values) == plan['bootstrap_replicates'] else None}
                     for k, values in measures.items()} for m, measures in samples.items()}
    return {'primary': primary, 'comparisons': comparisons, 'intervals': intervals,
            'attempted_pairs': len(rows), 'proxy_statuses': dict(Counter(r['proxy_status'] for r in rows)),
            'scored_pairs': sum(all(r['scores'][m] is not None for m in METHODS) for r in rows),
            'clinical_warning_accuracy': None, 'clinical_lead_time': None,
            'observation_gap_days': dict(Counter(r['gap_days'] for r in rows if r['joint_shift_min'] is not None))}


def run_proxy(source, plan, output, *, plots=False):
    # PSEUDOCODE: freeze a separate protocol -> issue held-out scores -> reveal later measurements -> report all comparisons.
    check_plan(plan)
    source, output = Path(source), Path(output)
    data = json.loads(source.read_text(encoding='utf-8')); digest = file_hash(source)
    output.mkdir(parents=True, exist_ok=False)
    from . import reference_stability
    _json(output / 'protocol.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'plan': plan,
        'source': str(source.resolve()), 'source_sha256': digest, 'original_workbook_sha256': data['source_sha256'],
        'code_hashes': {str(p): file_hash(p) for p in (Path(__file__).resolve(), Path(reference_stability.__file__).resolve())},
        'prior_inspection': 'Hospital data already inspected for descriptive correlations; not a new sealed clinical test set.'})
    predictions, lineage = held_out_scores(data['rows'], plan)
    # Save scores before loading later-record target fields into the evaluation table.
    _json(output / 'scores.json', {'predictions': predictions, 'folds': lineage})
    grouped = group_rows(data['rows']); paired = []
    for row in predictions:
        current = grouped[row['participant_id']][row['occasion']-1]
        following = grouped[row['participant_id']][row['occasion']]
        paired.append({**row, 'next_record_id': following['record_id'], **proxy_change(current, following)})
    evaluation = evaluate_predictions(paired, plan)
    _json(output / 'result.json', evaluation)
    _json(output / 'pairs.json', paired)
    flat = [{'record_id': r['record_id'], 'participant_id': r['participant_id'], 'next_record_id': r['next_record_id'],
             'gap_days': r['gap_days'], 'proxy_status': r['proxy_status'],
             'sleep_shift_min': r['sleep_shift_min'], 'meal_shift_min': r['meal_shift_min'],
             'proxy_change_60min': int(r['joint_shift_min'] >= 60) if r['joint_shift_min'] is not None else None,
             'sdnb_score': r['scores'][METHODS[0]], 'sdnb_alert': r['alerts'][METHODS[0]]['0.9'],
             'deviation_score': r['scores'][METHODS[1]], 'deviation_alert': r['alerts'][METHODS[1]]['0.9']} for r in paired]
    _csv(output / 'prediction-comparison.csv', flat, list(flat[0]))
    write_report(evaluation, output)
    if plots:
        plot_comparison(paired, output)
    if file_hash(source) != digest:
        raise ValueError('Normalized inputs changed during evaluation.')
    _json(output / 'manifest.json', {'files': {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}})
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if not all(file_hash(output / f) == h for f, h in manifest['files'].items()):
        raise ValueError('Artifact hash mismatch.')
    return {'output': str(output.resolve()), 'primary': evaluation['primary'],
            'proxy_statuses': evaluation['proxy_statuses'], 'clinical_warning_accuracy': None}


def write_report(result, output):
    # PSEUDOCODE: use ordinary language to distinguish observed timing-change agreement from clinical prediction accuracy.
    lines = ['# 后一次作息变化：探索性比对', '',
        '用前一次记录的数值指标打分，再和后一次记录比对。临时参考为：睡眠中点变化至少 60 分钟，同时三餐时刻的平均变化至少 60 分钟。它表示作息时点变动，可能是改善、周末安排或其他变化，不代表确诊紊乱。', '',
        '每个人只在未使用自己的背景/阈值人群中接受测试。背景来自其他人的第一次记录，尚未证实稳定；sDNB 采用已有评分公式，但本试验不声称已经找出满足理论检验的 DNB 群组。临时提醒取独立校准人群分数的前 10%，不是已验证的临床门槛。', '',
        f'尝试配对 {result["attempted_pairs"]} 次，数值评分可用 {result["scored_pairs"]} 次。不能核实日期或时刻的配对仍保留，不充当阴性。', '',
        '| 方法 | 可比次数 | 临时参考符合率 | 平衡符合率 | 命中变化 | 漏掉变化 | 误提醒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, label in zip(METHODS, ('sDNB 公式探索', '同指标偏离程度对照')):
        m = result['primary'][name]
        accuracy = f'{m["accuracy"]:.2%}' if m['accuracy'] is not None else '不可计算'
        balanced = f'{m["balanced_accuracy"]:.2%}' if m['balanced_accuracy'] is not None else '不可计算'
        lines.append(f'| {label} | {m["n"]} | {accuracy} | {balanced} | {m.get("tp", "—")} | {m.get("fn", "—")} | {m.get("fp", "—")} |')
    primary = result['primary'][METHODS[0]]
    if primary['n']:
        lines += ['', f'其中有 {primary["positives"]} 次达到上述变化规则；如果每次都回答“不会变化”，符合率也有 {primary["always_no_change_accuracy"]:.2%}。不能只看总体百分比判断模型好坏。']
    lines += ['', '全部 30/60/90 分钟规则、80%/90%/95% 分数门槛及按人抽样区间见 result.json；没有按结果选取最好的一格。连续变化量与分数的秩相关也一并保存，不必只依赖某个二分门槛。', '',
        '逐条结果见 prediction-comparison.csv；分数、测试人员、背景与校准人员及阈值保存在 scores.json。原始医院表和既有实验未改写，也未写入临床标签。', '',
        '**本轮只能评价“与后续已记录时点变化的一致性”，不能得出真实节律紊乱正确率。** 不规则采样间隔不是已证实的提前量；没有记录的中间日期无法判断。bootstrap 区间仅针对这次固定划分的预测，未覆盖重新训练或更换背景人群的不确定性。', '',
        '依据：[sDNB 原文](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005633)；[睡眠规律性指标研究](https://academic.oup.com/sleep/article/44/10/zsab103/6232042)。60 分钟是本试验的透明比较尺度，并非论文确立的通用紊乱诊断线。', '']
    (output / '结论.md').write_text('\n'.join(lines), encoding='utf-8')


def plot_comparison(rows, output):
    # PSEUDOCODE: plot held-out scores against observed subsequent shifts; no synthetic trajectory or clinical labels.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid = [r for r in rows if r['joint_shift_min'] is not None and all(r['scores'][m] is not None for m in METHODS)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for method, ax in zip(METHODS, axes):
        ax.scatter([r['scores'][method] for r in valid], [r['joint_shift_min'] for r in valid], s=20, alpha=.7)
        ax.axhline(60, color='gray', linestyle='--', label='Exploratory 60-min rule')
        ax.set_xlabel(method); ax.set_ylabel('Next recorded joint timing shift (minutes)'); ax.legend(fontsize=8)
    fig.suptitle('Person-held-out comparison; not clinical disruption labels')
    fig.savefig(output / 'comparison.png', dpi=170); plt.close(fig)
