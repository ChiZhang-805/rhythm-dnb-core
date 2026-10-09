"""End-to-end authored-cohort experiments; never bypass the formal observed-data gates."""

from collections import defaultdict
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
import csv
import numpy as np

from ..config import StudyConfig
from ..contracts import OutcomeEvent, EvaluationDay, MonitoringPeriod
from ..dnb.single_sample import sdnb_components
from ..measures.scaling import fit_scaler, transform
from ..outcomes.labels import future_label
from ..provenance import fingerprint
from ..timebase import instant
from .discover import discover_matrices
from .evaluate import event_metrics, cluster_intervals
from .experiment_data import DOMAIN, OBJECTIVE, TEXT, load_packet, save_json


def validate_protocol(protocol):
    # PSEUDOCODE: validate numerical research controls and the distinct authored panel identities.
    if protocol['domain'] != DOMAIN or tuple(protocol['objective_features']) != OBJECTIVE or tuple(protocol['text_features']) != TEXT:
        raise ValueError('This runner accepts only the explicit authored experiment panels.')
    if protocol['text_initialization'] != 'pinned_base_only' or protocol['primary_metric'] != 'risk_balanced_accuracy':
        raise ValueError('Unexpected initialization or primary endpoint.')
    for key in ('forecast_days', 'confirmation_days', 'discovery_pre_event_lead_hours', 'interval_repetitions'):
        if type(protocol[key]) is not int or protocol[key] < 1:
            raise ValueError('Invalid experiment setting: ' + key)
    fields = ('seed', 'horizon_days', 'min_lead_hours', 'reference_min_people', 'alarm_consecutive',
              'cooldown_days', 'max_false_alarms_per_30_days', 'module_stability', 'max_modules',
              'bootstrap_repetitions', 'permutation_repetitions', 'discovery_alpha', 'epsilon', 'pair_convention')
    return StudyConfig(**{k: protocol[k] for k in fields}, module_sizes=tuple(protocol['module_sizes']),
                       persistence_days=protocol['confirmation_days'] + 1,
                       calibration_min_events=protocol['calibration_min_event_people'])


def forecast_rows(rows, scores, protocol):
    # PSEUDOCODE: fix the same first monitoring days for everyone -> label future events using actual observed follow-up.
    groups = defaultdict(list)
    for row in rows:
        groups[row['participant_id']].append(row)
    forecasts, events, monitoring, reasons = [], [], [], defaultdict(int)
    for person, sequence in sorted(groups.items()):
        sequence.sort(key=lambda r: instant(r['observed_at']))
        onset_values = {r['outcome']['event_onset_at'] for r in sequence if r['outcome']['event_onset_at']}
        if len(onset_values) > 1:
            raise ValueError('Multiple conflicting first-event onsets.')
        person_events = []
        if onset_values:
            onset = instant(next(iter(onset_values)))
            confirmed = onset + timedelta(days=protocol['confirmation_days'])
            person_events = [OutcomeEvent(person, onset, confirmed, DOMAIN)]
        events.extend(person_events)
        first = instant(sequence[0]['issued_at'])
        selected = {instant(r['issued_at']): r for r in sequence}
        monitoring.append(MonitoringPeriod(person, first.date(), (first + timedelta(days=protocol['forecast_days'] - 1)).date(), 'UTC'))
        observed = {instant(r['observed_at']).date() for r in sequence}
        # Existing followup dates extend beyond the recorded sequence: cap them, not the outcome labels.
        followup = min(min(instant(r['outcome']['followup_end_at']) for r in sequence),
                       instant(sequence[-1]['issued_at']))
        available = max(instant(r['outcome']['label_observed_at']) for r in sequence)
        for offset in range(protocol['forecast_days']):
            issued = first + timedelta(days=offset)
            label = future_label(issued, person_events, followup_end=followup, observed_days=observed,
                                 horizon_days=protocol['horizon_days'], min_lead_hours=protocol['min_lead_hours'],
                                 confirmation_days=protocol['confirmation_days'], label_as_of=available)
            reasons[label.reason] += 1
            record = selected.get(issued)
            value = scores.get(record['record_id']) if record else None
            forecasts.append(EvaluationDay(person, issued, value, label.value, label.onset, available))
    return tuple(forecasts), tuple(events), tuple(monitoring), dict(reasons)


def policy_settings(protocol):
    # PSEUDOCODE: share the exact deployment policy between threshold fitting and held-out evaluation.
    return {'consecutive': protocol['alarm_consecutive'], 'cooldown_days': protocol['cooldown_days'],
            'horizon_days': protocol['horizon_days'], 'min_lead_hours': protocol['min_lead_hours'],
            'confirmation_days': protocol['confirmation_days']}


def choose_threshold(rows, events, monitoring, protocol):
    # PSEUDOCODE: search calibration scores only -> maximize detected events within the predeclared false-alarm budget.
    if len({e.participant_id for e in events}) < protocol['calibration_min_event_people']:
        return {'threshold': None, 'status': 'insufficient_calibration_events', 'candidates': []}
    scores = [r.score for r in rows if r.score is not None and r.label is not None]
    if not scores:
        return {'threshold': None, 'status': 'no_evaluable_scores', 'candidates': []}
    if sum(r.label == 0 and r.score is not None for r in rows) < validate_protocol(protocol).calibration_min_negative_days:
        return {'threshold': None, 'status': 'insufficient_calibration_negative_days', 'candidates': []}
    if not np.isfinite(scores).all() or min(scores) < 0:
        raise ValueError('Invalid experimental risk scores.')
    feasible, candidates = [], []
    for threshold in np.unique([0., *scores]):
        metrics = event_metrics(rows, float(threshold), events=events, monitoring=monitoring, **policy_settings(protocol))
        candidates.append({'threshold': float(threshold), 'event_sensitivity': metrics['event_sensitivity'],
                           'false_alarms_per_30_days': metrics['false_alarms_per_30_days']})
        if metrics['false_alarms_per_30_days'] is not None and metrics['false_alarms_per_30_days'] <= protocol['max_false_alarms_per_30_days']:
            feasible.append((metrics['event_sensitivity'], -metrics['false_alarms_per_30_days'], float(threshold)))
    best = max(feasible) if feasible else None
    return {'threshold': best[2] if best and best[0] > 0 else None,
            'status': 'frozen_on_calibration' if best and best[0] > 0 else 'no_useful_threshold', 'candidates': candidates}


def discover_cohort(rows, features, scaler, protocol, config):
    # PSEUDOCODE: use one earliest and one prespecified pre-event vector per development person, never calibration/test labels.
    groups = defaultdict(list)
    for row in rows:
        groups[row['participant_id']].append(row)
    stable, pre, selected = [], [], []
    for person, sequence in sorted(groups.items()):
        sequence.sort(key=lambda r: instant(r['issued_at']))
        onsets = {r['outcome']['event_onset_at'] for r in sequence if r['outcome']['event_onset_at']}
        if not onsets:
            continue
        if len(onsets) != 1:
            raise ValueError('Conflicting development onset.')
        onset = instant(next(iter(onsets)))
        candidates = [r for r in sequence if instant(r['issued_at']) <= onset - timedelta(hours=protocol['discovery_pre_event_lead_hours'])]
        if len(candidates) < 2:
            continue
        a, b = candidates[0], candidates[-1]
        stable.append([a['features'].get(k, np.nan) for k in features])
        pre.append([b['features'].get(k, np.nan) for k in features])
        selected.append({'participant_id': person, 'stable_record': a['record_id'], 'pre_event_record': b['record_id']})
    if len(stable) < 9:
        return {'chosen': [], 'candidates': [], 'status': 'insufficient_independent_pairs', 'pairs': selected}
    stable, pre = transform(stable, scaler), transform(pre, scaler)
    if not np.isfinite(stable).all() or not np.isfinite(pre).all():
        return {'chosen': [], 'candidates': [], 'status': 'incomplete_discovery_features', 'pairs': selected}
    if np.any(stable.std(0) <= config.epsilon) or np.any(pre.std(0) <= config.epsilon):
        return {'chosen': [], 'candidates': [], 'status': 'constant_discovery_features', 'pairs': selected}
    result = discover_matrices(stable, pre, features, config)
    return {**result, 'status': 'frozen' if result['chosen'] else 'no_qualified_DNB_module', 'pairs': selected}


def score_cohort(rows, features, reference, scaler, discovery, config):
    # PSEUDOCODE: keep reference fixed -> compute sDNB and simple magnitude/change comparators using only past information.
    scores = {name: {} for name in ('dnb', 'reference_deviation', 'previous_day_change')}
    components, previous = {}, {}
    for row in sorted(rows, key=lambda r: (r['participant_id'], instant(r['issued_at']))):
        rid, person = row['record_id'], row['participant_id']
        vector = transform([row['features'].get(k, np.nan) for k in features], scaler)
        complete = np.isfinite(vector).all()
        scores['reference_deviation'][rid] = float(np.mean(np.abs(vector))) if complete else None
        old = previous.get(person)
        adjacent = old is not None and instant(row['issued_at']) - old[0] == timedelta(days=1)
        scores['previous_day_change'][rid] = float(np.mean(np.abs(vector - old[1]))) if complete and adjacent and np.isfinite(old[1]).all() else None
        previous[person] = (instant(row['issued_at']), vector)
        results = [sdnb_components(vector, reference, features, record['module'],
                                   min_samples=config.reference_min_people, epsilon=config.epsilon,
                                   pair_convention=config.pair_convention) for record in discovery['chosen']]
        components[rid] = results
        values = [r['score'] for r in results if r['valid']]
        # A fixed selected universe cannot silently drop a failed module.
        scores['dnb'][rid] = float(max(values)) if values and len(values) == len(results) else None
    return scores, components


def run_experiment(packet, output, *, text_predictions=None):
    # PSEUDOCODE: verify the packet -> fit development/calibration only -> freeze all methods -> evaluate test once with uncertainty.
    data, manifest = load_packet(packet)
    from .experiment_gpu import implementation_id
    code_id = implementation_id()
    protocol = data['protocol']; config = validate_protocol(protocol)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Result directories are immutable; choose a new output.')
    features = list(OBJECTIVE)
    inference_id = None
    if text_predictions is not None:
        from .experiment_gpu import attach_predictions
        inference_id = attach_predictions(data, manifest, text_predictions)
        features += list(TEXT)
    roles = {role: {r['participant_id'] for r in data[role]} for role in ('reference', 'train', 'validation', 'test')}
    if any(roles[a] & roles[b] for a in roles for b in roles if a < b):
        raise ValueError('DNB people overlap across roles.')
    reference_rows = data['reference']
    if len(reference_rows) != len(roles['reference']) or len(reference_rows) < config.reference_min_people:
        raise ValueError('Reference requires independent people, one row each.')
    reference_matrix = [[r['features'].get(k, np.nan) for k in features] for r in reference_rows]
    scaler = fit_scaler(reference_matrix, features)
    reference = transform(reference_matrix, scaler)
    discovery = discover_cohort(data['train'], features, scaler, protocol, config)
    output.mkdir(parents=True)
    save_json(output / 'discovery.json', discovery)
    save_json(output / 'scaler.json', scaler)
    all_scores, diagnostics = {}, {}
    for role in ('validation', 'test'):
        all_scores[role], diagnostics[role] = score_cohort(data[role], features, reference, scaler, discovery, config)
    frozen, results = {}, {}
    for method, scores in all_scores['validation'].items():
        rows, events, periods, reasons = forecast_rows(data['validation'], scores, protocol)
        frozen[method] = choose_threshold(rows, events, periods, protocol)
    # Save the entire selection before any test metrics are computed; failed discovery has no fallback module.
    save_json(output / 'frozen-policy.json', {'packet_id': manifest['id'], 'protocol_id': fingerprint(protocol),
              'inference_id': inference_id, 'implementation_id': code_id, 'methods': frozen, 'test_used_for_selection': False})
    for method, scores in all_scores['test'].items():
        rows, events, periods, reasons = forecast_rows(data['test'], scores, protocol)
        threshold = frozen[method]['threshold']
        metrics = event_metrics(rows, threshold, events=events, monitoring=periods, **policy_settings(protocol))
        intervals = cluster_intervals(rows, threshold, repetitions=protocol['interval_repetitions'], seed=protocol['seed'],
                                      events=events, monitoring=periods, **policy_settings(protocol))
        results[method] = {'status': frozen[method]['status'], 'threshold': threshold,
                           'test': metrics, 'intervals': intervals, 'label_reasons': reasons}
        save_json(output / (method + '-forecasts.json'), [asdict(r) for r in rows])
    result = {'domain': DOMAIN, 'packet_id': manifest['id'], 'features': features, 'inference_id': inference_id,
              'implementation_id': code_id,
              'primary_method': 'dnb', 'primary_metric': protocol['primary_metric'],
              'methods': results, 'discovery_status': discovery['status'],
              'qualified_modules': len(discovery['chosen']), 'test_used_for_selection': False,
              'clinical_accuracy_established': False,
              'interpretation': 'Historical authored-cohort method experiment; not fresh prospective or human-gold validation.'}
    save_json(output / 'dnb-components.json', diagnostics)
    export_summary(result, output)
    # Publish completion only after all mandatory evidence files have been saved.
    save_json(output / 'result.json', result)
    return result


def export_summary(result, output):
    # PSEUDOCODE: export a compact table and Chinese result note without converting missing results to zero.
    output = Path(output)
    keys = ('risk_accuracy', 'risk_balanced_accuracy', 'event_sensitivity', 'alarm_ppv',
            'false_alarms_per_30_days', 'median_lead_days', 'classification_coverage')
    with (output / 'metrics.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['method', 'status', *keys])
        for name, item in result['methods'].items():
            writer.writerow([name, item['status'], *[item['test'][key] for key in keys]])
    lines = ['# 本次预警实验', '', '数据：现有自建模拟队列。不是临床验证，也不是全新的前瞻性测试。', '',
             '主指标：每天判断未来 7 天是否发生事件的平衡正确率；至少提前 24 小时。',
             '风险判断与实际提醒分开计算；提醒需连续两天超过阈值，并有 7 天冷却期。', '',
             '| 方法 | 平衡正确率 | 事件检出率 | 可判断比例 |', '|---|---:|---:|---:|']
    for name, item in result['methods'].items():
        values = [item['test'][key] for key in ('risk_balanced_accuracy', 'event_sensitivity', 'classification_coverage')]
        text = ['未形成可评估策略' if value is None else f'{value:.1%}' for value in values]
        lines.append('| ' + name + ' | ' + ' | '.join(text) + ' |')
    lines += ['', 'DNB 合格指标组：' + str(result['qualified_modules']) + '。没有合格组时保留未知，不自动改用其他方法冒充 DNB。',
              '95% 人员重采样区间、混淆矩阵和提前天数见 result.json；逐日输出见各 forecasts.json。',
              '阈值只从校准人群选择。结果不自动证明已达到实际部署要求。']
    with (output / 'summary.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
