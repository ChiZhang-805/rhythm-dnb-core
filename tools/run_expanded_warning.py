"""Fit on train people, freeze on calibration people, then evaluate the existing expanded test split."""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib
import numpy as np

from rhythm_dnb.provenance import file_hash, fingerprint
from rhythm_dnb.research.evaluate import event_metrics, replay
from rhythm_dnb.research.expanded_warning import (sequence_features, forecast_objects, policy,
                                                 threshold_table, paired_uncertainty)
from rhythm_dnb.research.experiment_data import save_json
from rhythm_dnb.research.temporal_warning import causal_smooth, select_learner, select_smoothing


LABELS = {'personal_dnb': '个人 DNB', 'rolling_dnb': '滚动 DNB', 'mean_deviation': '偏离对照',
          'history_control': '历史指标对照', 'history_dnb': '历史指标＋DNB'}


def read_json(path):
    # PSEUDOCODE: load a frozen artifact without modifying it.
    return json.loads(Path(path).read_text(encoding='utf-8'))


def summarize(result, output):
    # PSEUDOCODE: report every prespecified method, including failures; never relabel a combined model as pure DNB.
    lines = ['# 扩大数据后的预警实验', '',
             '245 条构造随访序列，每条 84 天；147 条训练、49 条校准、49 条测试。人物划分和结局保持不变。',
             '本次用七项客观指标预测未来七天内是否开始紊乱；文本评分另行补算，不虚构新增随访的原文。', '',
             '| 方法 | 正确率 | 平衡正确率 | 漏报事件 | 误报提醒 | 提前天数中位数 |',
             '|---|---:|---:|---:|---:|---:|']
    for name, label in LABELS.items():
        m = result['methods'][name]['test']
        lines.append(f'| {label} | {m["risk_accuracy"]:.2%} | {m["risk_balanced_accuracy"]:.2%} | '
                     f'{m["events"]-m["detected_events"]}/{m["events"]} | {m["false_alarms"]} | {m["median_lead_days"]} |')
    count = result['methods']['personal_dnb']['test']
    lines += ['', f'主测试含 {count["classified_days"]} 次有完整参考答案的判断，{count["events"]} 个事件。'
              f'共同监测日历共 {count["forecast_days"]} 人日；发病后与尾部随访不足的日期不当作阴性。',
              '正确率按每日风险判断统计；提醒另执行连续两天与七天冷却规则。门槛仅由校准组选择，误报预算相同。',
              '个人 DNB 使用本人前 28 个稳定日作为参考；滚动 DNB 用过去窗口。两者属于本次纵向扩展，'
              '不代表原经典群组检验已经通过。构造数据结果不能直接当作真实人群表现。']
    (output / 'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    from matplotlib.figure import Figure
    from rhythm_dnb.research.plots import _save
    fig = Figure(figsize=(11, 5), layout='constrained'); axes = fig.subplots(1, 2)
    names = ['Personal DNB', 'Rolling DNB', 'Deviation control', 'History control', 'History + DNB']
    metrics = [result['methods'][n]['test'] for n in LABELS]
    bars = axes[0].barh(names, [m['risk_balanced_accuracy']*100 for m in metrics])
    axes[0].bar_label(bars, fmt='%.1f%%', padding=3); axes[0].set_xlim(0, 108); axes[0].invert_yaxis()
    axes[0].set_xlabel('Balanced daily accuracy (%)')
    x = np.arange(len(names)); axes[1].bar(x-.2, [m['events']-m['detected_events'] for m in metrics], .4, label='Missed events')
    axes[1].bar(x+.2, [m['false_alarms'] for m in metrics], .4, label='False alerts')
    axes[1].set_xticks(x, names, rotation=30, ha='right'); axes[1].legend()
    fig.suptitle('Authored 84-day follow-up | held-out 49 sequences')
    _save(fig, output / 'comparison.png')


def run(packet, output):
    # PSEUDOCODE: validate sealed inputs -> compute causal features -> fit/develop -> freeze thresholds -> test once.
    packet, output = Path(packet), Path(output)
    plan = read_json(packet / 'plan.json')
    if fingerprint({k: v for k, v in plan.items() if k != 'id'}) != plan['id']:
        raise ValueError('Packet plan changed.')
    for name, digest in plan['files'].items():
        if Path(name).name != name or file_hash(packet / name) != digest:
            raise ValueError('Packet changed: ' + name)
    rows = read_json(packet/'measurements.json'); answers = {r['record_id']: r for r in read_json(packet/'answers.json')}
    sequences = {r['sequence_id']: r for r in read_json(packet/'sequences.json')}; config = read_json(packet/'config.json')
    if len(answers) != len(rows) or set(answers) != {r['record_id'] for r in rows}:
        raise ValueError('Duplicate/missing outcome identities.')
    roles = {role: [r for r in rows if r['split'] == role] for role in ('train', 'calibration', 'test')}
    people = {k: {r['parent_participant_id'] for r in v} for k, v in roles.items()}
    if any(people[a] & people[b] for a in people for b in people if a < b):
        raise ValueError('Parent identity leaked across roles.')
    if set(config['methods']) != set(LABELS):
        raise ValueError('Unknown requested method.')
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    files = {p.relative_to(root).as_posix(): file_hash(p) for p in (root/'src').rglob('*.py')}
    files['tools/run_expanded_warning.py'] = file_hash(Path(__file__))
    source = output/'source'
    for name in files:
        target = source/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(root/name, target)
    save_json(output/'execution-plan.json', {'packet_id': plan['id'], 'config': config, 'source': files,
              'source_id': fingerprint(files), 'created_at': datetime.now(timezone.utc).isoformat(),
              'people': {k: sorted(v) for k, v in people.items()}, 'test_used_for_selection': False})
    groups = defaultdict(list)
    for row in rows:
        groups[row['participant_id']].append(row)
    features, scalers = {}, {}
    for number, (person, items) in enumerate(sorted(groups.items()), 1):
        values, scaler = sequence_features(items, config); features.update(values); scalers[person] = scaler
        if number % 25 == 0:
            print(json.dumps({'stage': 'causal_features', 'people': number, 'total': len(groups)}), flush=True)
    save_json(output/'personal-baselines.json', scalers)
    ids = sorted(features)
    np.savez_compressed(output/'features.npz', record_ids=ids,
        control=[features[k]['control'] for k in ids], network=[features[k]['network'] for k in ids])
    active = [r for r in rows if r['record_id'] in features]
    dev = [r for r in roles['train'] if answers[r['record_id']]['future_event_7d'] is not None]
    y = np.asarray([answers[r['record_id']]['future_event_7d'] for r in dev], int)
    group_ids = np.asarray([r['parent_participant_id'] for r in dev])
    dev_ids = [r['record_id'] for r in dev]
    temporal = config['temporal']; all_scores, fits = {}, {}
    for method in ('personal_dnb', 'mean_deviation', 'rolling_dnb'):
        windows = config['rolling_windows'] if method == 'rolling_dnb' else [None]
        candidates = []
        for window in windows:
            raw = {rid: features[rid]['rolling'][window] if window is not None else features[rid][method] for rid in ids}
            raw = {k: float(v) if np.isfinite(v) else None for k, v in raw.items()}
            smoothed = {h: causal_smooth(active, raw, h) for h in temporal['half_lives_days']}
            selected = select_smoothing({h: [v[rid] for rid in dev_ids] for h, v in smoothed.items()},
                                        y, group_ids, temporal, config['seed'])
            quality = next(t['mean_auc'] for t in selected['trials'] if t['half_life'] == selected['half_life'])
            candidates.append((quality, window, selected, smoothed[selected['half_life']]))
        best = max(range(len(candidates)), key=lambda j: (round(candidates[j][0], 12), -j))
        quality, window, selected, scores = candidates[best]
        all_scores[method] = scores
        fits[method] = {'window': window, 'smoothing': selected,
                         'candidates': [{'window': w, 'development_auc': q, 'smoothing': s} for q, w, s, _ in candidates]}
    for method in ('history_control', 'history_dnb'):
        values = {rid: np.r_[features[rid]['control'], features[rid]['network'] if method == 'history_dnb' else []]
                  for rid in ids}
        model, receipt = select_learner([values[rid] for rid in dev_ids], y, group_ids, temporal, config['seed'])
        path = output/(method+'.joblib'); joblib.dump(model, path)
        predicted = model.predict_proba(np.array([values[rid] for rid in ids]))[:, 1]
        np.testing.assert_allclose(joblib.load(path).predict_proba(np.array([values[rid] for rid in ids]))[:, 1], predicted, atol=0, rtol=0)
        all_scores[method] = dict(zip(ids, map(float, predicted)))
        fits[method] = {**receipt, 'model_sha256': file_hash(path), 'input_columns': len(values[ids[0]])}
        print(json.dumps({'stage': 'development_model_fitted', 'method': method}), flush=True)
    save_json(output/'development-selection.json', fits)
    frozen, cal = {}, {}
    for method in config['methods']:
        days, events, periods = forecast_objects(roles['calibration'], answers, sequences, all_scores[method], config)
        chosen = threshold_table(days, config)
        metrics = event_metrics(days, chosen['threshold'], events=events, monitoring=periods, **policy(config))
        candidate = chosen['candidates'][chosen['selected_index']]
        for key in ('risk_confusion', 'detected_events', 'false_alarms', 'unknown_alarms'):
            if candidate[key] != metrics[key]:
                raise ValueError('Vectorized calibration differs from production replay: ' + key)
        frozen[method] = chosen; cal[method] = metrics
    save_json(output/'frozen-policy.json', {'packet_id': plan['id'], 'methods': frozen, 'test_used_for_selection': False})
    methods, evaluations = {}, {}
    for method in config['methods']:
        days, events, periods = forecast_objects(roles['test'], answers, sequences, all_scores[method], config)
        threshold = frozen[method]['threshold']
        metrics = event_metrics(days, threshold, events=events, monitoring=periods, **policy(config))
        methods[method] = {'threshold': threshold, 'calibration': cal[method], 'test': metrics}
        evaluations[method] = dict(days=days, events=events, periods=periods, threshold=threshold)
        save_json(output/(method+'-test-forecasts.json'), [asdict(r) for r in days])
        save_json(output/(method+'-test-alerts.json'), [{**asdict(r), 'warning': alarm, 'status': status}
                  for r, alarm, status in replay(days, threshold, consecutive=config['consecutive'], cooldown_days=config['cooldown_days'])])
        print(json.dumps({'stage': 'held_out_evaluation', 'method': method, 'balanced_accuracy': metrics['risk_balanced_accuracy']}), flush=True)
    uncertainty = paired_uncertainty(evaluations, config)
    result = {'packet_id': plan['id'], 'domain': plan['domain'], 'methods': methods, 'uncertainty': uncertainty,
              'primary_method': config['primary_method'], 'secondary_method': config['secondary_method'],
              'primary_metric': config['primary_metric'], 'source_id': fingerprint(files),
              'split_counts': {k: len(v) for k, v in people.items()},
              'total_followup_records': len(rows), 'test_used_for_selection': False,
              'clinical_accuracy_established': False, 'new_text_for_followup': False}
    save_json(output/'result.json', result); summarize(result, output)
    archive = {p.relative_to(output).as_posix(): file_hash(p) for p in output.rglob('*') if p.is_file()}
    save_json(output/'archive-manifest.json', {'files': archive, 'id': fingerprint(archive)})
    print(json.dumps({'stage': 'complete', 'output': str(output), 'files': len(archive)}), flush=True)


def main():
    # PSEUDOCODE: expose only an explicit frozen packet and a fresh result directory.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.packet, args.output)


if __name__ == '__main__':
    main()
