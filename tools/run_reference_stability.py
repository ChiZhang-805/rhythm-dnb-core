"""Test reference-person sensitivity with matched controls on the preserved exploratory cohort."""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import shutil

import joblib
import numpy as np

from rhythm_dnb.measures.scaling import fit_scaler
from rhythm_dnb.provenance import file_hash, fingerprint
from rhythm_dnb.research.experiment import choose_threshold, forecast_rows, policy_settings, validate_protocol
from rhythm_dnb.research.experiment_data import load_packet, save_json, OBJECTIVE, TEXT
from rhythm_dnb.research.experiment_gpu import implementation_id
from rhythm_dnb.research.evaluate import event_metrics
from rhythm_dnb.research.reference_stability import reference_draws, resample_scores, summarize_draws
from rhythm_dnb.research.temporal_warning import causal_history, causal_smooth, select_learner, select_smoothing
from rhythm_dnb.timebase import instant
from tools.run_dnbr_ablation import matrix, paired_accuracy_interval
from tools.run_dnbr_cross_validation import ROTATIONS, policy_decisions, validate_people, verify_parent
from tools.run_dnbr_warning import read_json
from tools.run_temporal_warning import (attach_preserved_predictions, check_config, make_features,
                                       development_ids, learned_scores)


METHODS = ('reference_robust_dnb', 'reference_robust_deviation', 'history_robust_dnb',
           'history_robust_control', 'history_dnb', 'history_baseline', 'temporal_dnb')
LEARNERS = METHODS[2:6]
SMOOTHERS = METHODS[:2] + ('temporal_dnb',)
LABELS = ('参考重采样 DNB', '参考重采样偏离对照', '历史＋参考稳定性＋DNB 监督模型',
          '历史＋参考稳定性监督对照', '原历史＋DNB 监督模型', '原历史监督对照', '原连续 DNB')
PAIRS = (('reference_robust_dnb', 'temporal_dnb'), ('reference_robust_dnb', 'reference_robust_deviation'),
         ('history_robust_dnb', 'history_robust_control'), ('history_robust_dnb', 'history_dnb'),
         ('history_robust_control', 'history_baseline'))


def check_stability_config(config):
    # PSEUDOCODE: refuse unsupported plans instead of silently ignoring their requested behavior.
    expected = {'purpose': 'exploratory_reference_resampling_without_outcome_changes',
                'primary_method': METHODS[0], 'secondary_method': METHODS[2],
                'selection': 'unchanged_temporal_development_search_and_alarm_policy',
                'summary_quantiles': [.25, .5, .75]}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Unsupported reference-stability plan.')
    if type(config['replicates']) is not int or config['replicates'] < 2 or type(config['seed']) is not int:
        raise ValueError('Invalid resampling budget or seed.')


def forecast_prefix(rows, days):
    # PSEUDOCODE: select the same first calendar visits per person without consulting outcomes or errors.
    groups = defaultdict(list)
    for row in rows:
        groups[row['participant_id']].append(row)
    ids = set()
    for records in groups.values():
        ordered = sorted(records, key=lambda r: instant(r['issued_at']))
        if len(ordered) < days:
            raise ValueError('Incomplete forecast prefix.')
        ids.update(r['record_id'] for r in ordered[:days])
    if len(ids) != days * len(groups):
        raise ValueError('Repeated forecast identifiers.')
    return [r for r in rows if r['record_id'] in ids]


def augment_features(rows, cache, summaries, temporal):
    # PSEUDOCODE: give both arms identical node/deviation histories; only the DNB arm receives network stability.
    ids = [r['record_id'] for r in rows]
    metadata = [{k: r[k] for k in ('record_id', 'participant_id', 'observed_at', 'issued_at')} for r in rows]
    histories, columns, valid = {}, {}, {}
    for name, keys in (('control', ('deviation',)), ('network', ('dnb', 'network_ratio'))):
        names = [f'log_{key}_{stat}' for key in keys for stat in ('median', 'iqr')]
        values = {rid: np.log1p([summaries[key][stat][rid] for key in keys for stat in ('median', 'iqr')]) for rid in ids}
        histories[name], columns[name], lineage = causal_history(metadata, values, names, temporal['history_days'])
        if any(lineage[rid] != cache['lineage'][rid] for rid in ids):
            raise ValueError('Forecast prefix omitted an available history record.')
        valid[name] = {rid: bool(np.isfinite(values[rid]).all()) for rid in ids}
    for name in ('history_robust_control', 'history_robust_dnb'):
        with_network = name.endswith('_dnb')
        cache['values'][name] = {rid: np.r_[cache['values']['history_baseline'][rid], histories['control'][rid],
                                                 histories['network'][rid] if with_network else []] for rid in ids}
        cache['columns'][name] = cache['columns']['history_baseline'] + columns['control'] + (columns['network'] if with_network else [])
        cache['current_valid'][name] = {rid: bool(cache['current_valid']['history_baseline'][rid] and valid['control'][rid]
                                                       and (not with_network or valid['network'][rid])) for rid in ids}
    for name, key in zip(METHODS[:2], ('dnb', 'deviation')):
        raw = {rid: float(summaries[key]['median'][rid]) if summaries[key]['valid'][rid] else None for rid in ids}
        cache['smooth'][name] = {h: causal_smooth(metadata, raw, h) for h in temporal['half_lives_days']}


def write_outputs(result, output):
    # PSEUDOCODE: present all prespecified arms and matched differences, never promote the highest test result.
    lines = ['# 参考人群稳定性实验', '', '原 180 名模拟人物、900 次判断、90 个构造事件；人群、结局及提醒规则不变。',
             '对 60 名稳定参考人物重复抽样 200 次，不增加独立样本。中间一半分数的跨度不是置信区间。', '',
             '| 方法 | 判断正确 | 平衡正确率 | 可判断比例 | 漏报事件 | 误报提醒 | 提前天数中位数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    metrics = [result['methods'][name]['test'] for name in METHODS]
    for label, m in zip(LABELS, metrics):
        c = m['risk_confusion']; accuracy = m['risk_balanced_accuracy']
        rate = '不可计算' if accuracy is None else f'{accuracy:.2%}'
        lines.append(f'| {label} | {c["tp"]+c["tn"]}/{m["classified_days"]} | {rate} | {m["classification_coverage"]:.1%} | '
                     f'{m["events"]-m["detected_events"]}/{m["events"]} | {m["false_alarms"]} | {m["median_lead_days"]} |')
    lines += ['', 'DNB 的额外贡献须看同条件对照；监督组合成绩不能称为纯 DNB 成绩。',
              '每人只接受一次外层测试。既有模拟数据已经反复查看，本轮仍为探索；不能当作真实人群或新的独立验证。',
              '配对区间按人抽样，未包含重新训练及多轮探索的不确定性。全部方法、模型、阈值、输出及代码保留。']
    (output / 'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    from matplotlib.figure import Figure
    from rhythm_dnb.research.plots import _save
    labels = ['Resampled DNB', 'Resampled deviation', 'History + stability + DNB learner',
              'History + stability control', 'Original history + DNB', 'Original history control', 'Original temporal DNB']
    figure = Figure(figsize=(13, 6), layout='constrained'); axes = figure.subplots(1, 2)
    bars = axes[0].barh(labels, [100*m['risk_balanced_accuracy'] if m['risk_balanced_accuracy'] is not None else 0 for m in metrics])
    axes[0].bar_label(bars, labels=[f'{m["risk_balanced_accuracy"]:.2%}' if m['risk_balanced_accuracy'] is not None else 'Unavailable' for m in metrics], padding=3)
    axes[0].invert_yaxis(); axes[0].set_xlim(0, 108); axes[0].set_xlabel('Balanced daily accuracy (%)')
    x = np.arange(len(metrics))
    axes[1].bar(x-.18, [m['events']-m['detected_events'] for m in metrics], .36, label='Missed events / 90')
    axes[1].bar(x+.18, [m['false_alarms'] for m in metrics], .36, label='False notifications')
    axes[1].set_xticks(x, labels, rotation=35, ha='right'); axes[1].legend()
    figure.suptitle('Exploratory authored simulation | 180 people, 900 decisions')
    _save(figure, output / 'comparison.png')


def run(args):
    # PSEUDOCODE: freeze a fresh plan, verify cached arithmetic, resample references, develop/calibrate/test, archive.
    parent, temporal_parent, output = (Path(getattr(args, k)).resolve() for k in ('parent', 'temporal_parent', 'output'))
    if output.exists():
        raise FileExistsError('Use a fresh output directory.')
    verify_parent(parent); verify_parent(temporal_parent)
    old_plan = read_json(parent / 'plan.json'); temporal_plan = read_json(temporal_parent / 'plan.json')
    previous = read_json(temporal_parent / 'result.json')
    data, manifest = load_packet(args.packet); people = validate_people(data)
    inference = attach_preserved_predictions(data, manifest, args.predictions, old_plan)
    if temporal_plan['packet_id'] != manifest['id'] or temporal_plan['inference_id'] != inference:
        raise ValueError('Parent experiments have different input bindings.')
    temporal = read_json(args.temporal_config); check_config(temporal)
    if temporal != temporal_plan['config']:
        raise ValueError('The existing temporal selection budget must remain unchanged.')
    config = read_json(args.config); check_stability_config(config)
    protocol = data['protocol']; validate_protocol(protocol)
    features = list(OBJECTIVE) + list(TEXT)
    scaler = fit_scaler([[r['features'][k] for k in features] for r in data['reference']], features)
    output.mkdir(parents=True); source = output / 'source'; source.mkdir()
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / 'src', source / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'))
    shutil.copytree(root / 'tools', source / 'tools', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in (args.config, args.temporal_config, root / 'pyproject.toml'):
        shutil.copy2(path, source / Path(path).name)
    prefixes = {role: forecast_prefix(data[role], protocol['forecast_days']) for role in ('train', 'validation', 'test')}
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'domain': manifest['domain'],
            'packet_id': manifest['id'], 'inference_id': inference, 'implementation_id': implementation_id(),
            'prediction_implementation_id': old_plan['implementation_id'], 'config': config, 'temporal_config': temporal,
            'parent_manifest_sha256': file_hash(parent / 'archive-manifest.json'),
            'temporal_manifest_sha256': file_hash(temporal_parent / 'archive-manifest.json'),
            'rotations': ROTATIONS, 'people': people, 'protocol': protocol, 'methods': METHODS, 'paired_comparisons': PAIRS,
            'forecast_records': {k: [r['record_id'] for r in v] for k, v in prefixes.items()},
            'historical_data_previously_inspected': True, 'test_used_for_current_fit_or_selection': False,
            'source': {p.relative_to(source).as_posix(): file_hash(p) for p in source.rglob('*') if p.is_file()},
            'runtime': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'joblib')}}
    save_json(output / 'plan.json', {**plan, 'id': fingerprint(plan)}); save_json(output / 'scaler.json', scaler)
    cached = {}; vectors = []; ordered_ids = []
    canonical = sorted(features); order = [features.index(k) for k in canonical]
    _, ref_ids, reference = matrix(parent / 'train-input' / 'reference.csv')
    for role in prefixes:
        cache = cached[role] = make_features(data[role], data['reference'], parent, role, scaler, temporal)
        ids, _, _ = development_ids(data[role], cache, protocol)
        if set(ids) != {r['record_id'] for r in prefixes[role]}:
            raise ValueError('Outcome-blind forecast prefix differs from the original evaluation dates.')
        _, row_ids, values = matrix(parent / (role+'-input') / 'targets.csv')
        lookup = dict(zip(row_ids, values))
        for row in prefixes[role]:
            ordered_ids.append(row['record_id']); vectors.append(lookup[row['record_id']][order])
        save_json(output / (role+'-arithmetic-audit.json'), cache['audit'])
        save_json(output / (role+'-lineage.json'), {r['record_id']: cache['lineage'][r['record_id']] for r in prefixes[role]})
        print('Verified original arithmetic and forecast dates: '+role, flush=True)
    draws = reference_draws([r['participant_id'] for r in data['reference']], config['replicates'], config['seed'])
    np.savez_compressed(output / 'reference-inputs.npz', reference=reference[:, order], targets=vectors, draws=draws)
    save_json(output / 'reference-identities.json', {'features': canonical, 'reference_records': ref_ids,
              'reference_people': [r['participant_id'] for r in data['reference']], 'target_records': ordered_ids})
    def progress(done, total):
        if done % 20 == 0 or done == total:
            print(f'Reference draws {done}/{total}', flush=True)
    values, memberships = resample_scores(reference[:, order], vectors, draws, protocol['epsilon'], progress=progress)
    np.savez_compressed(output / 'reference-scores.npz', **values, membership_counts=memberships)
    summaries = {}; report = {}
    for name, value in values.items():
        median, spread, valid = summarize_draws(value)
        summaries[name] = {stat: dict(zip(ordered_ids, arr)) for stat, arr in (('median', median), ('iqr', spread), ('valid', valid))}
        report[name] = {'evaluable_records': int(valid.sum()), 'total_records': len(valid),
                        'median_iqr': float(np.median(spread[valid])) if valid.any() else None,
                        'median_score': float(np.median(median[valid])) if valid.any() else None}
    save_json(output / 'reference-sensitivity.json', report)
    for role, rows in prefixes.items():
        augment_features(rows, cached[role], summaries, temporal)
    pooled = defaultdict(lambda: [[], [], []]); folds = []
    for number, roles in enumerate(ROTATIONS, 1):
        folder = output / f'fold-{number}'; folder.mkdir(); cache = cached[roles[0]]
        ids, y, groups = development_ids(data[roles[0]], cache, protocol)
        fitted, receipts, smooth = {}, {}, {}
        for name in LEARNERS:
            if not all(cache['current_valid'][name][rid] for rid in ids):
                raise ValueError('Incomplete development evidence; no selective training or draw removal.')
            x = np.array([cache['values'][name][rid] for rid in ids])
            fitted[name], receipts[name] = select_learner(x, y, groups, temporal, protocol['seed'])
            if set(receipts[name]['training_people']) != set(people[roles[0]]):
                raise ValueError('Learner omitted development people.')
            path = folder / (name+'.joblib'); joblib.dump(fitted[name], path)
            np.testing.assert_allclose(joblib.load(path).predict_proba(x), fitted[name].predict_proba(x), atol=0, rtol=0)
            receipts[name].update(columns=cache['columns'][name], model_sha256=file_hash(path))
            print(f'Fold {number}: {name} fit and replay verified', flush=True)
        for name in SMOOTHERS:
            vectors = {h: [scores[rid] for rid in ids] for h, scores in cache['smooth'][name].items()}
            smooth[name] = select_smoothing(vectors, y, groups, temporal, protocol['seed'])
        save_json(folder / 'models.json', {'learners': receipts, 'smoothing': smooth})
        frozen, methods = {}, {}
        for stage, role in zip(('validation', 'test'), roles[1:]):
            scores = {name: cached[role]['smooth'][name][s['half_life']] for name, s in smooth.items()}
            scores.update({name: learned_scores(prefixes[role], cached[role], name, model) for name, model in fitted.items()})
            for name in METHODS:
                days, events, periods, reasons = forecast_rows(data[role], scores[name], protocol)
                save_json(folder / (stage+'-'+name+'-forecasts.json'), [asdict(d) for d in days])
                if stage == 'validation':
                    frozen[name] = choose_threshold(days, events, periods, protocol)
                else:
                    threshold = frozen[name]['threshold']
                    methods[name] = {'threshold': threshold, 'test': event_metrics(days, threshold, events=events,
                                     monitoring=periods, **policy_settings(protocol)), 'label_reasons': reasons}
                    for target, part in zip(pooled[name], (policy_decisions(days, threshold), events, periods)):
                        target.extend(part)
            if stage == 'validation':
                save_json(folder / 'frozen-policy.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'methods': frozen, 'test_used': False})
        for name in ('history_dnb', 'history_baseline', 'temporal_dnb'):
            old = previous['folds'][number-1]['methods'][name]
            if methods[name]['test'] != old['test'] or not np.isclose(methods[name]['threshold'], old['threshold'], atol=1e-10, rtol=1e-8):
                raise ValueError('Original matched method no longer reproduces: '+name)
        fold = {'roles': roles, 'methods': methods, 'model_selection': receipts, 'smoothing_selection': smooth}
        folds.append(fold); save_json(folder / 'result.json', fold)
        print('Completed rotation '+str(number), flush=True)
    methods, pairs = {}, {}
    expected = set().union(*(set(people[k]) for k in prefixes))
    for name, (days, events, periods) in pooled.items():
        if {p.participant_id for p in periods} != expected or len(periods) != len(expected):
            raise ValueError('Pooled evaluation omitted or repeated people.')
        metrics = event_metrics(days, .5, events=events, monitoring=periods, **policy_settings(protocol))
        metrics.update(roc_auc=None, average_precision=None); methods[name] = {'test': metrics}
        save_json(output / (name+'-pooled-decisions.json'), [asdict(d) for d in days])
    for a, b in PAIRS:
        pairs[a+'_minus_'+b] = paired_accuracy_interval(pooled[a][0], pooled[b][0], protocol['seed'], protocol['interval_repetitions'])
    result = {'domain': manifest['domain'], 'plan_id': fingerprint(plan), 'primary_method': config['primary_method'],
              'secondary_method': config['secondary_method'], 'primary_metric': protocol['primary_metric'],
              'methods': methods, 'folds': folds, 'paired_differences': pairs, 'reference_sensitivity': report,
              'original_methods_reproduced': True, 'clinical_accuracy_established': False}
    save_json(output / 'result.json', result); write_outputs(result, output)
    files = {p.relative_to(output).as_posix(): file_hash(p) for p in output.rglob('*') if p.is_file()}
    save_json(output / 'archive-manifest.json', {'files': files, 'id': fingerprint(files)})
    verify_parent(output)
    print('Complete and hash verified: '+str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('parent', 'temporal-parent', 'packet', 'predictions', 'output'):
        parser.add_argument('--'+option, required=True)
    parser.add_argument('--config', default='configs/reference_stability.json')
    parser.add_argument('--temporal-config', default='configs/temporal_warning.json')
    run(parser.parse_args())
