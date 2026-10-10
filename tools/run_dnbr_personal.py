"""Evaluate personal DNB and a separately named combined warning model without changing historical outcomes."""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np

from rhythm_dnb.measures.scaling import fit_scaler
from rhythm_dnb.provenance import file_hash, fingerprint
from rhythm_dnb.research.experiment import choose_threshold, forecast_rows, policy_settings
from rhythm_dnb.research.experiment_data import load_packet, save_json, OBJECTIVE, TEXT
from rhythm_dnb.research.experiment_gpu import attach_predictions, implementation_id
from rhythm_dnb.research.evaluate import event_metrics
from tools.personal_warning import audit_personal, predict_model, select_model
from tools.run_dnbr_ablation import paired_accuracy_interval
from tools.run_dnbr_cross_validation import ROTATIONS, execute_r, policy_decisions, validate_people, verify_parent
from tools.run_dnbr_warning import export_scores_input, read_json, read_scores


METHODS = ('personal_dnb', 'combined_warning', 'magnitude_learner', 'dnbr_sdnb', 'reference_deviation')
LABELS = ('个人化 DNB', '偏离程度＋个人 DNB 组合', '只学偏离程度的匹配对照', '原固定指标组 DNB', '平均偏离程度法')


def check_config(config):
    # PSEUDOCODE: reject settings the implemented protocol cannot honor; no silently ignored knobs.
    expected = {'purpose': 'exploratory_personal_modules_and_combined_warning',
                'network': 'continuous_delta_pearson', 'distance': '2-abs(delta_PCC)', 'linkage': 'single',
                'features': 'all_prespecified_features', 'modules': 'all_proper_internal_tree_branches',
                'aggregation': 'maximum_all_modules_required', 'pair_convention': 'paper_k_squared',
                'primary_method': 'personal_dnb', 'secondary_method': 'combined_warning',
                'combined_predictors': ['reference_deviation', 'personal_dnb'], 'coefficient_bounds': 'nonnegative',
                'transform': 'log1p_then_training_standardization', 'selection_metric': 'participant_weighted_log_loss'}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Unsupported personal-warning protocol.')
    penalties = config['penalties']
    if (type(config['inner_folds']) is not int or config['inner_folds'] < 2 or
            not penalties or len(set(penalties)) != len(penalties) or
            any(not np.isfinite(v) or v <= 0 for v in penalties)):
        raise ValueError('Invalid development selection settings.')


def examples(rows, scores, protocol):
    # PSEUDOCODE: assemble only common eligible forecast days; current signals and their future labels stay explicit.
    a, events, periods, _ = forecast_rows(rows, scores['reference_deviation'], protocol)
    b, _, _, _ = forecast_rows(rows, scores['personal_dnb'], protocol)
    if [(d.participant_id, d.issued_at, d.label) for d in a] != [(d.participant_id, d.issued_at, d.label) for d in b]:
        raise ValueError('Predictor dates/labels are not aligned.')
    if any(d.label is None for d in a):
        raise ValueError('Development outcomes incomplete; no silent row selection.')
    x = np.array([[left.score if left.score is not None else np.nan,
                   right.score if right.score is not None else np.nan] for left, right in zip(a, b)])
    return x, np.array([d.label for d in a]), np.array([d.participant_id for d in a]), events, periods


def combined_scores(rows, scores, model, columns):
    # PSEUDOCODE: score all records independently using the frozen development model, preserving abstentions.
    x = np.array([[scores[k].get(r['record_id'], np.nan) for k in ('reference_deviation', 'personal_dnb')]
                  for r in rows], dtype=float)
    values = predict_model(model, x[:, columns])
    return {r['record_id']: float(v) if np.isfinite(v) else None for r, v in zip(rows, values)}


def write_summary(result, output):
    # PSEUDOCODE: show every locked route with its actual warning errors; never relabel the combination as pure DNB.
    lines = ['# 个人化 DNB 与组合预警实验', '',
             '同一批 180 名模拟人物，三组轮换、每人测试一次，共 900 次逐日判断。',
             '目标、结局、预测时间、报警规则均保留；模型权重仅在开发人群学习，阈值仅在校准人群选择。', '',
             '| 方法 | 答对 / 总数 | 正确率 | 覆盖率 | 漏报事件 / 90 | 误报提醒 | 提前天数中位数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, label in zip(METHODS, LABELS):
        m = result['methods'][name]['test']; c = m['risk_confusion']
        accuracy = f'{m["risk_accuracy"]:.2%}' if m['risk_accuracy'] is not None else '不可计算'
        lines.append(f'| {label} | {c["tp"] + c["tn"]}/{m["classified_days"]} | {accuracy} | '
                     f'{m["classification_coverage"]:.1%} | {m["events"] - m["detected_events"]}/{m["events"]} | '
                     f'{m["false_alarms"]} | {m["median_lead_days"]} |')
    lines += ['', '个人化：每次只用固定稳定参考和当前向量形成关联树，比较所有非平凡分支，不根据未来挑组合。',
              '这是连续关联网络的项目适配，不宣称通过原 DNB 统计检验；没有降低旧实验门槛。',
              '组合模型是独立的监督模型，不是纯 DNB；网络系数可为零，其成绩不能用来冒充 DNB 的增益。',
              '组合权重采用非负岭逻辑回归；惩罚强度用开发人群内部三折选定，各折单独拟合尺度。',
              '历史模拟测试已被查看，本轮为探索性结果，不能证明真实人群准确率。',
              '每种方法的配对区间、每人的输出和选定指标组已保留；没有在测试结果中挑选最高者替换主模型。']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot_result(result, output):
    # PSEUDOCODE: visualize locked methods on one common accuracy and event-error scale.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['Personal DNB', 'Combined warning', 'Magnitude learner', 'Fixed-module DNB', 'Mean deviation']
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    values = [result['methods'][k]['test']['risk_accuracy'] for k in METHODS]
    bars = axes[0].barh(names, [v * 100 if v is not None else 0 for v in values], color='#547A9D')
    axes[0].invert_yaxis(); axes[0].set_xlim(0, 100)
    axes[0].bar_label(bars, labels=[f'{v:.2%}' if v is not None else 'Unavailable' for v in values], padding=4)
    axes[0].set_xlabel('Daily accuracy (%)')
    index = np.arange(len(METHODS))
    missed = [result['methods'][k]['test']['events'] - result['methods'][k]['test']['detected_events'] for k in METHODS]
    false = [result['methods'][k]['test']['false_alarms'] for k in METHODS]
    axes[1].bar(index - .18, missed, .36, label='Missed events / 90', color='#C78355')
    axes[1].bar(index + .18, false, .36, label='False notifications', color='#849DA9')
    axes[1].set_xticks(index, names, rotation=25, ha='right'); axes[1].legend(frameon=False)
    figure.suptitle('180 people, 900 decisions | historical authored simulation')
    figure.tight_layout()
    figure.savefig(output / 'comparison.png', dpi=160, facecolor='white')
    figure.savefig(output / 'comparison.svg', facecolor='white')
    plt.close(figure)


def run(args):
    # PSEUDOCODE: freeze all routes -> audit label-free personal scores -> fit by development people -> calibrate -> test.
    parent, output = Path(args.parent).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a new experiment directory.')
    verify_parent(parent)
    previous, old_plan = read_json(parent / 'result.json'), read_json(parent / 'plan.json')
    data, manifest = load_packet(args.packet)
    people = validate_people(data); inference = attach_predictions(data, manifest, args.predictions)
    if manifest['id'] != old_plan['packet_id'] or inference != old_plan['inference_id']:
        raise ValueError('Inputs differ from the preserved experiment.')
    config = read_json(args.config); check_config(config)
    protocol = data['protocol']; settings = old_plan['settings']
    if config['epsilon'] != protocol['epsilon']:
        raise ValueError('Numerical guard differs from the parent experiment.')
    features = list(OBJECTIVE) + list(TEXT)
    scaler = fit_scaler([[r['features'][k] for k in features] for r in data['reference']], features)
    output.mkdir(parents=True); source = output / 'source'; source.mkdir()
    for name in ('run_dnbr_personal.py', 'score_dnbr_personal.R', 'personal_warning.py', 'score_dnbr.R',
                 'run_dnbr_cross_validation.py', 'run_dnbr_warning.py', 'run_dnbr_ablation.py', 'export_dnbr.py'):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    shutil.copy2(args.config, source / 'dnbr_personal.json')
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'domain': manifest['domain'],
            'packet_id': manifest['id'], 'inference_id': inference, 'implementation_id': implementation_id(),
            'parent_manifest_sha256': file_hash(parent / 'archive-manifest.json'), 'config': config,
            'rotations': ROTATIONS, 'people': people, 'protocol': protocol, 'methods': METHODS,
            'historical_data_previously_inspected': True, 'test_used_for_method_selection': False,
            'scoring_cache': 'Current vector plus fixed reference only; reusable across rotations; no outcome inputs.',
            'source': {p.name: file_hash(p) for p in source.iterdir()}}
    save_json(output / 'plan.json', {**plan, 'id': fingerprint(plan)})
    save_json(output / 'scaler.json', scaler)
    cached = {}
    for role in ('train', 'validation', 'test'):
        inputs, scored = output / (role + '-input'), output / (role + '-personal')
        _, targets = export_scores_input(data[role], data['reference'], features, scaler, [], settings,
                                         old_plan['discovery']['revision'], inputs)
        execute_r([args.rscript, str(Path(__file__).with_name('score_dnbr_personal.R')), str(inputs),
                   str(scored), args.library], output / (role + '-R.log'))
        personal, audit = audit_personal(inputs, scored, data[role])
        save_json(output / (role + '-audit.json'), audit)
        cached[role] = {'personal_dnb': personal, 'reference_deviation':
                        {r['record_id']: float(np.abs(x).mean()) for r, x in zip(data[role], targets)}}
        print('Personal scores audited: ' + role, flush=True)
    pooled = defaultdict(lambda: [[], [], []]); folds = []
    for number, roles in enumerate(ROTATIONS, 1):
        folder = output / f'fold-{number}'; folder.mkdir()
        x, y, groups, _, _ = examples(data[roles[0]], cached[roles[0]], protocol)
        fitted = {'combined_warning': select_model(x, y, groups, config, protocol['seed']),
                  'magnitude_learner': select_model(x[:, :1], y, groups, config, protocol['seed'])}
        if any(set(m['training_people']) != set(people[roles[0]]) for m in fitted.values()):
            raise ValueError('Model fit omitted development people.')
        save_json(folder / 'models.json', fitted)
        frozen, methods = {}, {}
        for role, role_name in zip(('validation', 'test'), roles[1:]):
            scores = dict(cached[role_name])
            for name, columns in (('combined_warning', [0, 1]), ('magnitude_learner', [0])):
                scores[name] = combined_scores(data[role_name], scores, fitted[name], columns)
            old_scores = parent / f'fold-{number}' / (role + '-scores/scores.csv')
            scores['dnbr_sdnb'] = read_scores(old_scores, [r['record_id'] for r in data[role_name]])
            for name in METHODS:
                days, events, periods, reasons = forecast_rows(data[role_name], scores[name], protocol)
                if role == 'validation':
                    frozen[name] = choose_threshold(days, events, periods, protocol)
                else:
                    threshold = frozen[name]['threshold']
                    metrics = event_metrics(days, threshold, events=events, monitoring=periods, **policy_settings(protocol))
                    methods[name] = {'threshold': threshold, 'test': metrics, 'label_reasons': reasons}
                    save_json(folder / (name + '-forecasts.json'), [asdict(d) for d in days])
                    for target, part in zip(pooled[name], (policy_decisions(days, threshold), events, periods)):
                        target.extend(part)
            if role == 'validation':
                save_json(folder / 'frozen-policy.json', {'created_at': datetime.now(timezone.utc).isoformat(),
                          'methods': frozen, 'test_used': False})
        for name in ('dnbr_sdnb', 'reference_deviation'):
            old = previous['folds'][number - 1]['methods'][name]
            if methods[name]['threshold'] != old['threshold'] or methods[name]['test'] != old['test']:
                raise ValueError('Preserved DNB or reference baseline no longer reproduces.')
        folds.append({'roles': roles, 'models': fitted, 'methods': methods})
        save_json(folder / 'result.json', folds[-1])
        print('Completed rotation ' + str(number), flush=True)
    methods, intervals = {}, {}
    expected_people = set().union(*(set(people[k]) for k in ('train', 'validation', 'test')))
    for name, (days, events, periods) in pooled.items():
        if {p.participant_id for p in periods} != expected_people or len(periods) != len(expected_people):
            raise ValueError('Pooled tests repeat or omit people.')
        metrics = event_metrics(days, .5, events=events, monitoring=periods, **policy_settings(protocol))
        metrics.update(roc_auc=None, average_precision=None)
        methods[name] = {'test': metrics}
        save_json(output / (name + '-pooled-decisions.json'), [asdict(d) for d in days])
        intervals[name] = paired_accuracy_interval(days, pooled['reference_deviation'][0],
                                                   protocol['seed'], protocol['interval_repetitions'])
    change = paired_accuracy_interval(pooled['personal_dnb'][0], pooled['dnbr_sdnb'][0],
                                     protocol['seed'], protocol['interval_repetitions'])
    result = {'domain': manifest['domain'], 'plan_id': fingerprint(plan), 'primary_method': 'personal_dnb',
              'methods': methods, 'folds': folds, 'paired_difference_from_reference': intervals,
              'personal_minus_fixed': change, 'clinical_accuracy_established': False}
    save_json(output / 'result.json', result)
    write_summary(result, output); plot_result(result, output)
    files = {p.relative_to(output).as_posix(): file_hash(p) for p in output.rglob('*') if p.is_file()}
    save_json(output / 'archive-manifest.json', {'files': files, 'id': fingerprint(files)})
    print('Complete: ' + str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('parent', 'packet', 'predictions', 'output', 'library'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--config', default='configs/dnbr_personal.json')
    parser.add_argument('--rscript', default='Rscript')
    run(parser.parse_args())
