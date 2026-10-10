"""Compare causal history with and without personal DNB on the sealed exploratory cohort."""

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import math
from pathlib import Path
import shutil

import joblib
import numpy as np

from rhythm_dnb.measures.scaling import fit_scaler, transform
from rhythm_dnb.provenance import file_hash, fingerprint
from rhythm_dnb.research.experiment import choose_threshold, forecast_rows, policy_settings, validate_protocol
from rhythm_dnb.research.experiment_data import load_packet, save_json, OBJECTIVE, TEXT
from rhythm_dnb.research.experiment_gpu import inference_tasks, implementation_id
from rhythm_dnb.research.evaluate import event_metrics
from rhythm_dnb.research.temporal_warning import causal_history, causal_smooth, select_learner, select_smoothing
from rhythm_dnb.timebase import instant
from rhythm_dnb.text.schema import CATEGORIES
from tools.personal_warning import audit_personal
from tools.run_dnbr_ablation import matrix, paired_accuracy_interval
from tools.run_dnbr_cross_validation import ROTATIONS, policy_decisions, validate_people, verify_parent
from tools.run_dnbr_warning import read_json


METHODS = ('temporal_dnb', 'temporal_reference', 'history_dnb', 'history_baseline',
           'personal_dnb', 'reference_deviation')
LABELS = ('连续个人 DNB', '连续平均偏离对照', '连续指标＋DNB 监督模型', '连续指标监督对照',
          '原个人 DNB', '原平均偏离对照')
PAIRS = (('temporal_dnb', 'personal_dnb'), ('temporal_dnb', 'temporal_reference'),
         ('history_dnb', 'history_baseline'), ('history_dnb', 'reference_deviation'))


def attach_preserved_predictions(data, manifest, path, verified_parent_plan):
    # PSEUDOCODE: reuse only the exact predictions already validated in the hash-verified parent experiment.
    result = read_json(path); tasks = inference_tasks(data)
    if (fingerprint({k: v for k, v in result.items() if k != 'id'}) != result['id'] or
            result['id'] != verified_parent_plan['inference_id'] or
            result['implementation_id'] != verified_parent_plan['implementation_id'] or
            manifest['id'] != verified_parent_plan['packet_id'] or result['packet_id'] != manifest['id'] or
            result['tasks_id'] != fingerprint(tasks) or
            result['checkpoint_dataset_id'] != data['text-development']['manifest']['id'] or
            result['initialization'] or not result['experimental_uncalibrated'] or
            set(result['predictions']) != set(tasks)):
        raise ValueError('Predictions differ from the verified parent or the sealed independent text experiment.')
    for key, estimates in result['predictions'].items():
        if set(estimates) != set(CATEGORIES[tasks[key]['category']][1]) or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100 for v in estimates.values()):
            raise ValueError('Invalid preserved text prediction.')
    for role in ('reference', 'train', 'validation', 'test'):
        for row in data[role]:
            for category, text in row['texts'].items():
                for key, value in result['predictions'][fingerprint([category, text])].items():
                    if 'text_' + key in TEXT:
                        row['features']['text_' + key] = value
    return result['id']


def check_config(config):
    # PSEUDOCODE: reject unsupported selectors and malformed grids instead of silently ignoring them.
    expected = {'purpose': 'exploratory_causal_history_and_dnb_increment', 'primary_method': 'temporal_dnb',
                'secondary_method': 'history_dnb', 'selection': 'person_weighted_development_log_loss',
                'smoothing_selection': 'mean_development_fold_auc'}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Unsupported temporal-warning protocol.')
    for key, minimum in (('history_days', 2), ('inner_folds', 2), ('tree_iterations', 1), ('tree_min_samples_leaf', 2)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError('Invalid integer setting: ' + key)
    for key in ('half_lives_days', 'logistic_c', 'tree_leaves', 'tree_l2'):
        values = config[key]
        if not values or len(set(values)) != len(values) or any(type(v) not in (int, float) or not np.isfinite(v) for v in values):
            raise ValueError('Invalid parameter grid: ' + key)
        if any(v < 0 if key == 'half_lives_days' else v <= 0 for v in values):
            raise ValueError('Invalid parameter range: ' + key)
    if 0 not in config['half_lives_days'] or any(type(v) is not int or v < 2 for v in config['tree_leaves']):
        raise ValueError('Include the no-smoothing control and valid tree sizes.')
    rate = config['tree_learning_rate']
    if type(rate) not in (int, float) or not np.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError('Invalid boosting learning rate.')


def make_features(rows, reference_rows, parent, role, scaler, config):
    # PSEUDOCODE: verify reused R scores against Python and sealed inputs before extracting causal history.
    inputs, scored = parent / (role + '-input'), parent / (role + '-personal')
    personal, audit = audit_personal(inputs, scored, rows)
    features, ids, targets = matrix(inputs / 'targets.csv')
    ref_features, ref_ids, reference = matrix(inputs / 'reference.csv')
    if features != scaler['features'] or ref_features != features or ids != [r['record_id'] for r in rows]:
        raise ValueError('Cached matrices do not match the frozen feature order.')
    if ref_ids != [r['record_id'] for r in reference_rows]:
        raise ValueError('Reference identities changed.')
    for actual, records in ((targets, rows), (reference, reference_rows)):
        expected = transform([[r['features'][k] for k in features] for r in records], scaler)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-9)
    parts = defaultdict(list)
    with (scored / 'components.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            parts[row['record_id']].append(row)
    network, winners = {}, {}
    for rid in ids:
        if personal[rid] is None:
            network[rid] = [np.nan, np.nan]; winners[rid] = None
            continue
        winner = max(parts[rid], key=lambda p: float(p['score']))
        ratio = float(winner['sPCC_in']) / float(winner['sPCC_out'])
        if not np.isfinite(ratio) or ratio < 0:
            raise ValueError('Invalid audited network ratio.')
        network[rid] = np.log1p([personal[rid], ratio])
        winners[rid] = {'module_id': int(winner['module_id']), 'features': winner['genes'].split(',')}
    # Pass only availability metadata to temporal feature construction; never the outcome object.
    metadata = [{k: r[k] for k in ('record_id', 'participant_id', 'observed_at', 'issued_at')} for r in rows]
    node_values = dict(zip(ids, targets))
    base, base_names, lineage = causal_history(metadata, node_values, features, config['history_days'])
    dnb, dnb_names, network_lineage = causal_history(metadata, network, ['log_personal_dnb', 'log_network_ratio'], config['history_days'])
    if lineage != network_lineage:
        raise ValueError('Matched histories do not use the same records.')
    mean = {rid: float(np.abs(x).mean()) for rid, x in node_values.items()}
    smooth = {name: {h: causal_smooth(metadata, scores, h) for h in config['half_lives_days']}
              for name, scores in (('temporal_dnb', personal), ('temporal_reference', mean))}
    return {'values': {'history_baseline': base, 'history_dnb': {rid: np.r_[base[rid], dnb[rid]] for rid in ids}},
            'columns': {'history_baseline': base_names, 'history_dnb': base_names + dnb_names},
            'current_valid': {'history_baseline': {rid: bool(np.isfinite(node_values[rid]).all()) for rid in ids},
                              'history_dnb': {rid: bool(np.isfinite(node_values[rid]).all() and np.isfinite(network[rid]).all()) for rid in ids}},
            'raw': {'personal_dnb': personal, 'reference_deviation': mean}, 'smooth': smooth,
            'audit': audit, 'lineage': lineage, 'selected_modules': winners}


def development_ids(rows, cache, protocol):
    # PSEUDOCODE: align features to the original eligible forecast dates, not post-event rows.
    days, _, _, _ = forecast_rows(rows, cache['raw']['reference_deviation'], protocol)
    lookup = {(r['participant_id'], instant(r['issued_at'])): r['record_id'] for r in rows}
    if len(lookup) != len(rows) or any(d.label is None for d in days):
        raise ValueError('Ambiguous dates or incomplete development outcomes.')
    ids = [lookup[(d.participant_id, instant(d.issued_at))] for d in days]
    return ids, np.array([d.label for d in days]), np.array([d.participant_id for d in days])


def learned_scores(rows, cache, name, fitted):
    # PSEUDOCODE: missing history can be modeled, but missing current evidence still means abstention.
    ids = [r['record_id'] for r in rows]
    valid = np.array([cache['current_valid'][name][rid] for rid in ids])
    scores = {rid: None for rid in ids}
    if valid.any():
        x = np.array([cache['values'][name][rid] for rid in ids])[valid]
        probabilities = fitted.predict_proba(x)[:, 1]
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError('Invalid learned probabilities.')
        scores.update({rid: float(p) for rid, p in zip(np.array(ids)[valid], probabilities)})
    return scores


def write_summary(result, output):
    # PSEUDOCODE: keep every prespecified result visible and distinguish the supervised combination from DNB.
    lines = ['# 连续记录与 DNB 增量实验', '',
             '180 名既有模拟人物，每人只测试一次，共 900 次判断；目标仍是未来七天节律紊乱预警。',
             '没有改人群、标签、预测日期或提醒规则。训练只用开发人物，阈值只用校准人物。', '',
             '| 方法 | 正确 / 总数 | 平衡正确率 | 覆盖率 | 漏报事件 / 90 | 误报提醒 | 提前天数中位数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, label in zip(METHODS, LABELS):
        m = result['methods'][name]['test']; c = m['risk_confusion']
        rate = '不可计算' if m['risk_balanced_accuracy'] is None else f'{m["risk_balanced_accuracy"]:.2%}'
        lines.append(f'| {label} | {c["tp"]+c["tn"]}/{m["classified_days"]} | {rate} | '
                     f'{m["classification_coverage"]:.1%} | {m["events"]-m["detected_events"]}/{m["events"]} | '
                     f'{m["false_alarms"]} | {m["median_lead_days"]} |')
    lines += ['', '连续 DNB 仅平滑既有个人 DNB 分数；监督组合模型另有学习步骤，不能称为纯 DNB。',
              '两种监督方法使用相同模型搜索预算；DNB 的额外价值应与“连续指标监督对照”比较。',
              '既有模拟测试已被多次查看，本轮是探索结果；配对区间按人重采样，未计重新训练不确定性或多次探索影响。',
              '模型、阈值、全部输出、代码及哈希均保留；不据测试成绩替换正式模型。']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot_result(result, output):
    # PSEUDOCODE: show accuracy, misses and false notifications for all locked methods on common axes.
    from matplotlib.figure import Figure
    from rhythm_dnb.research.plots import _save
    names = ['Temporal DNB', 'Temporal deviation', 'History + DNB learner', 'History-only learner', 'Raw personal DNB', 'Raw deviation']
    figure = Figure(figsize=(13, 5), layout='constrained'); axes = figure.subplots(1, 2)
    metrics = [result['methods'][k]['test'] for k in METHODS]
    bars = axes[0].barh(names, [100*m['risk_balanced_accuracy'] if m['risk_balanced_accuracy'] is not None else 0 for m in metrics])
    axes[0].bar_label(bars, labels=[f'{m["risk_balanced_accuracy"]:.2%}' if m['risk_balanced_accuracy'] is not None else 'Unavailable' for m in metrics], padding=3)
    axes[0].invert_yaxis(); axes[0].set_xlim(0, 108); axes[0].set_xlabel('Balanced daily accuracy (%)')
    x = np.arange(len(METHODS))
    axes[1].bar(x-.18, [m['events']-m['detected_events'] for m in metrics], .36, label='Missed events / 90')
    axes[1].bar(x+.18, [m['false_alarms'] for m in metrics], .36, label='False notifications')
    axes[1].set_xticks(x, names, rotation=35, ha='right'); axes[1].legend()
    figure.suptitle('Exploratory authored simulation | 180 people, 900 decisions')
    _save(figure, output / 'comparison.png')


def run(args):
    # PSEUDOCODE: freeze plan -> audit cached scores -> fit each development group -> calibrate -> test -> archive.
    parent, output = Path(args.parent).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh experiment directory.')
    verify_parent(parent)
    old_plan, previous = read_json(parent / 'plan.json'), read_json(parent / 'result.json')
    data, manifest = load_packet(args.packet)
    people = validate_people(data)
    inference = attach_preserved_predictions(data, manifest, args.predictions, old_plan)
    if manifest['id'] != old_plan['packet_id'] or inference != old_plan['inference_id']:
        raise ValueError('Inputs differ from the preserved parent experiment.')
    config = read_json(args.config); check_config(config)
    protocol = data['protocol']; validate_protocol(protocol)
    features = list(OBJECTIVE) + list(TEXT)
    scaler = fit_scaler([[r['features'][k] for k in features] for r in data['reference']], features)
    output.mkdir(parents=True); source = output / 'source'; source.mkdir()
    for name in ('run_temporal_warning.py', 'personal_warning.py', 'run_dnbr_ablation.py', 'run_dnbr_cross_validation.py', 'run_dnbr_warning.py', 'export_dnbr.py'):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    import rhythm_dnb.research.temporal_warning as implementation
    shutil.copy2(implementation.__file__, source / 'temporal_warning.py')
    shutil.copy2(args.config, source / 'temporal_warning.json')
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'domain': manifest['domain'],
            'packet_id': manifest['id'], 'inference_id': inference, 'implementation_id': implementation_id(),
            'prediction_implementation_id': old_plan['implementation_id'],
            'parent_manifest_sha256': file_hash(parent / 'archive-manifest.json'), 'config': config,
            'rotations': ROTATIONS, 'people': people, 'protocol': protocol, 'methods': METHODS, 'paired_comparisons': PAIRS,
            'historical_data_previously_inspected': True, 'test_used_for_current_fit_or_selection': False,
            'source': {p.name: file_hash(p) for p in source.iterdir()},
            'runtime': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'joblib')}}
    save_json(output / 'plan.json', {**plan, 'id': fingerprint(plan)}); save_json(output / 'scaler.json', scaler)
    cached = {}
    for role in ('train', 'validation', 'test'):
        cached[role] = make_features(data[role], data['reference'], parent, role, scaler, config)
        save_json(output / (role + '-audit.json'), cached[role]['audit'])
        save_json(output / (role + '-lineage.json'), cached[role]['lineage'])
        save_json(output / (role + '-selected-modules.json'), cached[role]['selected_modules'])
        print('Cached arithmetic and causal history verified: ' + role, flush=True)
    pooled = defaultdict(lambda: [[], [], []]); folds = []
    for number, roles in enumerate(ROTATIONS, 1):
        folder = output / f'fold-{number}'; folder.mkdir()
        cache = cached[roles[0]]; ids, y, groups = development_ids(data[roles[0]], cache, protocol)
        fitted, receipts, smooth = {}, {}, {}
        for name in ('history_dnb', 'history_baseline'):
            if not all(cache['current_valid'][name][rid] for rid in ids):
                raise ValueError('Incomplete development evidence; no selective training.')
            x = np.array([cache['values'][name][rid] for rid in ids])
            fitted[name], receipts[name] = select_learner(x, y, groups, config, protocol['seed'])
            if set(receipts[name]['training_people']) != set(people[roles[0]]):
                raise ValueError('Model fit omitted development people.')
            model_path = folder / (name + '.joblib'); joblib.dump(fitted[name], model_path)
            # Round-trip only this newly created local model, never an untrusted pickle.
            np.testing.assert_allclose(joblib.load(model_path).predict_proba(x), fitted[name].predict_proba(x), atol=0, rtol=0)
            receipts[name]['columns'] = cache['columns'][name]
            receipts[name]['model_sha256'] = file_hash(model_path)
            print(f'Fold {number} fitted {name}: {receipts[name]["selected"]}', flush=True)
        for name in ('temporal_dnb', 'temporal_reference'):
            vectors = {h: [scores[rid] for rid in ids] for h, scores in cache['smooth'][name].items()}
            smooth[name] = select_smoothing(vectors, y, groups, config, protocol['seed'])
        save_json(folder / 'models.json', {'learners': receipts, 'smoothing': smooth})
        frozen, methods = {}, {}
        for stage, role in zip(('validation', 'test'), roles[1:]):
            scores = dict(cached[role]['raw'])
            scores.update({name: cached[role]['smooth'][name][s['half_life']] for name, s in smooth.items()})
            scores.update({name: learned_scores(data[role], cached[role], name, model) for name, model in fitted.items()})
            for name in METHODS:
                days, events, periods, reasons = forecast_rows(data[role], scores[name], protocol)
                save_json(folder / (stage + '-' + name + '-forecasts.json'), [asdict(d) for d in days])
                if stage == 'validation':
                    frozen[name] = choose_threshold(days, events, periods, protocol)
                else:
                    threshold = frozen[name]['threshold']
                    metrics = event_metrics(days, threshold, events=events, monitoring=periods, **policy_settings(protocol))
                    methods[name] = {'threshold': threshold, 'test': metrics, 'label_reasons': reasons}
                    for target, part in zip(pooled[name], (policy_decisions(days, threshold), events, periods)):
                        target.extend(part)
            if stage == 'validation':
                save_json(folder / 'frozen-policy.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'methods': frozen, 'test_used': False})
        for name in ('personal_dnb', 'reference_deviation'):
            old = previous['folds'][number-1]['methods'][name]
            if methods[name]['test'] != old['test'] or methods[name]['threshold'] != old['threshold']:
                raise ValueError('Preserved personal DNB or reference baseline no longer reproduces.')
        fold = {'roles': roles, 'methods': methods, 'model_selection': receipts, 'smoothing_selection': smooth}
        folds.append(fold); save_json(folder / 'result.json', fold)
        print('Completed rotation ' + str(number), flush=True)
    methods, pairs = {}, {}
    expected_people = set().union(*(set(people[k]) for k in ('train', 'validation', 'test')))
    for name, (days, events, periods) in pooled.items():
        if {p.participant_id for p in periods} != expected_people or len(periods) != len(expected_people):
            raise ValueError('Pooled tests repeat or omit people.')
        metrics = event_metrics(days, .5, events=events, monitoring=periods, **policy_settings(protocol))
        metrics.update(roc_auc=None, average_precision=None)
        methods[name] = {'test': metrics}
        save_json(output / (name + '-pooled-decisions.json'), [asdict(d) for d in days])
    for a, b in PAIRS:
        pairs[a + '_minus_' + b] = paired_accuracy_interval(pooled[a][0], pooled[b][0], protocol['seed'], protocol['interval_repetitions'])
    result = {'domain': manifest['domain'], 'plan_id': fingerprint(plan), 'primary_method': config['primary_method'],
              'secondary_method': config['secondary_method'], 'primary_metric': protocol['primary_metric'],
              'methods': methods, 'folds': folds, 'paired_differences': pairs, 'clinical_accuracy_established': False}
    save_json(output / 'result.json', result); write_summary(result, output); plot_result(result, output)
    files = {p.relative_to(output).as_posix(): file_hash(p) for p in output.rglob('*') if p.is_file()}
    save_json(output / 'archive-manifest.json', {'files': files, 'id': fingerprint(files)})
    print('Complete: ' + str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('parent', 'packet', 'predictions', 'output'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--config', default='configs/temporal_warning.json')
    run(parser.parse_args())
