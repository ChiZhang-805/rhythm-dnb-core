"""Evaluate every existing longitudinal person once, without replacing the historical experiment."""

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import csv
from pathlib import Path
import shutil
import subprocess

from rhythm_dnb.measures.scaling import fit_scaler
from rhythm_dnb.provenance import file_hash, fingerprint
from rhythm_dnb.research.experiment import (choose_threshold, forecast_rows, policy_settings,
                                          score_cohort, validate_protocol)
from rhythm_dnb.research.experiment_data import load_packet, save_json, OBJECTIVE, TEXT
from rhythm_dnb.research.experiment_gpu import attach_predictions, implementation_id
from rhythm_dnb.research.evaluate import event_metrics, cluster_intervals
from rhythm_dnb.research.plots import plot_experiment_result
from rhythm_dnb.timebase import instant
from tools.export_dnbr import prepare_matrices
from tools.run_dnbr_warning import (read_json, verify_files, choose_modules, write_matrix,
                                   export_scores_input, read_scores, audit_components)


ROTATIONS = (('train', 'validation', 'test'), ('validation', 'test', 'train'),
             ('test', 'train', 'validation'))


def validate_people(data):
    # PSEUDOCODE: require isolated original cohorts and exclusion of every DNB person from text development.
    roles = {k: {r['participant_id'] for r in data[k]} for k in ('reference', 'train', 'validation', 'test')}
    if any(not v for v in roles.values()) or any(roles[a] & roles[b] for a in roles for b in roles if a < b):
        raise ValueError('Original people must be nonempty and disjoint across roles.')
    records = [r['record_id'] for k in roles for r in data[k]]
    if len(records) != len(set(records)) or len(data['reference']) != len(roles['reference']):
        raise ValueError('Repeated record or reference person.')
    if any(r['split'] != k for k in roles for r in data[k]):
        raise ValueError('Original split does not match its packet file.')
    people = {'rhythm:' + p for group in roles.values() for p in group}
    development = data['text-development']
    if not people <= set(development['manifest']['excluded_dnb_people']):
        raise ValueError('Text model exclusion does not cover the entire DNB cohort.')
    if people & {r.get('participant_id') for r in development['rows']}:
        raise ValueError('Text development contains DNB people.')
    return {k: sorted(v) for k, v in roles.items()}


def development_pairs(rows, protocol):
    # PSEUDOCODE: retain the original earliest/latest pre-event pairing rule on this fold's development people only.
    groups = defaultdict(list)
    for row in rows:
        groups[row['participant_id']].append(row)
    pairs = []
    for person, sequence in sorted(groups.items()):
        onsets = {r['outcome']['event_onset_at'] for r in sequence if r['outcome']['event_onset_at']}
        if len(onsets) > 1:
            raise ValueError('Conflicting first-event onsets.')
        if not onsets:
            continue
        limit = instant(next(iter(onsets))) - timedelta(hours=protocol['discovery_pre_event_lead_hours'])
        eligible = sorted((r for r in sequence if instant(r['issued_at']) <= limit),
                          key=lambda r: instant(r['issued_at']))
        if len(eligible) >= 2:
            pairs.append({'participant_id': person, 'stable_record': eligible[0]['record_id'],
                          'pre_event_record': eligible[-1]['record_id']})
    return pairs


def export_development(reference, rows, features, scaler, protocol, settings, binding, output):
    # PSEUDOCODE: create a new fold export while preserving all original packet files and outcome values.
    train = [{**r, 'split': 'train'} for r in rows]
    pairs = development_pairs(train, protocol)
    matrix, meta = prepare_matrices(reference, train, pairs, features, scaler)
    output.mkdir(parents=True)
    write_matrix(output / 'matrix.csv', matrix.T, features, [r['record_id'] for r in meta])
    with (output / 'samples.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(meta[0]))
        writer.writeheader()
        writer.writerows(meta)
    save_json(output / 'pairs.json', pairs)
    save_json(output / 'settings.json', settings)
    save_json(output / 'scaler.json', scaler)
    payload = {**binding, 'purpose': settings['purpose'], 'features': features,
               'stages': ['stable', 'pre_event'], 'people': len(pairs), 'observations': len(meta),
               'files': {p.name: file_hash(p) for p in output.iterdir()}}
    save_json(output / 'manifest.json', {**payload, 'id': fingerprint(payload)})


def policy_decisions(days, threshold):
    # PSEUDOCODE: preserve each frozen fold decision exactly for pooled replay; keep abstentions unknown.
    return tuple(replace(d, score=None if d.score is None or threshold is None else float(d.score > threshold))
                 for d in days)


def verify_parent(directory):
    # PSEUDOCODE: verify the preserved parent archive before reusing its method and reference results.
    manifest = read_json(directory / 'archive-manifest.json')
    if fingerprint(manifest['files']) != manifest['id']:
        raise ValueError('Parent archive manifest identity mismatch.')
    for name, digest in manifest['files'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_hash(path) != digest:
            raise ValueError('Parent experiment file mismatch: ' + name)


def execute_r(command, log):
    # PSEUDOCODE: save all R diagnostics and fail visibly rather than substitute another method.
    with log.open('x', encoding='utf-8') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)


def run_fold(data, manifest, inference, roles, settings, discovery, scaler, args, output):
    # PSEUDOCODE: develop modules -> freeze calibration -> evaluate untouched people under the same policy.
    features = list(OBJECTIVE) + list(TEXT)
    protocol = data['protocol']
    config = validate_protocol(protocol)
    exported, developed = output / 'development-input', output / 'development'
    binding = {'domain': manifest['domain'], 'packet_id': manifest['id'], 'inference_id': inference,
               'roles': dict(zip(('train', 'validation', 'test'), roles))}
    export_development(data['reference'], data[roles[0]], features, scaler, protocol, discovery, binding, exported)
    execute_r([args.rscript, str(Path(__file__).with_name('run_dnbr.R')), 'analyze',
               str(exported), str(developed), args.library], output / 'discovery-R.log')
    verify_files(developed, read_json(developed / 'hashes.json'))
    development = read_json(developed / 'result.json')
    if (development['input_id'] != read_json(exported / 'manifest.json')['id'] or
            development['upstream_revision'] != discovery['revision']):
        raise ValueError('Upstream development provenance differs.')
    with (developed / 'scores.csv').open(encoding='utf-8', newline='') as stream:
        modules = choose_modules(list(csv.DictReader(stream)), features, settings)
    save_json(output / 'selected-modules.json', modules)
    frozen, methods, pooled = {}, {}, {}
    for role, source in zip(('validation', 'test'), roles[1:]):
        rows = data[source]
        inputs, scored = output / (role + '-input'), output / (role + '-scores')
        reference, targets = export_scores_input(rows, data['reference'], features, scaler, modules,
                                                 settings, discovery['revision'], inputs)
        execute_r([args.rscript, str(Path(__file__).with_name('score_dnbr.R')), str(inputs),
                   str(scored), args.library], output / (role + '-R.log'))
        values = read_scores(scored / 'scores.csv', [r['record_id'] for r in rows])
        audit = audit_components(scored / 'components.csv', rows, reference, targets,
                                 features, modules, settings['epsilon'])
        save_json(output / (role + '-arithmetic-audit.json'), audit)
        baseline, _ = score_cohort(rows, features, reference, scaler, {'chosen': []}, config)
        scores = {'dnbr_sdnb': values, **{k: v for k, v in baseline.items() if k != 'dnb'}}
        for name, values in scores.items():
            days, events, periods, reasons = forecast_rows(rows, values, protocol)
            if role == 'validation':
                frozen[name] = choose_threshold(days, events, periods, protocol)
            else:
                threshold = frozen[name]['threshold']
                metrics = event_metrics(days, threshold, events=events, monitoring=periods, **policy_settings(protocol))
                methods[name] = {'status': frozen[name]['status'], 'threshold': threshold,
                                 'test': metrics, 'label_reasons': reasons}
                save_json(output / (name + '-forecasts.json'), [asdict(d) for d in days])
                pooled[name] = (policy_decisions(days, threshold), events, periods)
        if role == 'validation':
            save_json(output / 'frozen-policy.json', {'methods': frozen, 'test_used_for_calibration': False,
                       'frozen_at': datetime.now(timezone.utc).isoformat()})
            print('Calibration frozen: ' + output.name, flush=True)
    result = {'roles': binding['roles'], 'selected_modules': modules, 'methods': methods}
    save_json(output / 'result.json', result)
    print(output.name + ': ' + str(methods['dnbr_sdnb']['test']['risk_accuracy']), flush=True)
    return result, pooled


def write_summary(result, output):
    # PSEUDOCODE: put the actual warning accuracy first and distinguish people, days and simulation evidence.
    lines = ['# 全部现有纵向模拟样本的预警评估', '',
             '目标：依据当前及过去记录，判断未来 7 天是否出现节律失稳事件，至少提前 24 小时。',
             '使用既有模拟结局，未新造数据、未改标签；并非真实人群验证。',
             f'共 {result["test_people"]} 人，每人只进入一次测试；三组轮换重新发现指标组和校准阈值。', '',
             '| 方法 | 答对／可判断次数 | 正确率 | 平衡正确率 | 检出事件 | 误报提醒 | 提前天数中位数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, item in result['methods'].items():
        m = item['test']; c = m['risk_confusion']
        accuracy = '不可计算' if m['risk_accuracy'] is None else f'{m["risk_accuracy"]:.2%}'
        balanced = '不可计算' if m['risk_balanced_accuracy'] is None else f'{m["risk_balanced_accuracy"]:.2%}'
        lines.append(f'| {name} | {c["tp"] + c["tn"]}/{m["classified_days"]} | {accuracy} | {balanced} | '
                     f'{m["detected_events"]}/{m["events"]} | {m["false_alarms"]} | {m["median_lead_days"]} |')
    lines += ['', '逐日风险判断和实际通知分开统计：通知仍需连续两天超过阈值，并有七天冷却期。',
              '保留原来的 300 次测试；本次包含它，不是另加 900 个独立样本。',
              '数据曾参与历史开发，本次属于探索性交叉验证。95% 区间按人重采样，未包含重新训练的变动。',
              '不挑选最好一组；不把简单对照的成绩当成 DNB 成绩；候选组不等于通过原统计检验。']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(args):
    # PSEUDOCODE: freeze rotations before computation -> test every person once -> pool all outcomes and archive.
    output, parent = Path(args.output).resolve(), Path(args.parent).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output; preserve previous experiments.')
    verify_parent(parent)
    data, manifest = load_packet(args.packet)
    people = validate_people(data)
    inference = attach_predictions(data, manifest, args.predictions)
    previous = read_json(parent / 'result.json')
    if previous['packet_id'] != manifest['id'] or previous['inference_id'] != inference:
        raise ValueError('Inputs differ from the preserved parent experiment.')
    settings = read_json(parent / 'source/dnbr_warning.json')
    discovery = read_json(parent / 'development-input/settings.json')
    protocol = data['protocol']
    validate_protocol(protocol)
    features = list(OBJECTIVE) + list(TEXT)
    scaler = fit_scaler([[r['features'][k] for k in features] for r in data['reference']], features)
    output.mkdir(parents=True)
    source = output / 'source'; source.mkdir()
    for name in ('run_dnbr_cross_validation.py', 'run_dnbr_warning.py', 'export_dnbr.py', 'run_dnbr.R', 'score_dnbr.R'):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    plan = {'domain': manifest['domain'], 'packet_id': manifest['id'], 'inference_id': inference,
            'implementation_id': implementation_id(), 'people': people, 'rotations': ROTATIONS,
            'settings': settings, 'discovery': discovery, 'protocol': protocol,
            'historical_data_previously_inspected': True, 'selection_based_on_fold_results': False,
            'parent_result_sha256': file_hash(parent / 'result.json'),
            'source': {p.name: file_hash(p) for p in source.iterdir()},
            'created_at': datetime.now(timezone.utc).isoformat()}
    save_json(output / 'plan.json', {**plan, 'id': fingerprint(plan)})
    folds, pooled = [], defaultdict(lambda: [[], [], []])
    for number, roles in enumerate(ROTATIONS):
        fold, predictions = run_fold(data, manifest, inference, roles, settings, discovery, scaler,
                                      args, output / f'fold-{number + 1}')
        if number == 0:
            for name, old in previous['methods'].items():
                if fold['methods'][name]['test'] != old['test'] or fold['methods'][name]['threshold'] != old['threshold']:
                    raise ValueError('First rotation does not reproduce the preserved experiment.')
        folds.append(fold)
        for name, parts in predictions.items():
            for target, part in zip(pooled[name], parts):
                target.extend(part)
    methods = {}
    for name, (days, events, periods) in pooled.items():
        if len({p.participant_id for p in periods}) != sum(len(people[k]) for k in ('train', 'validation', 'test')):
            raise ValueError('Pooled tests repeat or omit people.')
        metrics = event_metrics(days, .5, events=events, monitoring=periods, **policy_settings(protocol))
        metrics.update(roc_auc=None, average_precision=None,
                       classification_definition='Each person uses their fold threshold; pooled binary decisions only.')
        methods[name] = {'test': metrics, 'intervals': {}}
        save_json(output / (name + '-pooled-decisions.json'), [asdict(d) for d in days])
    result = {'domain': manifest['domain'], 'plan_id': fingerprint(plan), 'primary_method': 'dnbr_sdnb',
              'primary_metric': protocol['primary_metric'], 'test_people': len(pooled['dnbr_sdnb'][2]),
              'folds': folds, 'methods': methods, 'clinical_accuracy_established': False,
              'interval_scope': 'Conditional on fitted fold models; participant resampling, not refitting uncertainty.',
              'pooled_auc_unavailable_reason': 'Thresholded decisions pooled; raw scales differ across folds.'}
    save_json(output / 'point-estimates.json', result)
    print('Pooled accuracy: ' + str(methods['dnbr_sdnb']['test']['risk_accuracy']), flush=True)
    for name, (days, events, periods) in pooled.items():
        print('Participant intervals: ' + name, flush=True)
        methods[name]['intervals'] = cluster_intervals(days, .5, repetitions=protocol['interval_repetitions'],
            seed=protocol['seed'], events=events, monitoring=periods, **policy_settings(protocol))
    write_summary(result, output)
    plot_experiment_result(result, output / 'comparison.png')
    save_json(output / 'result.json', result)
    files = {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}
    save_json(output / 'archive-manifest.json', {'files': files, 'id': fingerprint(files)})
    print('Complete: ' + str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('packet', 'predictions', 'parent', 'output', 'library'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--rscript', default='Rscript')
    run(parser.parse_args())
