"""Run the exploratory R DNB adapter through frozen calibration and person-level evaluation."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np

from rhythm_dnb.measures.scaling import fit_scaler, transform
from rhythm_dnb.provenance import file_hash, fingerprint
from rhythm_dnb.research.experiment import (choose_threshold, forecast_rows, policy_settings,
                                          score_cohort, validate_protocol)
from rhythm_dnb.research.experiment_data import load_packet, save_json, OBJECTIVE, TEXT
from rhythm_dnb.research.experiment_gpu import attach_predictions, implementation_id
from rhythm_dnb.research.evaluate import event_metrics, cluster_intervals
from rhythm_dnb.research.plots import plot_experiment_result


def read_json(path):
    # PSEUDOCODE: load explicit UTF-8 experiment artifacts.
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_files(directory, hashes):
    # PSEUDOCODE: enforce relative file identities before trusting an upstream result.
    for name, digest in hashes.items():
        if Path(name).name != name or file_hash(Path(directory) / name) != digest:
            raise ValueError('Artifact hash mismatch: ' + name)


def choose_modules(rows, features, settings):
    # PSEUDOCODE: rank eligible development modules -> deduplicate gene sets -> freeze the fixed budget.
    chosen, seen = [], set()
    eligible = [row for row in rows if row['stage'] == settings['selection_stage'] and
                row['usable'].upper() == 'TRUE']
    for row in sorted(eligible, key=lambda r: (-float(r['SCORE']), r['resource'])):
        module = tuple(sorted(row['genes'].split(',')))
        if (len(module) != len(set(module)) or not set(module) <= set(features) or
                not 2 <= len(module) < len(features) or not math.isfinite(float(row['SCORE']))):
            raise ValueError('Invalid upstream development module.')
        if module not in seen:
            chosen.append({'module': list(module), 'resource': row['resource'],
                           'development_score': float(row['SCORE'])})
            seen.add(module)
        if len(chosen) >= settings['max_modules']:
            break
    return chosen


def write_matrix(path, matrix, features, records):
    # PSEUDOCODE: export feature-by-record values without any future outcomes.
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['feature', *records])
        writer.writerows([name, *matrix[:, i]] for i, name in enumerate(features))


def export_scores_input(rows, reference_rows, features, scaler, modules, settings, revision, output):
    # PSEUDOCODE: export immutable reference plus current vectors; labels never enter the R scoring process.
    output.mkdir(parents=True)
    reference = transform([[r['features'][k] for k in features] for r in reference_rows], scaler)
    targets = transform([[r['features'][k] for k in features] for r in rows], scaler)
    if not np.isfinite(reference).all() or not np.isfinite(targets).all():
        raise ValueError('Incomplete measurements; refuse implicit filling.')
    write_matrix(output / 'reference.csv', reference, features, [r['record_id'] for r in reference_rows])
    write_matrix(output / 'targets.csv', targets, features, [r['record_id'] for r in rows])
    payload = {k: settings[k] for k in ('pair_convention', 'epsilon', 'sdnb_size_multiplier')}
    payload.update(modules=[m['module'] for m in modules], upstream_revision=revision,
                   files={name: file_hash(output / name) for name in ('reference.csv', 'targets.csv')})
    save_json(output / 'manifest.json', payload)
    return reference, targets


def read_scores(path, expected_ids):
    # PSEUDOCODE: require one finite nonnegative score or an explicit abstention per requested record.
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(expected_ids) or {r['record_id'] for r in rows} != set(expected_ids):
        raise ValueError('R score identities differ from the requested cohort.')
    result = {}
    for row in rows:
        value = float(row['score']) if row['score'] else None
        if row['status'] == 'ok':
            if value is None or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid successful R score.')
        elif row['status'] not in ('no_development_module', 'invalid_module') or value is not None:
            raise ValueError('Invalid R abstention status.')
        result[row['record_id']] = value
    return result


def audit_components(path, rows, reference, targets, features, modules, epsilon):
    # PSEUDOCODE: independently cross-check every R component before accepting any warning metric.
    from rhythm_dnb.dnb.single_sample import sdnb_components
    if not modules:
        return {'checked': 0, 'maximum_absolute_error': None}
    positions = {r['record_id']: i for i, r in enumerate(rows)}
    checked, maximum, seen = 0, 0., set()
    with Path(path).open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['record_id'], int(row['module_id']))
            if key in seen or key[0] not in positions or not 1 <= key[1] <= len(modules):
                raise ValueError('Duplicate or unexpected R component.')
            seen.add(key)
            expected = sdnb_components(targets[positions[key[0]]], reference, features,
                modules[key[1] - 1]['module'], epsilon=epsilon, pair_convention='paper_k_squared')
            if (row['valid'].upper() == 'TRUE') != expected['valid']:
                raise ValueError('R/Python validity mismatch.')
            for name in ('sED_in', 'sPCC_in', 'sPCC_out', 'score'):
                if expected[name.lower()] is not None:
                    value = float(row[name])
                    np.testing.assert_allclose(value, expected[name.lower()], rtol=1e-9, atol=1e-10)
                    maximum = max(maximum, abs(value - expected[name.lower()]))
            checked += 1
    if checked != len(rows) * len(modules):
        raise ValueError('Missing R component rows.')
    return {'checked': checked, 'maximum_absolute_error': maximum,
            'purpose': 'independent arithmetic audit; R scores used for evaluation'}


def summary(result, output):
    # PSEUDOCODE: describe the warning endpoint and distinguish candidates, daily errors and event alarms.
    lines = ['# R DNB 预警实验', '',
             '沿用已训练模型和文本分数；作者 DNBr 找指标组，R 适配器按单样本 DNB 公式计算个人分数。',
             '这是新的探索性方案，不改写旧实验。固定指标组属于项目适配，不是原论文逐样本发现的完整复现。',
             '只使用自建模拟队列；这批历史测试曾被查看，不是新的封存测试，也不能作为临床正确率。', '',
             '每天判断未来 7 天是否出现事件，至少提前 24 小时。实际提醒需连续两天超过阈值。', '',
             '| 方法 | 每日平衡正确率 | 事件检出 | 误报提醒 | 提前天数中位数 |',
             '|---|---:|---:|---:|---:|']
    for name, item in result['methods'].items():
        m = item['test']; accuracy = m['risk_balanced_accuracy']
        lines.append(f'| {name} | {accuracy:.1%} | {m["detected_events"]}/{m["events"]} | '
                     f'{m["false_alarms"]} | {m["median_lead_days"]} |' if accuracy is not None else
                     f'| {name} | 未形成可判断策略 | {m["detected_events"]}/{m["events"]} | '
                     f'{m["false_alarms"]} | 无 |')
    lines += ['', f'开发阶段保留 {len(result["selected_modules"])} 个候选组，未宣称通过原实验的统计检验。',
              '阈值只用校准人群确定。正确率按每日风险判断计算；漏掉几个人、发了几次提醒另行统计。',
              'result.json 包含按人重采样区间、混淆矩阵、覆盖率和每 30 天误报数；comparison.png 为结果图。',
              '模型能给出分数和预测，不等于已经证明节律临界转变机制或真实人群泛化。']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(args):
    # PSEUDOCODE: freeze method -> verify author discovery -> calibrate -> freeze threshold -> score/evaluate test.
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh experiment directory.')
    settings = read_json(args.config)
    if (settings['purpose'] != 'exploratory_dnbr_discovery_fixed_module_sdnb' or
            settings['selection_stage'] != 'pre_event' or type(settings['max_modules']) is not int or
            settings['max_modules'] < 1 or settings['aggregation'] != 'maximum_all_modules_required' or
            settings['pair_convention'] != 'paper_k_squared' or settings['sdnb_size_multiplier'] != 1 or
            not 0 < settings['epsilon'] < 1):
        raise ValueError('Unexpected exploratory settings.')
    development, exported = Path(args.development), Path(args.input)
    source_manifest = read_json(exported / 'manifest.json')
    if fingerprint({k: v for k, v in source_manifest.items() if k != 'id'}) != source_manifest['id']:
        raise ValueError('Development input identity mismatch.')
    verify_files(exported, source_manifest['files'])
    verify_files(development, read_json(development / 'hashes.json'))
    discovery_result = read_json(development / 'result.json')
    if discovery_result['input_id'] != source_manifest['id']:
        raise ValueError('Upstream result belongs to different development inputs.')
    data, manifest = load_packet(args.packet)
    inference = attach_predictions(data, manifest, args.predictions)
    if manifest['id'] != source_manifest['packet_id'] or inference != source_manifest['inference_id']:
        raise ValueError('Packet/predictions differ from R development.')
    protocol = data['protocol']; config = validate_protocol(protocol)
    if settings['max_modules'] != protocol['max_modules'] or settings['epsilon'] != protocol['epsilon']:
        raise ValueError('Changed frozen module-count budget or numerical guard.')
    features = list(OBJECTIVE) + list(TEXT)
    scaler = read_json(exported / 'scaler.json')
    if scaler['features'] != features:
        raise ValueError('Feature order changed.')
    roles = {k: {r['participant_id'] for r in data[k]} for k in ('reference', 'train', 'validation', 'test')}
    if any(roles[a] & roles[b] for a in roles for b in roles if a < b):
        raise ValueError('People overlap across roles.')
    if len(data['reference']) != len(roles['reference']) or len(roles['reference']) < config.reference_min_people:
        raise ValueError('Invalid reference people.')
    fitted = fit_scaler([[r['features'][k] for k in features] for r in data['reference']], features)
    for key in ('mean', 'scale'):
        np.testing.assert_allclose(fitted[key], scaler[key], atol=1e-12, rtol=1e-12)
    if fitted['clock_centers'].keys() != scaler['clock_centers'].keys():
        raise ValueError('Reference clock anchors differ from development.')
    for key, value in fitted['clock_centers'].items():
        np.testing.assert_allclose(value, scaler['clock_centers'][key], atol=1e-12, rtol=1e-12)
    output.mkdir(parents=True)
    source = output / 'source'; source.mkdir()
    for path in (Path(__file__), Path(__file__).with_name('score_dnbr.R'), Path(args.config)):
        shutil.copy2(path, source / path.name)
    shutil.copytree(development, output / 'development')
    shutil.copytree(exported, output / 'development-input')
    with (development / 'scores.csv').open(encoding='utf-8', newline='') as stream:
        modules = choose_modules(list(csv.DictReader(stream)), features, settings)
    plan = {'settings': settings, 'packet_id': manifest['id'], 'protocol_id': fingerprint(protocol),
            'inference_id': inference, 'implementation_id': implementation_id(),
            'created_at': datetime.now(timezone.utc).isoformat(), 'features': features,
            'selected_modules': modules, 'upstream_revision': discovery_result['upstream_revision'],
            'development_result_sha256': file_hash(development / 'result.json'),
            'source': {p.name: file_hash(p) for p in source.iterdir()},
            'historical_test_previously_inspected': True, 'test_used_to_select_this_plan': False}
    save_json(output / 'plan.json', {**plan, 'id': fingerprint(plan)})
    save_json(output / 'protocol.json', protocol)
    scores, audits, frozen = {}, {}, {}
    for role in ('validation', 'test'):
        print('Scoring ' + role, flush=True)
        inputs, scored = output / (role + '-input'), output / (role + '-scores')
        reference, targets = export_scores_input(data[role], data['reference'], features, scaler, modules,
                                                 settings, plan['upstream_revision'], inputs)
        with (output / (role + '-R.log')).open('x', encoding='utf-8') as log:
            subprocess.run([args.rscript, str(source / 'score_dnbr.R'), str(inputs), str(scored), args.library],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        primary = read_scores(scored / 'scores.csv', [r['record_id'] for r in data[role]])
        audits[role] = audit_components(scored / 'components.csv', data[role], reference, targets,
                                       features, modules, settings['epsilon'])
        baseline, _ = score_cohort(data[role], features, reference, scaler, {'chosen': []}, config)
        scores[role] = {'dnbr_sdnb': primary, **{k: v for k, v in baseline.items() if k != 'dnb'}}
        if role == 'validation':
            for method, values in scores[role].items():
                days, events, periods, _ = forecast_rows(data[role], values, protocol)
                frozen[method] = choose_threshold(days, events, periods, protocol)
            save_json(output / 'frozen-policy.json', {'plan_id': fingerprint(plan), 'methods': frozen,
                      'frozen_at': datetime.now(timezone.utc).isoformat(), 'test_used_for_calibration': False})
            print('Calibration frozen before test scoring.', flush=True)
    save_json(output / 'arithmetic-audit.json', audits)
    methods = {}
    for method, values in scores['test'].items():
        print('Evaluating ' + method, flush=True)
        days, events, periods, reasons = forecast_rows(data['test'], values, protocol)
        threshold = frozen[method]['threshold']
        metrics = event_metrics(days, threshold, events=events, monitoring=periods, **policy_settings(protocol))
        intervals = cluster_intervals(days, threshold, repetitions=protocol['interval_repetitions'],
            seed=protocol['seed'], events=events, monitoring=periods, **policy_settings(protocol))
        methods[method] = {'status': frozen[method]['status'], 'threshold': threshold, 'test': metrics,
                           'intervals': intervals, 'label_reasons': reasons}
        save_json(output / (method + '-forecasts.json'), [asdict(d) for d in days])
    result = {'domain': manifest['domain'], 'packet_id': manifest['id'], 'plan_id': fingerprint(plan),
              'inference_id': inference, 'primary_method': 'dnbr_sdnb', 'primary_metric': protocol['primary_metric'],
              'selected_modules': modules, 'methods': methods, 'clinical_accuracy_established': False,
              'interpretation': 'New exploratory R adapter on an already-inspected authored cohort. '
                                'Candidate ranking is not statistical confirmation of a critical transition.'}
    summary(result, output)
    plot_experiment_result(result, output / 'comparison.png')
    save_json(output / 'result.json', result)
    files = {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}
    save_json(output / 'archive-manifest.json', {'files': files, 'id': fingerprint(files)})
    print(json.dumps({'status': 'complete', 'result': str(output / 'result.json')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('packet', 'predictions', 'input', 'development', 'config', 'output', 'library'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--rscript', default='Rscript')
    run(parser.parse_args())
