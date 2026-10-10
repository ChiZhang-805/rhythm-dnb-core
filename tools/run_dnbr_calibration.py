"""Score independent development text on GPU, fit monotone corrections, and replay the same DNB experiment."""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

from rhythm_dnb.contracts import EvaluationDay
from rhythm_dnb.measures.scaling import fit_scaler
from rhythm_dnb.provenance import canonical_json, file_hash, fingerprint
from rhythm_dnb.research.experiment import policy_settings
from rhythm_dnb.research.experiment_data import load_packet, save_json, OBJECTIVE, TEXT
from rhythm_dnb.research.experiment_gpu import attach_predictions, implementation_id, verify_new_checkpoint
from rhythm_dnb.research.evaluate import event_metrics
from rhythm_dnb.timebase import instant
from tools.run_dnbr_ablation import paired_accuracy_interval
from tools.run_dnbr_cross_validation import ROTATIONS, run_fold, validate_people, verify_parent
from tools.run_dnbr_warning import read_json
from tools.text_score_calibration import development_rows, select_curves, apply_curve, STRENGTHS, SEED


SOURCES = ('run_dnbr_calibration.py', 'text_score_calibration.py', 'personal_warning.py',
           'run_dnbr_cross_validation.py', 'run_dnbr_ablation.py', 'run_dnbr_warning.py',
           'export_dnbr.py', 'run_dnbr.R', 'score_dnbr.R')


def read_decisions(path):
    # PSEUDOCODE: restore explicit instants so independent saved forecasts align with in-memory replay dates.
    return [EvaluationDay(**{**row, **{k: instant(row[k]) if row.get(k) is not None else None
                                       for k in ('issued_at', 'onset', 'label_available_at')}}) for row in read_json(path)]


def verify_plan(saved, plan):
    # PSEUDOCODE: compare canonical content so JSON lists match tuples without accepting changed settings or stale hashes.
    identity = fingerprint(plan)
    content = {k: v for k, v in saved.items() if k != 'id'}
    if saved.get('id') != identity or fingerprint(content) != identity:
        raise ValueError('Saved calibration plan differs; do not resume incompatible work.')


def freeze(args):
    # PSEUDOCODE: freeze data exclusions, correction rules and code before any new GPU predictions or test results.
    data, manifest = load_packet(args.packet)
    people = validate_people(data)
    model, identity = verify_new_checkpoint(args.checkpoint, data, manifest)
    inference = attach_predictions(data, manifest, args.predictions)
    raw = read_json(args.predictions)
    if identity != raw['model_id']:
        raise ValueError('Calibration checkpoint differs from cached cohort predictions.')
    rows = development_rows(data)
    plan = {'packet_id': manifest['id'], 'inference_id': inference, 'model_id': identity,
            'implementation_id': implementation_id(), 'development_rows_id': fingerprint(rows),
            'development_counts': {str(k): sum(r['calibration_fold'] == k for r in rows) for k in (0, 1)},
            'text_metrics': list(TEXT), 'strengths': list(STRENGTHS), 'seed': SEED,
            'fit': 'person-weighted bounded isotonic regression, cross-fit by connected people/templates',
            'selection': 'lowest text development cross-fitted MAE; tie favors less correction; refit full development',
            'strength_rationale': 'Identity through 75% correction in fixed quarter increments; retain at least 25% '
                                  'of original slope to avoid flat reference variance. Experimental grid, not an optimum claim.',
            'precision': 'fp32', 'batch_size': 1, 'people': people, 'protocol': data['protocol'],
            'rotations': ROTATIONS, 'domain': manifest['domain'], 'test_used_for_selection': False,
            'development_used_for_original_checkpoint_selection': True, 'historical_dnb_test_previously_inspected': True,
            'evidence_calibration_changed': False, 'source': {name: file_hash(Path(__file__).with_name(name)) for name in SOURCES}}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'plan.json'
    if path.exists():
        verify_plan(read_json(path), plan)
    else:
        save_json(path, {**plan, 'id': fingerprint(plan)})
        source = output/'source'; source.mkdir()
        for name in SOURCES:
            shutil.copy2(Path(__file__).with_name(name), source/name)
    return data, manifest, rows, plan


def score(args):
    # PSEUDOCODE: predict only excluded-development texts; bind each atomic cache entry to model and execution profile.
    from rhythm_dnb.text.predict import TextPredictor
    import torch
    data, manifest, rows, plan = freeze(args)
    output = Path(args.output); cache = output/'cache'; cache.mkdir(exist_ok=True)
    tasks = {fingerprint([r['category'], r['text']]): {'category': r['category'], 'text': r['text']} for r in rows}
    if (output/'predictions.json').exists():
        saved = read_json(output/'predictions.json')
        if saved['plan_id'] != fingerprint(plan) or saved['id'] != fingerprint({k: v for k, v in saved.items() if k != 'id'}):
            raise ValueError('Completed text prediction identity mismatch.')
        return
    torch.set_num_threads(4)
    predictor = TextPredictor(args.checkpoint, base_path=args.base, device='cuda', allow_experimental=True, precision='fp32')
    predictions, profile = {}, None
    started = time.monotonic()
    for number, (key, task) in enumerate(sorted(tasks.items()), 1):
        path = cache/(key+'.json')
        prediction = predictor.predict(**task) if number == 1 or not path.exists() else None
        if prediction is not None:
            current = prediction['inference_profile']
            if profile is not None and profile != current:
                raise ValueError('GPU inference profile changed during calibration.')
            profile = current
        if path.exists():
            item = read_json(path)
            if item['task'] != task or item['model_id'] != plan['model_id'] or item['profile'] != profile:
                raise ValueError('Calibration prediction cache mismatch.')
        else:
            item = {'task': task, 'model_id': plan['model_id'], 'profile': profile, 'estimates': prediction['estimates']}
            partial = path.with_suffix('.partial')
            partial.write_text(canonical_json(item), encoding='utf-8'); partial.replace(path)
        predictions[key] = item['estimates']
        if number % 100 == 0 or number == len(tasks):
            progress = {'completed': number, 'total': len(tasks), 'elapsed_seconds': time.monotonic()-started,
                        'updated_at': datetime.now(timezone.utc).isoformat()}
            (output/'progress.json').write_text(canonical_json(progress), encoding='utf-8')
            print(canonical_json(progress), flush=True)
    raw = read_json(args.predictions)
    if profile != raw['profile']:
        raise ValueError('Development and DNB prediction profiles differ.')
    result = {'plan_id': fingerprint(plan), 'model_id': plan['model_id'], 'profile': profile, 'predictions': predictions}
    save_json(output/'predictions.json', {**result, 'id': fingerprint(result)})


def evaluate(args):
    # PSEUDOCODE: fit from development scores -> freeze curves -> transform measurements only -> refit/calibrate/test DNB.
    data, manifest, rows, plan = freeze(args)
    root, parent = Path(args.output), Path(args.parent).resolve()
    verify_parent(parent)
    previous, old_plan = read_json(parent/'result.json'), read_json(parent/'plan.json')
    if old_plan['packet_id'] != plan['packet_id'] or old_plan['inference_id'] != plan['inference_id']:
        raise ValueError('Parent differs from frozen inputs.')
    predictions = read_json(root/'predictions.json')
    if predictions['plan_id'] != fingerprint(plan) or predictions['id'] != fingerprint({k:v for k,v in predictions.items() if k != 'id'}):
        raise ValueError('Development predictions differ from plan.')
    curves = select_curves(rows, predictions['predictions'])
    save_json(root/'calibration.json', {'curves': curves, 'plan_id': fingerprint(plan),
                                       'dnb_outcomes_used': False, 'independent_text_test': False})
    for role in ('reference', 'train', 'validation', 'test'):
        for row in data[role]:
            for name, adjustment in curves.items():
                row['features'][name] = float(apply_curve(adjustment['curve'], [row['features'][name]], adjustment['strength'])[0])
    output = root/'evaluation'; output.mkdir()
    features, protocol = list(OBJECTIVE)+list(TEXT), data['protocol']
    scaler = fit_scaler([[r['features'][k] for k in features] for r in data['reference']], features)
    save_json(output/'scaler.json', scaler)
    binding = fingerprint({'raw_inference': plan['inference_id'], 'calibration': file_hash(root/'calibration.json')})
    folds, pooled = [], defaultdict(lambda: [[], [], []])
    for number, roles in enumerate(ROTATIONS, 1):
        fold, parts = run_fold(data, manifest, binding, roles, old_plan['settings'], old_plan['discovery'], scaler,
                               args, output/f'fold-{number}')
        folds.append(fold)
        for method, triple in parts.items():
            for destination, values in zip(pooled[method], triple):
                destination.extend(values)
    methods = {}
    for method, (days, events, periods) in pooled.items():
        expected = set().union(*(set(plan['people'][k]) for k in ('train', 'validation', 'test')))
        if len(periods) != len(expected) or {p.participant_id for p in periods} != expected:
            raise ValueError('Calibration replay repeats or omits people.')
        metrics = event_metrics(days, .5, events=events, monitoring=periods, **policy_settings(protocol))
        metrics.update(roc_auc=None, average_precision=None)
        original = read_decisions(parent/(method+'-pooled-decisions.json'))
        interval = paired_accuracy_interval(days, original, protocol['seed'], protocol['interval_repetitions'])
        methods[method] = {'test': metrics, 'original': previous['methods'][method]['test'], 'paired_difference': interval}
        save_json(output/(method+'-pooled-decisions.json'), [asdict(d) for d in days])
    result = {'plan_id': fingerprint(plan), 'calibration_id': binding, 'methods': methods, 'folds': folds,
              'clinical_accuracy_established': False, 'test_used_for_calibrator_selection': False}
    save_json(output/'result.json', result)
    lines = ['# 文本分数校准后的预警实验', '', '校准仅使用与 DNB 人物隔离的开发文本；未改变事件、划分或报警政策。',
             '开发文本曾用于原模型选择，校准误差为开发比较；DNB 仍是已查看构造队列的探索结果。', '',
             '| 方法 | 原正确率 | 校准后正确率 | 漏报事件 / 90 | 误报提醒 | 提前天数中位数 | 覆盖率 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, title in (('dnbr_sdnb','DNB'),('reference_deviation','平均偏离程度')):
        item=methods[name]; m=item['test']
        rate = f'{m["risk_accuracy"]:.2%}' if m['risk_accuracy'] is not None else '不可计算'
        lines.append(f'| {title} | {item["original"]["risk_accuracy"]:.2%} | {rate} | '
                     f'{m["events"]-m["detected_events"]} | {m["false_alarms"]} | {m["median_lead_days"]} | '
                     f'{m["classification_coverage"]:.1%} |')
    lines += ['', '当前只学习程度分数的单调修正；没有改变语境依据接收门槛，也不能修复文本说错人物或时间。',
              '保留原模型及原始推理，校准另存；不按本次最高测试成绩自动替换正式方法。']
    (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files = {p.relative_to(root).as_posix(): file_hash(p) for p in root.rglob('*') if p.is_file()}
    save_json(root/'archive-manifest.json', {'files': files, 'id': fingerprint(files)})
    print('Complete: '+str(output),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('score','evaluate'))
    for name in ('packet','checkpoint','predictions','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--base')
    parser.add_argument('--parent')
    parser.add_argument('--library')
    parser.add_argument('--rscript',default='Rscript')
    args=parser.parse_args()
    (score if args.command=='score' else evaluate)(args)
