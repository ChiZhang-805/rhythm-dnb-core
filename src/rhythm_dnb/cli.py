"""Thin CLI over workflows. Every input and output path is explicit."""

import argparse
from datetime import date, datetime
import json
import os
from pathlib import Path
from .provenance import canonical_json


def _read(path):
    # PSEUDOCODE: decode a caller-supplied JSON artifact; do not resolve paths from the working directory.
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _cases(payloads):
    # PSEUDOCODE: decode offline labels separately from the forecast-only requests they accompany.
    from .contracts import parse_request, OutcomeLabel
    from .workflows.develop import LabeledCase
    result = []
    for payload in payloads:
        label = dict(payload['label'])
        if label.get('onset'):
            label['onset'] = datetime.fromisoformat(label['onset'])
        result.append(LabeledCase(parse_request(payload['request']), OutcomeLabel(**label),
                                  datetime.fromisoformat(payload['label_available_at']), payload.get('outcome_protocol_id')))
    return result


def _events(payloads):
    # PSEUDOCODE: reconstruct the independent event registry, preserving actual confirmation times.
    from .contracts import OutcomeEvent
    if payloads is None:
        return None
    return [OutcomeEvent(r['participant_id'], datetime.fromisoformat(r['onset']),
                         datetime.fromisoformat(r['confirmed_at']), r.get('definition', 'sleep-eating-activity')) for r in payloads]


def _monitoring(payloads):
    # PSEUDOCODE: decode the independent local-noon monitoring schedule, including missing forecast days.
    from .contracts import MonitoringPeriod
    if payloads is None:
        return None
    return [MonitoringPeriod(r['participant_id'], date.fromisoformat(r['first_issue_day']),
                             date.fromisoformat(r['last_issue_day']), r['timezone']) for r in payloads]


def main(argv=None):
    # PSEUDOCODE: parse an explicit operation -> call its workflow -> save/print a strict JSON receipt.
    parser = argparse.ArgumentParser(prog='rhythm-dnb', description='Behavioral rhythm DNB research core')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('hardware', help='Inspect GPU devices visible on this host')
    audit = sub.add_parser('audit'); audit.add_argument('--database', required=True); audit.add_argument('--output', required=True)
    score = sub.add_parser('score'); score.add_argument('--bundle', required=True); score.add_argument('--request', required=True); score.add_argument('--state', help='Previous complete score output or its state object'); score.add_argument('--output', required=True); score.add_argument('--method', choices=('single_sample', 'rolling'), default='single_sample')
    develop = sub.add_parser('develop'); develop.add_argument('--input', required=True); develop.add_argument('--study', required=True); develop.add_argument('--output-dir', required=True)
    validate = sub.add_parser('validate'); validate.add_argument('--bundle', required=True); validate.add_argument('--cases', required=True); validate.add_argument('--as-of', required=True); validate.add_argument('--output', required=True); validate.add_argument('--bootstrap', type=int, help='Defaults to the frozen study setting; 0 disables intervals')
    validate.add_argument('--events', required=True)
    validate.add_argument('--monitoring', required=True)
    prepare = sub.add_parser('prepare'); prepare.add_argument('--input', required=True); prepare.add_argument('--output', required=True)
    quantify = sub.add_parser('quantify'); quantify.add_argument('--input', required=True); quantify.add_argument('--output', required=True)
    acquire = sub.add_parser('download-text'); acquire.add_argument('--config', required=True); acquire.add_argument('--output-dir', required=True); acquire.add_argument('--cache-dir', required=True)
    plot = sub.add_parser('plot-text'); plot.add_argument('--result', required=True); plot.add_argument('--predictions', required=True); plot.add_argument('--output-dir', required=True)
    criteria = sub.add_parser('fit-endpoint'); criteria.add_argument('--input', required=True); criteria.add_argument('--study', required=True); criteria.add_argument('--cutoff', required=True); criteria.add_argument('--output', required=True)
    endpoint = sub.add_parser('endpoints'); endpoint.add_argument('--input', required=True); endpoint.add_argument('--study', required=True); endpoint.add_argument('--as-of', required=True); endpoint.add_argument('--output', required=True)
    label = sub.add_parser('label', help='Generate an offline forecast target from a frozen endpoint timeline')
    label.add_argument('--timeline', required=True); label.add_argument('--request', required=True); label.add_argument('--study', required=True)
    label.add_argument('--followup-end', required=True); label.add_argument('--as-of', required=True); label.add_argument('--output', required=True)
    check = sub.add_parser('check-text', help='Validate training inputs without loading model weights onto a device')
    check.add_argument('--corpus', required=True); check.add_argument('--base', required=True); check.add_argument('--config', required=True)
    check.add_argument('--development-only', action='store_true'); check.add_argument('--output', required=True)
    train = sub.add_parser('train-text'); train.add_argument('--corpus', required=True); train.add_argument('--base', required=True); train.add_argument('--config', required=True); train.add_argument('--output-dir', required=True)
    train.add_argument('--development-only', action='store_true', help='Tune using a train/validation-only corpus; never calibrate or test')
    train.add_argument('--save-resume-state', action='store_true', help='Save full optimizer/random state at each completed epoch')
    train.add_argument('--resume-state', help='Verified epoch receipt from a previous run; use a new output directory')
    text = sub.add_parser('score-text'); text.add_argument('--checkpoint', required=True); text.add_argument('--category', required=True); text.add_argument('--text', required=True); text.add_argument('--output', required=True)
    text.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    text.add_argument('--base', help='Exact local base weights required for Qwen adapters')
    export_experiment = sub.add_parser('export-text-experiment', help='Freeze authored/model-scored references without relabeling their origin')
    export_experiment.add_argument('--database', required=True); export_experiment.add_argument('--output-dir', required=True)
    expand = sub.add_parser('expand-text-experiment', help='Freeze rhythm narratives and new participant holdouts for continuation')
    expand.add_argument('--database', required=True); expand.add_argument('--legacy-development', required=True)
    expand.add_argument('--legacy-test', required=True); expand.add_argument('--checkpoint', required=True)
    expand.add_argument('--output-dir', required=True); expand.add_argument('--seed', type=int, default=20261006)
    expand.add_argument('--supplement', help='Reviewed authored complex training examples in JSON Lines format')
    review = sub.add_parser('review-text-experiment', help='Screen doubtful weak references in an existing frozen export')
    review.add_argument('--corpus-dir', required=True); review.add_argument('--output-dir', required=True)
    refresh = sub.add_parser('refresh-text-source', help='Read refreshed authored references and preserve source times without reopening an old test')
    refresh.add_argument('--corpus-dir', required=True); refresh.add_argument('--database', required=True); refresh.add_argument('--output-dir', required=True)
    coverage = sub.add_parser('audit-text', help='Report reference coverage and evidence states for every metric')
    coverage.add_argument('--corpus', required=True); coverage.add_argument('--output', required=True)
    blind = sub.add_parser('prepare-text-review', help='Prepare two blinded independent annotation packets')
    blind.add_argument('--corpus', required=True); blind.add_argument('--output-dir', required=True)
    blind.add_argument('--per-metric', type=int, required=True); blind.add_argument('--seed', type=int, required=True)
    adjudicate = sub.add_parser('compare-text-reviews', help='Compare completed independent reviews without averaging or writing labels')
    adjudicate.add_argument('--coordinator', required=True); adjudicate.add_argument('--first', required=True); adjudicate.add_argument('--second', required=True)
    adjudicate.add_argument('--tolerance', type=float, required=True); adjudicate.add_argument('--output', required=True)
    seal = sub.add_parser('seal-text-experiment', help='Seal all-metric train/validation/test data excluding previous holdout exposure')
    seal.add_argument('--corpus', required=True); seal.add_argument('--exposed', required=True); seal.add_argument('--output-dir', required=True)
    seal.add_argument('--initial-checkpoint-id')
    comparison = sub.add_parser('compare-warnings', help='Calibrate and test frozen methods on identical people, outcomes and calendars')
    comparison.add_argument('--input', required=True); comparison.add_argument('--study', required=True); comparison.add_argument('--output', required=True)
    sensitivity = sub.add_parser('plan-window-sensitivity', help='Prespecify predictor windows without changing the target event')
    sensitivity.add_argument('--study', required=True); sensitivity.add_argument('--candidates', required=True); sensitivity.add_argument('--output', required=True)
    train_experiment = sub.add_parser('train-text-experiment', help='Train only on the experimental development partition')
    train_experiment.add_argument('--corpus', required=True); train_experiment.add_argument('--base', required=True)
    train_experiment.add_argument('--config', required=True); train_experiment.add_argument('--output-dir', required=True)
    train_experiment.add_argument('--save-resume-state', action='store_true')
    train_experiment.add_argument('--resume-state')
    train_experiment.add_argument('--initialize-from', help='Start a new data stage from an experimental checkpoint')
    text_study = sub.add_parser('study-text', help='Run a prespecified repeated experiment without reading test data')
    text_study.add_argument('--corpus', required=True); text_study.add_argument('--protocol', required=True)
    text_study.add_argument('--base', required=True); text_study.add_argument('--output-dir', required=True); text_study.add_argument('--initialize-from')
    evaluate_experiment = sub.add_parser('evaluate-text-experiment', help='Evaluate a chosen experimental model on its sealed test set')
    evaluate_experiment.add_argument('--corpus', required=True); evaluate_experiment.add_argument('--base', required=True)
    evaluate_experiment.add_argument('--checkpoint', required=True); evaluate_experiment.add_argument('--output-dir', required=True)
    evaluate_experiment.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    evaluate_experiment.add_argument('--regression-only', action='store_true', help='Explicitly mark a previously inspected test as a development check')
    score_experiment = sub.add_parser('score-text-experiment', help='Return experimental reference estimates; not a calibrated DNB input')
    score_experiment.add_argument('--checkpoint', required=True); score_experiment.add_argument('--base', required=True)
    score_experiment.add_argument('--category', required=True); score_experiment.add_argument('--text', required=True)
    score_experiment.add_argument('--output', required=True)
    score_experiment.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args(argv)
    if args.command == 'hardware':
        from .text.runtime import hardware_report
        print(canonical_json(hardware_report())); return 0
    from .research.report import save_report
    if args.command == 'audit':
        from .research.report import audit_legacy_store
        result = audit_legacy_store(args.database)
    elif args.command == 'score':
        from .api import RhythmPredictor
        from .contracts import parse_request, parse_alarm_state
        state = parse_alarm_state(_read(args.state) if args.state else {})
        result = RhythmPredictor.from_bundle(args.bundle).predict(parse_request(_read(args.request)), state, method=args.method)
    elif args.command in ('prepare', 'quantify'):
        from .contracts import parse_observation
        from .workflows.prepare import prepare_day, quantify_day
        payload = _read(args.input)
        payload['observations'] = [parse_observation(r) for r in payload['observations']]
        payload['day'] = date.fromisoformat(payload['day']); payload['issued_at'] = datetime.fromisoformat(payload['issued_at'])
        checkpoint = payload.pop('text_checkpoint', None)
        text_base = payload.pop('text_base', None)
        text_device = payload.pop('text_device', 'auto')
        if checkpoint is not None:
            from .text.predict import TextPredictor
            payload['text_predictor'] = TextPredictor(checkpoint, base_path=text_base, device=text_device)
        result = (prepare_day if args.command == 'prepare' else quantify_day)(**payload)
    elif args.command == 'fit-endpoint':
        from .config import load_study
        from .workflows.endpoints import fit_endpoint_criteria
        result = fit_endpoint_criteria(_read(args.input), load_study(args.study), args.cutoff)
    elif args.command == 'endpoints':
        from .config import load_study
        from .contracts import Provenance
        from .workflows.endpoints import EndpointDay, build_endpoint_timeline
        payload = _read(args.input)
        rows = [EndpointDay(r['participant_id'], date.fromisoformat(r['day']), r['timezone'], r['values'],
                 datetime.fromisoformat(r['available_at']), Provenance(**r['provenance'])) for r in payload['days']]
        result = build_endpoint_timeline(rows, date.fromisoformat(payload['baseline_start']), payload['criteria'],
                                         load_study(args.study), as_of=args.as_of)
    elif args.command == 'label':
        from .config import load_study
        from .contracts import parse_request
        from .workflows.endpoints import parse_timeline, label_timeline
        timeline = parse_timeline(_read(args.timeline))
        request = parse_request(_read(args.request))
        if request.participant_id != timeline['participant_id'] or request.timezone != timeline['timezone']:
            raise ValueError('Forecast request and endpoint timeline must use the same person and timezone.')
        result = {'request': request, 'label': label_timeline(request.issued_at, timeline, load_study(args.study),
                  followup_end=args.followup_end, as_of=args.as_of),
                  'label_available_at': datetime.fromisoformat(args.as_of), 'outcome_protocol_id': timeline['protocol_id']}
    elif args.command == 'develop':
        from .config import load_study
        from .contracts import parse_panel
        from .dnb.reference import ReferenceCandidate
        from .research.discover import DiscoveryPair
        from .workflows.develop import develop as fit
        from .bundles import save_bundle
        payload = _read(args.input)
        references = [ReferenceCandidate(parse_panel(r['panel']), r['stable'], r['stability_evidence_id'], datetime.fromisoformat(r['stability_available_at']), r.get('baseline', True)) for r in payload['reference']]
        pairs = [DiscoveryPair(r['participant_id'], tuple(r['stable']), tuple(r['pre_event']), datetime.fromisoformat(r['evidence_available_at']), r['evidence_id'],
                 parse_panel(r['stable_panel']) if r.get('stable_panel') else None,
                 parse_panel(r['pre_event_panel']) if r.get('pre_event_panel') else None,
                 datetime.fromisoformat(r['event_onset']) if r.get('event_onset') else None,
                 r.get('outcome_protocol_id')) for r in payload['pairs']]
        bundle = fit(references, pairs, _cases(payload['calibration']), load_study(args.study),
            **{k: payload[k] for k in ('reference_cutoff', 'discovery_cutoff', 'calibration_cutoff')}, text_model_id=payload.get('text_model_id'),
            text_checkpoint=payload.get('text_checkpoint'),
            calibration_events=_events(payload.get('calibration_events')),
            calibration_monitoring=_monitoring(payload.get('calibration_monitoring')))
        print(canonical_json({'bundle': save_bundle(bundle, args.output_dir), 'status': bundle['calibration']['status']})); return 0
    elif args.command == 'validate':
        from .bundles import load_bundle
        from .workflows.validate import validate as evaluate
        result = evaluate(load_bundle(args.bundle), _cases(_read(args.cases)), bootstrap_repetitions=args.bootstrap, evaluation_as_of=args.as_of,
                          events=_events(_read(args.events)) if args.events else None,
                          monitoring=_monitoring(_read(args.monitoring)) if args.monitoring else None)
    elif args.command == 'download-text':
        from .text.weights import acquire_base
        print(canonical_json(acquire_base(_read(args.config), args.output_dir, args.cache_dir))); return 0
    elif args.command == 'plot-text':
        from .text.plots import plot_evaluation
        print(canonical_json(plot_evaluation(_read(args.result), _read(args.predictions), args.output_dir))); return 0
    elif args.command == 'check-text':
        from .text.preflight import check_training_inputs
        result = check_training_inputs(_read(args.corpus), args.base, _read(args.config), development_only=args.development_only)
    elif args.command == 'train-text':
        from .text.train import train as fit_text
        result = fit_text(_read(args.corpus), args.base, args.output_dir, _read(args.config), development_only=args.development_only,
                          save_resume_state=args.save_resume_state, resume_state=args.resume_state)
        if int(os.environ.get('RANK', '0')) == 0:
            print(canonical_json({'checkpoint': result['best_checkpoint'], 'test': result['test']}))
        return 0
    elif args.command == 'score-text':
        from .text.predict import TextPredictor
        result = TextPredictor(args.checkpoint, device=args.device, base_path=args.base).predict(args.category, args.text)
    elif args.command == 'export-text-experiment':
        from .text.experiment import export_corpus
        print(canonical_json(export_corpus(args.database, args.output_dir))); return 0
    elif args.command == 'expand-text-experiment':
        from .text.expansion import export_expansion
        result = export_expansion(args.database, args.legacy_development, args.legacy_test, args.checkpoint,
                                  args.output_dir, seed=args.seed, supplement=args.supplement)
        print(canonical_json({'counts': result['counts'], 'output_dir': args.output_dir})); return 0
    elif args.command == 'review-text-experiment':
        from .text.expansion import refine_frozen_corpus
        print(canonical_json(refine_frozen_corpus(args.corpus_dir,args.output_dir))); return 0
    elif args.command == 'refresh-text-source':
        from .text.expansion import refresh_source_corpus
        print(canonical_json(refresh_source_corpus(args.corpus_dir, args.database, args.output_dir))); return 0
    elif args.command == 'audit-text':
        from .text.review import coverage_report
        result = coverage_report(_read(args.corpus)['rows'])
    elif args.command == 'prepare-text-review':
        from .text.review import export_blind_review
        print(canonical_json(export_blind_review(_read(args.corpus)['rows'], args.output_dir, per_metric=args.per_metric, seed=args.seed))); return 0
    elif args.command == 'compare-text-reviews':
        from .text.review import compare_blind_review
        result = compare_blind_review(_read(args.coordinator), _read(args.first), _read(args.second), tolerance=args.tolerance)
    elif args.command == 'seal-text-experiment':
        from .text.experiment import seal_experiment
        print(canonical_json(seal_experiment(_read(args.corpus)['rows'], _read(args.exposed)['rows'], args.output_dir,
            initial_checkpoint_id=args.initial_checkpoint_id))); return 0
    elif args.command == 'compare-warnings':
        from .config import load_study
        from .contracts import EvaluationDay
        from .research.comparison import compare_methods
        payload = _read(args.input)
        methods = {}
        for name, partitions in payload['methods'].items():
            methods[name] = {}
            for role, records in partitions.items():
                decoded = []
                for record in records:
                    record = dict(record)
                    for key in ('issued_at', 'onset', 'label_available_at'):
                        if record.get(key) is not None:
                            record[key] = datetime.fromisoformat(record[key])
                    decoded.append(EvaluationDay(**record))
                methods[name][role] = decoded
        result = compare_methods(methods, calibration_events=_events(payload['calibration_events']), test_events=_events(payload['test_events']),
            calibration_monitoring=_monitoring(payload['calibration_monitoring']), test_monitoring=_monitoring(payload['test_monitoring']),
            calibration_cutoff=payload['calibration_cutoff'], evaluation_as_of=payload['evaluation_as_of'],
            config=load_study(args.study), fitted_people=payload['fitted_people'], bootstrap_repetitions=payload.get('bootstrap_repetitions', 0))
    elif args.command == 'plan-window-sensitivity':
        from .config import load_study
        from .research.comparison import window_sensitivity
        result = window_sensitivity(load_study(args.study), _read(args.candidates))
    elif args.command == 'train-text-experiment':
        from .text.experiment import train_experiment as fit_experiment
        result = fit_experiment(_read(args.corpus), args.base, args.output_dir, _read(args.config),
                                save_resume_state=args.save_resume_state, resume_state=args.resume_state,
                                initialize_from=args.initialize_from)
        if int(os.environ.get('RANK', '0')) == 0:
            print(canonical_json({'checkpoint': result['best_checkpoint'], 'validation_mae': result['validation_mae']}))
        return 0
    elif args.command == 'study-text':
        from .text.study import run_text_study
        print(canonical_json(run_text_study(_read(args.corpus), _read(args.protocol), args.base, args.output_dir,
                                         initialize_from=args.initialize_from))); return 0
    elif args.command == 'evaluate-text-experiment':
        from .text.experiment import evaluate_experiment as evaluate_text
        result = evaluate_text(_read(args.corpus), args.checkpoint, args.base, args.output_dir, device=args.device, regression_only=args.regression_only)
        print(canonical_json({'checkpoint': result['best_checkpoint'], 'test': result['test']})); return 0
    elif args.command == 'score-text-experiment':
        from .text.predict import TextPredictor
        result = TextPredictor(args.checkpoint, device=args.device, base_path=args.base, allow_experimental=True).predict(args.category, args.text)
    save_report(result, args.output)
    print(canonical_json({'output': args.output})); return 0
