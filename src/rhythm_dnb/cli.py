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
    score = sub.add_parser('score'); score.add_argument('--bundle', required=True); score.add_argument('--request', required=True); score.add_argument('--state'); score.add_argument('--output', required=True); score.add_argument('--method', choices=('single_sample', 'rolling'), default='single_sample')
    develop = sub.add_parser('develop'); develop.add_argument('--input', required=True); develop.add_argument('--study', required=True); develop.add_argument('--output-dir', required=True)
    validate = sub.add_parser('validate'); validate.add_argument('--bundle', required=True); validate.add_argument('--cases', required=True); validate.add_argument('--as-of', required=True); validate.add_argument('--output', required=True); validate.add_argument('--bootstrap', type=int, default=1000)
    validate.add_argument('--events', required=True)
    validate.add_argument('--monitoring', required=True)
    prepare = sub.add_parser('prepare'); prepare.add_argument('--input', required=True); prepare.add_argument('--output', required=True)
    quantify = sub.add_parser('quantify'); quantify.add_argument('--input', required=True); quantify.add_argument('--output', required=True)
    acquire = sub.add_parser('download-text'); acquire.add_argument('--config', required=True); acquire.add_argument('--output-dir', required=True); acquire.add_argument('--cache-dir', required=True)
    plot = sub.add_parser('plot-text'); plot.add_argument('--result', required=True); plot.add_argument('--predictions', required=True); plot.add_argument('--output-dir', required=True)
    criteria = sub.add_parser('fit-endpoint'); criteria.add_argument('--input', required=True); criteria.add_argument('--study', required=True); criteria.add_argument('--cutoff', required=True); criteria.add_argument('--output', required=True)
    endpoint = sub.add_parser('endpoints'); endpoint.add_argument('--input', required=True); endpoint.add_argument('--study', required=True); endpoint.add_argument('--as-of', required=True); endpoint.add_argument('--output', required=True)
    train = sub.add_parser('train-text'); train.add_argument('--corpus', required=True); train.add_argument('--base', required=True); train.add_argument('--config', required=True); train.add_argument('--output-dir', required=True)
    text = sub.add_parser('score-text'); text.add_argument('--checkpoint', required=True); text.add_argument('--category', required=True); text.add_argument('--text', required=True); text.add_argument('--output', required=True)
    text.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    text.add_argument('--base', help='Exact local base weights required for Qwen adapters')
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
        from .contracts import parse_request, AlarmState
        state = _read(args.state) if args.state else {}
        for key in ('last_day', 'last_alarm_day'):
            if state.get(key):
                state[key] = date.fromisoformat(state[key])
        result = RhythmPredictor.from_bundle(args.bundle).predict(parse_request(_read(args.request)), AlarmState(**state), method=args.method)
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
    elif args.command == 'train-text':
        from .text.train import train as fit_text
        result = fit_text(_read(args.corpus), args.base, args.output_dir, _read(args.config))
        if int(os.environ.get('RANK', '0')) == 0:
            print(canonical_json({'checkpoint': result['best_checkpoint'], 'test': result['test']}))
        return 0
    elif args.command == 'score-text':
        from .text.predict import TextPredictor
        result = TextPredictor(args.checkpoint, device=args.device, base_path=args.base).predict(args.category, args.text)
    save_report(result, args.output)
    print(canonical_json({'output': args.output})); return 0
