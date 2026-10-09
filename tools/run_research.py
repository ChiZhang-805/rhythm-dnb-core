"""Prepare locally, then run the same frozen authored experiment on a GPU server."""

import argparse
import json
from pathlib import Path


def main():
    # PSEUDOCODE: dispatch explicit stages without silently starting a GPU workload from preparation.
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    for name in ('database', 'master', 'corpus', 'protocol', 'output'):
        prepare.add_argument('--' + name, required=True)
    preflight = commands.add_parser('preflight')
    for name in ('packet', 'base', 'config', 'output'):
        preflight.add_argument('--' + name, required=True)
    evaluate = commands.add_parser('evaluate')
    for name in ('packet', 'output'):
        evaluate.add_argument('--' + name, required=True)
    evaluate.add_argument('--predictions')
    evaluate.add_argument('--plots', action='store_true')
    gpu = commands.add_parser('gpu')
    gpu.add_argument('--plots', action='store_true')
    worker = commands.add_parser('_train', help=argparse.SUPPRESS)
    for command in (gpu, worker):
        for name in ('packet', 'base', 'plan', 'output'):
            command.add_argument('--' + name, required=True)
    worker.add_argument('--candidate', type=int, required=True)
    worker.add_argument('--resume')
    args = parser.parse_args()
    if args.command == 'prepare':
        from rhythm_dnb.research.experiment import validate_protocol
        from rhythm_dnb.research.experiment_data import prepare_packet
        protocol = json.loads(Path(args.protocol).read_text(encoding='utf-8'))
        validate_protocol(protocol)
        result = prepare_packet(args.database, args.corpus, args.master, args.output, protocol)
    elif args.command == 'preflight':
        from rhythm_dnb.research.experiment_gpu import preflight as check
        result = check(args.packet, args.base, json.loads(Path(args.config).read_text(encoding='utf-8')), args.output)
        result = {k: v for k, v in result.items() if k not in ('readiness', 'configs')}
    elif args.command == 'evaluate':
        from rhythm_dnb.research.experiment import run_experiment
        result = run_experiment(args.packet, args.output, text_predictions=args.predictions)
        if args.plots:
            from rhythm_dnb.research.plots import plot_experiment_result
            plot_experiment_result(result, Path(args.output) / 'comparison')
        result = {k: v for k, v in result.items() if k != 'methods'}
    elif args.command == '_train':
        from rhythm_dnb.research.experiment_gpu import train_candidate
        result = train_candidate(args.packet, args.base, args.plan, args.candidate, args.output, resume=args.resume)
        result = {k: v for k, v in result.items() if k not in ('corpus', 'history')}
    else:
        from rhythm_dnb.research.experiment_gpu import fit_gpu, score_text
        from rhythm_dnb.research.experiment import run_experiment
        output = Path(args.output)
        checkpoint = fit_gpu(args.packet, args.base, args.plan, output / 'training', Path(__file__).resolve())
        predictions = score_text(args.packet, checkpoint, args.base, output / 'text-scores')
        attempts = sorted(output.glob('joint-evaluation-*'))
        completed = [p for p in attempts if (p / 'result.json').exists()]
        if completed:
            result = json.loads((completed[-1] / 'result.json').read_text(encoding='utf-8'))
            from rhythm_dnb.research.experiment_data import load_packet
            from rhythm_dnb.research.experiment_gpu import attach_predictions
            data, manifest = load_packet(args.packet)
            if result['inference_id'] != attach_predictions(data, manifest, predictions) or result['packet_id'] != manifest['id']:
                raise ValueError('Saved evaluation differs from current packet/predictions.')
        else:
            destination = output / f'joint-evaluation-{len(attempts) + 1:03d}'
            result = run_experiment(args.packet, destination, text_predictions=predictions)
        if args.plots:
            from rhythm_dnb.research.plots import plot_experiment_result
            destination = completed[-1] if completed else destination
            if not (destination / 'comparison.png').exists():
                plot_experiment_result(result, destination / 'comparison')
        result = {k: v for k, v in result.items() if k != 'methods'}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
