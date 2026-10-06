"""Run prespecified repeated text experiments using validation only and retain every outcome."""

import math
from pathlib import Path
from statistics import mean, stdev
from ..provenance import canonical_json, fingerprint
from .config import validate_config
from .experiment import prepare_development, train_experiment
from .review import training_readiness


def summarize_repetitions(results):
    # PSEUDOCODE: require a matched seed set -> aggregate all repetitions -> rank arms on validation only.
    if not results:
        raise ValueError('No completed repetitions.')
    arms = {}
    for row in results:
        if not math.isfinite(row['validation_mae']) or row.get('test') is not None:
            raise ValueError('Repeated selection may use finite validation errors only.')
        bucket = arms.setdefault(row['arm'], {})
        if row['seed'] in bucket:
            raise ValueError('Duplicate seed within one experimental arm.')
        bucket[row['seed']] = row
    seeds = {tuple(sorted(bucket)) for bucket in arms.values()}
    if len(seeds) != 1 or len(next(iter(seeds))) < 2:
        raise ValueError('Arms need the same two or more prespecified seeds.')
    summary = {}
    for name, runs in arms.items():
        values = [r['validation_mae'] for r in runs.values()]
        summary[name] = {'validation_mean_mae': mean(values), 'seed_standard_deviation': stdev(values),
                         'runs': list(runs.values())}
    selected = min(summary, key=lambda name: (summary[name]['validation_mean_mae'], name))
    return {'arms': summary, 'selected_arm': selected, 'test_used_for_selection': False,
            'automatic_model_promotion': False, 'clinical_validation': False,
            'interpretation': 'seed spread is repeatability evidence, not a population confidence interval; per-head and context checks remain required'}


def run_text_study(payload, protocol, base_path, output_dir, *, initialize_from=None):
    # PSEUDOCODE: preflight every arm -> freeze the entire plan -> execute all seeds -> save validation-only comparison.
    import os
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('The repeated-study coordinator runs on one GPU; invoke individual distributed runs separately.')
    partitions, corpus = prepare_development(payload)
    if set(protocol) != {'seeds', 'arms', 'rationale'} or not protocol['rationale']:
        raise ValueError('A study must explicitly declare seeds, named arm configs and their rationale.')
    seeds = protocol['seeds']
    if not isinstance(seeds, list) or len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('Prespecify at least two distinct seeds; do not select the best seed after testing.')
    planned = []
    for arm, raw in protocol['arms'].items():
        if not isinstance(arm, str) or not arm or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in arm):
            raise ValueError('Arm names must be safe lowercase path components.')
        for seed in seeds:
            config = validate_config({**raw, 'seed': seed})
            readiness = training_readiness(partitions, config, experimental=True)
            if readiness['blockers']:
                raise ValueError(arm + ': ' + '; '.join(readiness['blockers']))
            planned.append({'arm': arm, 'seed': seed, 'config': config})
    if not planned:
        raise ValueError('No experimental arms.')
    receipt = {'protocol': protocol, 'corpus_id': corpus['id'], 'runs': planned,
               'selection': 'mean_validation_mae_across_all_prespecified_seeds', 'test_loaded': False}
    receipt['id'] = fingerprint(receipt)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    (output / 'protocol.json').write_text(canonical_json(receipt), encoding='utf-8')
    results = []
    for run in planned:
        result = train_experiment(payload, base_path, output / (run['arm'] + '-' + str(run['seed'])), run['config'],
                                  save_resume_state=True, initialize_from=initialize_from)
        results.append({**{k: run[k] for k in ('arm', 'seed')}, 'validation_mae': result['validation_mae'],
                        'best_checkpoint': result['best_checkpoint'], 'test': None})
        (output / 'completed-runs.json').write_text(canonical_json(results), encoding='utf-8')
    summary = summarize_repetitions(results)
    (output / 'summary.json').write_text(canonical_json(summary), encoding='utf-8')
    return summary
