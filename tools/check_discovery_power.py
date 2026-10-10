"""Compare permutation screening on prespecified artificial controls; never read participant datasets."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from rhythm_dnb.dnb.modules import candidate_modules
from rhythm_dnb.provenance import file_hash, fingerprint
from rhythm_dnb.research.discover import _component_weights, _components, _directional_statistic
from rhythm_dnb.research.experiment import validate_protocol
from rhythm_dnb.research.experiment_data import save_json


def permutation_adjustments(statistics):
    """Row zero is the observation; other rows are common, paired-label permutations."""
    # PSEUDOCODE: reproduce maxT -> rank each candidate on its own permutation distribution -> single-step minP.
    values = np.asarray(statistics, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError('Finite observation-plus-permutation statistics are required.')
    # Include the observed row and use worst ranks for ties, so Monte Carlo p-values are never zero.
    p = rankdata(-values, method='max', axis=0) / len(values)
    maxima = values.max(axis=1)
    minimum_p = p.min(axis=1)
    max_t = np.array([np.mean(maxima >= x) for x in values[0]])
    min_p = np.array([np.mean(minimum_p <= x) for x in p[0]])
    return {'maxT': max_t, 'minP': min_p, 'unadjusted': p[0]}


def control_matrices(seed, scenario, people=30, width=10):
    # PSEUDOCODE: generate all cases from fixed stochastic mechanisms; never retry a seed until a test passes.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 291]))
    common = rng.normal(size=(people, 1))
    a = .8 * common + .6 * rng.normal(size=(people, width))
    if scenario == 'exchangeable_null':
        b = .8 * rng.normal(size=(people, 1)) + .6 * rng.normal(size=(people, width))
    elif scenario == 'mean_shift_only':
        b = a + np.array([3., -3., 3.] + [0.] * (width - 3))
    elif scenario == 'known_dnb_mechanism':
        b = rng.normal(size=(people, width))
        latent = rng.normal(size=(people, 1)) * 3
        b[:, :3] = latent * [1, -1, 1] + rng.normal(scale=.08, size=(people, 3))
    else:
        raise ValueError('Unknown artificial control.')
    return a, b


def screen(a, b, names, config):
    # PSEUDOCODE: use the identical statistic and common permutations for both corrections; do not choose a warning threshold.
    modules = list(candidate_modules(names, method='bounded_exhaustive', sizes=config.module_sizes).values())
    indices = []
    for module in modules:
        inside = np.array([names.index(x) for x in module])
        outside = np.array([i for i, x in enumerate(names) if x not in module])
        indices.append((inside, outside, np.triu_indices(len(inside), 1)))
    weights = _component_weights(indices, len(names))
    statistics = np.zeros((config.permutation_repetitions + 1, len(modules)))
    base, pre = _components(a, weights), _components(b, weights)
    statistics[0] = _directional_statistic(base, pre, config.epsilon)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    for j in range(1, len(statistics)):
        swap = rng.integers(0, 2, size=(len(a), 1)).astype(bool)
        x, y = _components(np.where(swap, b, a), weights), _components(np.where(swap, a, b), weights)
        statistics[j] = _directional_statistic(x, y, config.epsilon)
    adjusted = permutation_adjustments(statistics)
    result = {}
    for method in ('maxT', 'minP'):
        chosen = [m for m, p, t in zip(modules, adjusted[method], statistics[0]) if p <= config.discovery_alpha and t > 0]
        result[method] = {'screened_modules': chosen, 'any_rejection': bool(chosen),
                          'known_group_detected': any(set(m) <= set(names[:3]) for m in chosen),
                          'known_group_adjusted_p': float(adjusted[method][modules.index(names[:3])])}
    return result


def main():
    # PSEUDOCODE: freeze an artificial-control plan before execution -> save every case and a final summary.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--scenarios', nargs='+', choices=['exchangeable_null', 'mean_shift_only', 'known_dnb_mechanism'],
                        default=['exchangeable_null', 'mean_shift_only', 'known_dnb_mechanism'])
    args = parser.parse_args()
    if not 2 <= args.repeats <= 1000:
        raise ValueError('Use 2 to 1000 prespecified repeats.')
    if args.seed_offset < 0 or len(set(args.scenarios)) != len(args.scenarios):
        raise ValueError('Use a nonnegative seed offset and distinct scenarios.')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a new directory; preserve previous findings.')
    protocol = json.loads(Path(args.protocol).read_text(encoding='utf-8'))
    config = validate_protocol(protocol)
    plan = {'purpose': 'artificial_permutation_screen_power_diagnostic_not_DNB_accuracy',
            'created_at': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': file_hash(args.protocol),
            'script_sha256': file_hash(__file__), 'config': asdict(config),
            'scenarios': args.scenarios,
            'data_seeds': [protocol['seed'] + args.seed_offset + i for i in range(args.repeats)], 'people': 30, 'width': 10,
            'methods': ['original_maxT', 'exploratory_single_step_minP'],
            'alpha_unchanged': config.discovery_alpha, 'no_participant_data_read': True,
            'bootstrap_gate_applied': False,
            'limitation': 'Screening only. Global-null exchangeability; strong FWER under partial nulls is not established.'}
    output.mkdir(parents=True); save_json(output / 'plan.json', plan)
    (output / 'source.py').write_bytes(Path(__file__).read_bytes())
    names = [f'control_{i}' for i in range(plan['width'])]
    results = []
    for scenario in plan['scenarios']:
        for seed in plan['data_seeds']:
            a, b = control_matrices(seed, scenario)
            row = {'scenario': scenario, 'seed': seed, 'matrices_id': fingerprint([a.tolist(), b.tolist()]),
                   'methods': screen(a, b, names, config)}
            save_json(output / f'{scenario}-{seed}.json', row); results.append(row)
            print(f'{scenario} seed={seed}: maxT={row["methods"]["maxT"]["any_rejection"]} minP={row["methods"]["minP"]["any_rejection"]}', flush=True)
    summary = {}
    for scenario in plan['scenarios']:
        rows = [r for r in results if r['scenario'] == scenario]
        summary[scenario] = {m: {'any_rejection_runs': sum(r['methods'][m]['any_rejection'] for r in rows),
                                'known_group_detected_runs': sum(r['methods'][m]['known_group_detected'] for r in rows),
                                'total_runs': len(rows)} for m in ('maxT', 'minP')}
    save_json(output / 'result.json', {'plan_id': fingerprint(plan), 'summary': summary,
                                      'scientific_policy_changed': False, 'dnb_warning_accuracy': None})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
