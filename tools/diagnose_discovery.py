"""Explain a frozen negative DNB finding using development people only; never refit a warning policy."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path

import numpy as np

from rhythm_dnb.provenance import file_hash, fingerprint
from rhythm_dnb.measures.scaling import fit_scaler, transform
from rhythm_dnb.research.discover import discover_matrices
from rhythm_dnb.research.experiment import validate_protocol
from rhythm_dnb.research.experiment_data import load_packet, save_json, OBJECTIVE, TEXT
from rhythm_dnb.research.experiment_gpu import implementation_id
from rhythm_dnb.timebase import instant


def gate_summary(discovery, protocol):
    # PSEUDOCODE: count overlapping failure reasons and sequential gates without choosing a replacement module.
    rows = discovery['candidates']
    base = np.array([r['stable_components'] for r in rows], dtype=float)
    pre = np.array([r['pre_event_components'] for r in rows], dtype=float)
    if base.shape != (len(rows), 3) or not np.isfinite([base, pre]).all():
        raise ValueError('Invalid saved discovery components.')
    directions = (pre[:, 0] > base[:, 0], pre[:, 1] > base[:, 1], pre[:, 2] < base[:, 2])
    three = np.logical_and.reduce(directions) & (pre[:, 2] > protocol['epsilon'])
    stable = np.array([r['stability'] >= protocol['module_stability'] for r in rows])
    significant = np.array([r['max_stat_adjusted_p'] <= protocol['discovery_alpha'] for r in rows])
    if any(bool(three[i]) != r['all_three_conditions'] or
           bool(three[i] and stable[i] and significant[i]) != r['eligible'] for i, r in enumerate(rows)):
        raise ValueError('Saved gate decisions differ from the frozen protocol.')
    return {'candidates': len(rows), 'sd_increase': int(directions[0].sum()),
            'internal_correlation_increase': int(directions[1].sum()),
            'external_correlation_decrease': int(directions[2].sum()),
            'all_three': int(three.sum()), 'bootstrap_pass_any': int(stable.sum()),
            'three_and_bootstrap': int((three & stable).sum()),
            'all_gates': int((three & stable & significant).sum()),
            'max_bootstrap_stability': max(r['stability'] for r in rows),
            'min_adjusted_p': min(r['max_stat_adjusted_p'] for r in rows)}


def manual_components(matrix, names, module):
    # PSEUDOCODE: independently sum individual Pearson pairs; do not reuse discovery's matrix averaging masks.
    inside = [names.index(name) for name in module]
    outside = [i for i in range(len(names)) if i not in inside]
    sd = sum(float(np.std(matrix[:, i], ddof=1)) for i in inside) / len(inside)
    internal = [abs(float(np.corrcoef(matrix[:, i], matrix[:, j])[0, 1]))
                for offset, i in enumerate(inside) for j in inside[offset + 1:]]
    external = [abs(float(np.corrcoef(matrix[:, i], matrix[:, j])[0, 1]))
                for i in inside for j in outside]
    return [sd, sum(internal) / len(internal), sum(external) / len(external)]


def numerical_controls(config, width):
    """Artificial software controls only, never training data or a rhythm-accuracy result."""
    # PSEUDOCODE: fix one seed and matrix size -> run identical and known-mechanism controls at unchanged study gates.
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 291]))
    names = [f'control_{i}' for i in range(width)]
    common = rng.normal(size=(30, 1))
    stable = .8 * common + .6 * rng.normal(size=(30, width))
    transition = rng.normal(size=(30, width))
    latent = rng.normal(size=(30, 1)) * 3
    transition[:, :3] = latent * [1, -1, 1] + rng.normal(scale=.08, size=(30, 3))
    result = {}
    for name, values in [('identical', stable.copy()), ('known_dnb_mechanism', transition)]:
        scan = discover_matrices(stable, values, names, config)
        selected = [r['module'] for r in scan['chosen']]
        result[name] = {'qualified_modules': selected, 'matrix_hash': fingerprint(values.tolist()),
                        'seed': config.seed, 'people': 30, 'features': width,
                        'known_module_details': next(r for r in scan['candidates'] if r['module'] == names[:3]),
                        'purpose': 'artificial_software_check_not_research_accuracy'}
        print('Control finished: ' + name, flush=True)
    result['negative_passed'] = not result['identical']['qualified_modules']
    result['positive_passed'] = any(set(m) <= set(names[:3]) for m in result['known_dnb_mechanism']['qualified_modules'])
    return result


def diagnose(packet, evaluation, predictions, output, *, controls=False):
    # PSEUDOCODE: verify immutable evidence -> analyze reference/train only -> save new diagnostics without changing the old run.
    output, evaluation = Path(output), Path(evaluation)
    if output.exists():
        raise FileExistsError('Preserve existing diagnostics; use a new directory.')
    data, manifest = load_packet(packet, names={'reference', 'train', 'protocol'})
    protocol = data['protocol']; config = validate_protocol(protocol)
    discovery = json.loads((evaluation / 'discovery.json').read_text(encoding='utf-8'))
    frozen = json.loads((evaluation / 'frozen-policy.json').read_text(encoding='utf-8'))
    scaler = json.loads((evaluation / 'scaler.json').read_text(encoding='utf-8'))
    if frozen['packet_id'] != manifest['id'] or frozen['implementation_id'] != implementation_id():
        raise ValueError('Discovery belongs to a different packet or implementation.')
    names = list(OBJECTIVE)
    if predictions:
        prediction = json.loads(Path(predictions).read_text(encoding='utf-8'))
        if (prediction['packet_id'] != manifest['id'] or prediction['id'] != frozen['inference_id'] or
            fingerprint({k: v for k, v in prediction.items() if k != 'id'}) != prediction['id']):
            raise ValueError('Text predictions differ from the saved evaluation.')
        names += list(TEXT)
        for row in data['reference'] + data['train']:
            for category, text in row['texts'].items():
                for key, value in prediction['predictions'][fingerprint([category, text])].items():
                    if 'text_' + key in TEXT:
                        row['features']['text_' + key] = value
    elif frozen['inference_id'] is not None:
        raise ValueError('The joint evaluation requires its original predictions.')
    fitted = fit_scaler([[r['features'][k] for k in names] for r in data['reference']], names)
    if fingerprint(fitted) != fingerprint(scaler):
        # Tiny platform roundoff is allowed, but a changed scale or clock anchor is not.
        if fitted['features'] != scaler['features'] or fitted['clock_centers'].keys() != scaler['clock_centers'].keys():
            raise ValueError('Reference scaler changed.')
        for key in ('mean', 'scale'):
            np.testing.assert_allclose(fitted[key], scaler[key], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(list(fitted['clock_centers'].values()), list(scaler['clock_centers'].values()), atol=1e-12, rtol=1e-12)
    by_id = {r['record_id']: r for r in data['train']}
    before, after, pairs = [], [], []
    groups = defaultdict(list)
    for r in data['train']:
        groups[r['participant_id']].append(r)
    for pair in discovery['pairs']:
        a, b = by_id[pair['stable_record']], by_id[pair['pre_event_record']]
        if a['participant_id'] != b['participant_id'] or a['participant_id'] != pair['participant_id']:
            raise ValueError('Discovery pair is not the same development person.')
        onset = instant(b['outcome']['event_onset_at'])
        eligible = sorted([r for r in groups[a['participant_id']] if
                           (onset - instant(r['issued_at'])).total_seconds() >= 3600 * protocol['discovery_pre_event_lead_hours']],
                          key=lambda r: instant(r['issued_at']))
        if [eligible[0]['record_id'], eligible[-1]['record_id']] != [a['record_id'], b['record_id']]:
            raise ValueError('Saved pair differs from the prespecified time rule.')
        before.append([a['features'][k] for k in names]); after.append([b['features'][k] for k in names])
        pairs.append({**pair, 'stable_lead_hours': (onset - instant(a['issued_at'])).total_seconds() / 3600,
                      'pre_event_lead_hours': (onset - instant(b['issued_at'])).total_seconds() / 3600,
                      'days_between_measurements': (instant(b['observed_at']) - instant(a['observed_at'])).total_seconds() / 86400})
    before, after = transform(before, scaler), transform(after, scaler)
    errors = []
    for row in discovery['candidates']:
        for matrix, key in [(before, 'stable_components'), (after, 'pre_event_components')]:
            errors.extend(abs(np.array(manual_components(matrix, names, row['module'])) - row[key]))
    if max(errors) > 1e-10:
        raise ValueError('Independent numerical recalculation differs from the original result.')
    summary = gate_summary(discovery, protocol)
    feature_rows = [{'feature': name, 'stable_sd': float(before[:, j].std(ddof=1)),
                     'pre_event_sd': float(after[:, j].std(ddof=1)),
                     'sd_ratio': float(after[:, j].std(ddof=1) / before[:, j].std(ddof=1)),
                     'mean_change': float(after[:, j].mean() - before[:, j].mean())} for j, name in enumerate(names)]
    trajectory = []
    for event_group in (False, True):
        selected = [sorted(v, key=lambda r: instant(r['observed_at'])) for v in groups.values()
                    if bool(v[0]['outcome']['event_onset_at']) == event_group]
        for day in range(min(map(len, selected))):
            values = transform([[seq[day]['features'][k] for k in names] for seq in selected], scaler)
            for j, name in enumerate(names):
                trajectory.append({'event_group': event_group, 'day': day, 'feature': name,
                                   'people': len(selected), 'mean': float(values[:, j].mean()),
                                   'between_person_sd': float(values[:, j].std(ddof=1))})
    output.mkdir(parents=True)
    for name, rows in [('features', feature_rows), ('pairs', pairs), ('trajectories', trajectory)]:
        with (output / (name + '.csv')).open('x', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {'created_at': datetime.now(timezone.utc).isoformat(), 'packet_id': manifest['id'],
              'implementation_id': implementation_id(), 'inference_id': frozen['inference_id'],
              'scope': 'development_only_posthoc_diagnostic_not_new_validation',
              'validation_or_test_rows_analyzed': False, 'protocol_modified': False,
              'discovery_sha256': file_hash(evaluation / 'discovery.json'), 'gates': summary,
              'independent_recalculation_max_error': float(max(errors)), 'paired_people': len(pairs),
              'features': feature_rows, 'dnb_accuracy': None,
              'controls': numerical_controls(config, len(names)) if controls else None}
    save_json(output / 'diagnosis.json', result)
    return result


def main():
    # PSEUDOCODE: expose explicit paths and optional software controls; never launch training.
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'evaluation', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--predictions')
    parser.add_argument('--controls', action='store_true')
    args = parser.parse_args()
    print(json.dumps(diagnose(args.packet, args.evaluation, args.predictions, args.output, controls=args.controls), indent=2))


if __name__ == '__main__':
    main()
