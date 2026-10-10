"""Export development-only, reference-scaled rhythm matrices for the upstream R package."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rhythm_dnb.measures.scaling import fit_scaler, transform
from rhythm_dnb.provenance import file_hash, fingerprint
from rhythm_dnb.research.experiment_data import load_packet, save_json, OBJECTIVE, TEXT
from rhythm_dnb.timebase import instant


def prepare_matrices(reference, train, pairs, names, scaler):
    # PSEUDOCODE: validate cohort isolation and paired chronology -> apply one reference scale to both stages.
    if not reference or not train or not pairs:
        raise ValueError('Reference, development records and original pairs are required.')
    ref_ids = [r['participant_id'] for r in reference]
    if len(set(ref_ids)) != len(ref_ids) or any(r['split'] != 'reference' for r in reference):
        raise ValueError('Reference must contain one observation per independent reference person.')
    if any(r['split'] != 'train' or r['participant_id'] in ref_ids for r in train):
        raise ValueError('Only isolated training people may enter R discovery.')
    by_id = {r['record_id']: r for r in train}
    if len(by_id) != len(train):
        raise ValueError('Duplicate development record IDs.')
    seen, columns, meta = set(), [], []
    for stage, key in [('stable', 'stable_record'), ('pre_event', 'pre_event_record')]:
        for pair in pairs:
            person = pair['participant_id']
            if stage == 'stable':
                if person in seen:
                    raise ValueError('Repeated person in discovery pairs.')
                seen.add(person)
            a, b = by_id[pair['stable_record']], by_id[pair['pre_event_record']]
            if a['participant_id'] != person or b['participant_id'] != person:
                raise ValueError('Discovery pair identity differs.')
            if not instant(a['issued_at']) < instant(b['issued_at']):
                raise ValueError('Discovery pair is not chronological.')
            row = by_id[pair[key]]
            columns.append([row['features'][name] for name in names])
            meta.append({k: row[k] for k in ('record_id', 'participant_id', 'split', 'observed_at', 'issued_at')})
            meta[-1]['stage'] = stage
    values = transform(columns, scaler)
    if len(seen) < 9 or not np.isfinite(values).all():
        raise ValueError('At least nine complete paired people are required.')
    for stage in ('stable', 'pre_event'):
        subset = values[[r['stage'] == stage for r in meta]]
        if np.any(np.std(subset, axis=0, ddof=1) <= 1e-12):
            raise ValueError('Constant feature would produce undefined correlation in DNBr.')
    return values.T, meta


def export(packet, evaluation, predictions, config, output):
    # PSEUDOCODE: verify frozen inputs -> preserve original training pairs -> export without reading held-out row content.
    output, evaluation = Path(output), Path(evaluation)
    if output.exists():
        raise FileExistsError('Use a new output directory; preserve previous experiments.')
    data, manifest = load_packet(packet, names={'reference', 'train', 'protocol'})
    frozen = json.loads((evaluation / 'frozen-policy.json').read_text(encoding='utf-8'))
    discovery = json.loads((evaluation / 'discovery.json').read_text(encoding='utf-8'))
    scaler = json.loads((evaluation / 'scaler.json').read_text(encoding='utf-8'))
    if frozen['packet_id'] != manifest['id']:
        raise ValueError('Evaluation belongs to a different packet.')
    names = list(OBJECTIVE)
    if frozen['inference_id'] is not None:
        if not predictions:
            raise ValueError('Original text predictions are required for the joint panel.')
        prediction = json.loads(Path(predictions).read_text(encoding='utf-8'))
        if (prediction['packet_id'] != manifest['id'] or prediction['id'] != frozen['inference_id'] or
                fingerprint({k: v for k, v in prediction.items() if k != 'id'}) != prediction['id']):
            raise ValueError('Text prediction identity differs from the original run.')
        names += list(TEXT)
        for row in data['reference'] + data['train']:
            for category, text in row['texts'].items():
                scores = prediction['predictions'][fingerprint([category, text])]
                row['features'].update({'text_' + k: v for k, v in scores.items() if 'text_' + k in TEXT})
    elif predictions:
        raise ValueError('Cannot add text to an objective-only saved evaluation.')
    if names != scaler['features']:
        raise ValueError('Feature order differs from the saved reference scale.')
    fitted = fit_scaler([[r['features'][k] for k in names] for r in data['reference']], names)
    for key in ('mean', 'scale'):
        np.testing.assert_allclose(fitted[key], scaler[key], atol=1e-12, rtol=1e-12)
    if fitted['clock_centers'].keys() != scaler['clock_centers'].keys():
        raise ValueError('Clock anchors differ.')
    for key, value in fitted['clock_centers'].items():
        np.testing.assert_allclose(value, scaler['clock_centers'][key], atol=1e-12, rtol=1e-12)
    by_id = {r['record_id']: r for r in data['train']}
    for pair in discovery['pairs']:
        a, b = by_id[pair['stable_record']], by_id[pair['pre_event_record']]
        onset = instant(b['outcome']['event_onset_at'])
        eligible = sorted((r for r in data['train'] if r['participant_id'] == pair['participant_id'] and
                           (onset - instant(r['issued_at'])).total_seconds() >=
                           3600 * data['protocol']['discovery_pre_event_lead_hours']),
                          key=lambda r: instant(r['issued_at']))
        if not eligible or (a['record_id'], b['record_id']) != (eligible[0]['record_id'], eligible[-1]['record_id']):
            raise ValueError('Discovery pairs differ from the preserved time rule.')
    matrix, meta = prepare_matrices(data['reference'], data['train'], discovery['pairs'], names, scaler)
    settings = json.loads(Path(config).read_text(encoding='utf-8'))
    output.mkdir(parents=True)
    with (output / 'matrix.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['feature'] + [r['record_id'] for r in meta])
        writer.writerows([name, *matrix[i]] for i, name in enumerate(names))
    with (output / 'samples.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(meta[0]))
        writer.writeheader(); writer.writerows(meta)
    save_json(output / 'settings.json', settings)
    save_json(output / 'scaler.json', scaler)
    payload = {'purpose': 'development_analysis_not_validated_warning_model', 'domain': manifest['domain'],
               'packet_id': manifest['id'], 'inference_id': frozen['inference_id'],
               'parent_implementation_id': frozen['implementation_id'],
               'parent_discovery_sha256': file_hash(evaluation / 'discovery.json'),
               'features': names, 'stages': ['stable', 'pre_event'],
               'people': len(meta) // 2, 'observations': len(meta),
               'selection_changed': True, 'warning_policy_calibrated': False,
               'files': {name: file_hash(output / name) for name in
                         ('matrix.csv', 'samples.csv', 'settings.json', 'scaler.json')}}
    save_json(output / 'manifest.json', {**payload, 'id': fingerprint(payload)})
    return payload


def main():
    # PSEUDOCODE: require explicit immutable inputs and a fresh destination.
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'evaluation', 'config', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--predictions')
    args = parser.parse_args()
    print(json.dumps(export(args.packet, args.evaluation, args.predictions, args.config, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
