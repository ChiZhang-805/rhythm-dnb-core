"""Frozen exploratory reference-stability predictors; scoring never reads outcomes or fits parameters."""

from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np

from ..contracts import AlarmState
from ..measures.scaling import transform
from ..provenance import file_hash, fingerprint
from ..timebase import forecast_bounds, instant
from ..warning.policy import advance
from .reference_stability import resample_scores, summarize_draws
from .temporal_warning import causal_history, causal_smooth


METHODS = ('reference_robust_dnb', 'reference_robust_deviation', 'history_robust_dnb', 'history_robust_control')
PURPOSE = 'frozen_exploratory_reference_stability_warning'


def implementation_receipt():
    # PSEUDOCODE: bind inference to the exact installed arithmetic and library versions, independent of local paths.
    root = Path(__file__).resolve().parents[1]
    files = ('research/frozen_warning.py', 'research/reference_stability.py', 'research/temporal_warning.py',
             'measures/scaling.py', 'measures/panel.py', 'warning/policy.py', 'timebase.py', 'contracts.py')
    return {'source': {name: file_hash(root/name) for name in files},
            'runtime': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'joblib')}}


def load_bundle(directory):
    # PSEUDOCODE: verify every local bundle file and arithmetic binding before loading its locally trusted model.
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if fingerprint(manifest['files']) != manifest['id']:
        raise ValueError('Bundle manifest identity mismatch.')
    for name, digest in manifest['files'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or file_hash(path) != digest:
            raise ValueError('Bundle file missing, changed or outside the bundle: '+name)
    if not {'metadata.json', 'reference.npz'} <= set(manifest['files']):
        raise ValueError('Bundle missing required files.')
    meta = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    if meta['purpose'] != PURPOSE or meta['method'] not in METHODS or meta['implementation'] != implementation_receipt():
        raise ValueError('Unsupported method or changed inference runtime; revalidate before using this bundle.')
    with np.load(directory/'reference.npz', allow_pickle=False) as arrays:
        reference, draws = arrays['reference'].copy(), arrays['draws'].copy()
    if reference.shape != (len(meta['reference_people']), len(meta['reference_features'])):
        raise ValueError('Reference dimensions changed.')
    if (not np.isfinite(reference).all() or draws.shape != (meta['replicates'], len(reference))
            or not np.issubdtype(draws.dtype, np.integer) or np.any(draws < 0) or np.any(draws >= len(reference))):
        raise ValueError('Invalid saved reference draws.')
    if len(set(meta['reference_people'])) != len(reference) or set(meta['reference_features']) != set(meta['scaler']['features']):
        raise ValueError('Repeated reference people or feature mismatch.')
    model = None
    if meta['method'].startswith('history_'):
        if 'model.joblib' not in manifest['files']:
            raise ValueError('Missing frozen learner.')
        model = joblib.load(directory/'model.joblib')
    return {'metadata': meta, 'reference': reference, 'draws': draws, 'model': model, 'id': manifest['id']}


def validate_request(payload, meta):
    # PSEUDOCODE: require explicit provenance and model identity; reject future data, truth fields and invalid units.
    if set(payload) != {'source_id', 'source_sha256', 'domain', 'measurement_contract_id', 'as_of', 'rows'}:
        raise ValueError('Request must contain only source metadata and prediction inputs, never outcomes.')
    if (not isinstance(payload['source_id'], str) or not payload['source_id'].strip()
            or not isinstance(payload['source_sha256'], str) or len(payload['source_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in payload['source_sha256'])):
        raise ValueError('Explicit source identity and SHA256 are required.')
    if payload['measurement_contract_id'] != meta['measurement_contract_id']:
        raise ValueError('Text model, objective definitions or feature units differ from the frozen experiment.')
    if payload['domain'] not in ('authored_simulation_not_clinical_validation', 'observed_exploratory_not_validated'):
        raise ValueError('This predictor is an exploratory artifact, not a clinical deployment.')
    cutoff = instant(payload['as_of']); rows = payload['rows']; ids = set(); times = set()
    if not isinstance(rows, list) or not rows:
        raise ValueError('A nonempty history is required.')
    names = meta['scaler']['features']
    for row in rows:
        if set(row) != {'record_id', 'participant_id', 'observed_at', 'available_at', 'issued_at', 'features'}:
            raise ValueError('History contains unsupported fields or outcome information.')
        if any(not isinstance(row[k], str) or not row[k].strip() for k in ('record_id', 'participant_id')):
            raise ValueError('Nonempty record and participant identities are required.')
        observed, available, issued = [instant(row[k]) for k in ('observed_at', 'available_at', 'issued_at')]
        if not observed <= available <= issued <= cutoff:
            raise ValueError('A record is not available at its prediction time or exceeds as_of.')
        # The existing simulation was calibrated on daily UTC-noon forecasts, not arbitrary local clock schedules.
        if (issued.hour, issued.minute, issued.second, issued.microsecond) != (12, 0, 0, 0):
            raise ValueError('This experimental bundle expects one daily UTC-noon forecast.')
        identity = (row['participant_id'], issued)
        if row['record_id'] in ids or identity in times:
            raise ValueError('Duplicate record or person/forecast date.')
        ids.add(row['record_id']); times.add(identity)
        if set(row['features']) != set(names):
            raise ValueError('Provide every feature explicitly; unknown values must be null.')
        for key, value in row['features'].items():
            if value is None:
                continue
            if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                raise ValueError('Invalid numeric feature: '+key)
            if (key.startswith('text_') and value > 100 or key == 'sleep_midpoint_h' and value >= 24
                    or key == 'sleep_duration_h' and value > 24
                    or key in ('exercise_minutes', 'screen_time_min') and value > 1440
                    or key == 'resting_hr_bpm' and value == 0):
                raise ValueError('Feature value contradicts its units: '+key)
    return rows


def prepare_scores(bundle, payload, *, progress=None):
    # PSEUDOCODE: derive frozen reference summaries and causal features once for all matched methods.
    meta = bundle['metadata']; rows = validate_request(payload, meta)
    names = meta['scaler']['features']; ids = [r['record_id'] for r in rows]
    values = transform([[r['features'][k] if r['features'][k] is not None else np.nan for k in names] for r in rows], meta['scaler'])
    complete = np.isfinite(values).all(axis=1)
    order = [names.index(k) for k in meta['reference_features']]
    summaries = {key: np.full((len(rows), 2), np.nan) for key in ('dnb', 'network_ratio', 'deviation')}
    if complete.any():
        draws, _ = resample_scores(bundle['reference'], values[complete][:, order], bundle['draws'], meta['epsilon'], progress=progress)
        for key, scores in draws.items():
            median, spread, _ = summarize_draws(scores)
            summaries[key][complete] = np.c_[median, spread]
    metadata = [{k: row[k] for k in ('record_id', 'participant_id', 'observed_at', 'issued_at')} for row in rows]
    base, base_names, lineage = causal_history(metadata, dict(zip(ids, values)), names, meta['history_days'])
    control_values = dict(zip(ids, np.log1p(summaries['deviation'])))
    network_values = dict(zip(ids, np.log1p(np.c_[summaries['dnb'], summaries['network_ratio']])))
    control, control_names, _ = causal_history(metadata, control_values, ['log_deviation_median', 'log_deviation_iqr'], meta['history_days'])
    network, network_names, _ = causal_history(metadata, network_values,
        ['log_dnb_median', 'log_dnb_iqr', 'log_network_ratio_median', 'log_network_ratio_iqr'], meta['history_days'])
    learned = {'history_robust_control': {rid: np.r_[base[rid], control[rid]] for rid in ids},
               'history_robust_dnb': {rid: np.r_[base[rid], control[rid], network[rid]] for rid in ids}}
    valid = {'history_robust_control': {rid: bool(complete[i] and np.isfinite(control_values[rid]).all()) for i, rid in enumerate(ids)},
             'history_robust_dnb': {rid: bool(complete[i] and np.isfinite(control_values[rid]).all() and np.isfinite(network_values[rid]).all()) for i, rid in enumerate(ids)}}
    return {'feature_contract_id': meta['feature_contract_id'], 'request_id': fingerprint(payload), 'rows': rows,
            'metadata': metadata, 'learned': learned, 'valid': valid, 'lineage': lineage,
            'columns': {'history_robust_control': base_names+control_names, 'history_robust_dnb': base_names+control_names+network_names},
            'raw': {method: {rid: float(summaries[key][i, 0]) if np.isfinite(summaries[key][i]).all() else None for i, rid in enumerate(ids)}
                    for method, key in (('reference_robust_dnb', 'dnb'), ('reference_robust_deviation', 'deviation'))}}


def predict_prepared(bundle, prepared):
    # PSEUDOCODE: apply saved weights/threshold and replay the saved persistence policy; never fit on input people.
    meta = bundle['metadata']; method = meta['method']; rows = prepared['rows']
    if prepared['feature_contract_id'] != meta['feature_contract_id']:
        raise ValueError('Prepared features belong to another reference or measurement contract.')
    if method.startswith('history_'):
        if prepared['columns'][method] != meta['columns']:
            raise ValueError('Frozen learner feature order changed.')
        ids = [r['record_id'] for r in rows if prepared['valid'][method][r['record_id']]]
        scores = {r['record_id']: None for r in rows}
        if ids:
            matrix = np.array([prepared['learned'][method][rid] for rid in ids])
            probabilities = bundle['model'].predict_proba(matrix)[:, 1]
            if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
                raise ValueError('Frozen learner produced invalid scores.')
            scores.update({rid: float(value) for rid, value in zip(ids, probabilities)})
    else:
        scores = causal_smooth(prepared['metadata'], prepared['raw'][method], meta['half_life'])
    states, output = {}, []
    for row in sorted(rows, key=lambda r: (r['participant_id'], instant(r['issued_at']))):
        person = row['participant_id']; score = scores[row['record_id']]; threshold = meta['threshold']
        warning, status, state = advance(states.get(person, AlarmState()), person, bundle['id'],
            forecast_bounds(row['issued_at'], 'UTC')[0], score, threshold,
            consecutive=meta['alarm_consecutive'], cooldown_days=meta['cooldown_days'], context=method+'|UTC')
        states[person] = state
        output.append({'record_id': row['record_id'], 'participant_id': person, 'issued_at': row['issued_at'],
                       'score': score, 'risk_exceeds_threshold': None if score is None or threshold is None else int(score > threshold),
                       'warning': warning, 'status': status, 'state': asdict(state)})
    return {'bundle_id': bundle['id'], 'request_id': prepared['request_id'], 'method': method,
            'interpretation': 'exploratory_warning_not_diagnosis_or_calibrated_probability',
            'rows': output, 'lineage': prepared['lineage']}


def predict(bundle, payload, *, progress=None):
    # PSEUDOCODE: run the full frozen prediction path from quantified history, without an outcome argument.
    return predict_prepared(bundle, prepare_scores(bundle, payload, progress=progress))


def require_unseen_people(payload, bundles):
    # PSEUDOCODE: refuse to describe previously used people as an independent new evaluation cohort.
    people = {r['participant_id'] for r in payload['rows']}
    exposed = set().union(*(set(b['metadata']['previously_used_people']) for b in bundles))
    overlap = sorted(people & exposed)
    if overlap:
        raise ValueError(f'Independent evaluation refused: {len(overlap)} previously used people. Preserve stable identities.')
    return people
