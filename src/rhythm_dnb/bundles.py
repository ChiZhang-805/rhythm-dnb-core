"""Strict, content-addressed frozen research bundles. Existing bundles are immutable."""

from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .config import StudyConfig
from .provenance import fingerprint, canonical_json
from .timebase import instant
from .dnb.modules import candidate_modules
from .measures.panel import get_panel
from .measures.scaling import transform, fit_scaler


def make_bundle(config, reference, discovery, calibration=None, *, text_model_id=None):
    # PSEUDOCODE: assemble versioned frozen artifacts, then run the same checks as loading.
    payload = {'schema_version': 2, 'study': asdict(config), 'reference': reference,
               'discovery': discovery, 'calibration': calibration, 'text_model_id': text_model_id}
    bundle = {**payload, 'id': fingerprint(payload)}
    check_compatibility(bundle)
    return bundle


def check_compatibility(bundle):
    # PSEUDOCODE: verify hashes and semantics -> enforce disjoint roles and ordered fitting cutoffs.
    if set(bundle) != {'schema_version', 'study', 'reference', 'discovery', 'calibration', 'text_model_id', 'id'}:
        raise ValueError('Unexpected bundle fields.')
    if bundle['schema_version'] != 2 or fingerprint({k: v for k, v in bundle.items() if k != 'id'}) != bundle['id']:
        raise ValueError('Bundle checksum/version mismatch.')
    study = dict(bundle['study']); study['module_sizes'] = tuple(study['module_sizes'])
    config = StudyConfig(**study)
    reference, discovery, calibration = (bundle[k] for k in ('reference', 'discovery', 'calibration'))
    for artifact in (reference, discovery, calibration):
        if artifact is not None and fingerprint({k: v for k, v in artifact.items() if k != 'id'}) != artifact.get('id'):
            raise ValueError('Artifact checksum mismatch.')
    features = get_panel(config.panel_id)
    if config.panel_id == 'joint12' and not bundle['text_model_id']:
        raise ValueError('The joint panel requires a pinned text model identity.')
    if reference.get('text_model_id') != bundle['text_model_id']:
        raise ValueError('Reference and bundle text models differ.')
    if tuple(reference['features']) != features or reference['simulation'] != config.simulation:
        raise ValueError('Reference panel/domain mismatch.')
    if tuple(reference['scaler']['features']) != features:
        raise ValueError('Scaler order mismatch.')
    raw, scaled = np.asarray(reference['raw_matrix']), np.asarray(reference['matrix'])
    if raw.shape != scaled.shape or raw.shape != (len(reference['people']), len(features)):
        raise ValueError('Reference dimensions differ.')
    if len(set(reference['people'])) != len(raw) or len(raw) < config.reference_min_people or not np.isfinite(raw).all():
        raise ValueError('Reference is not a complete independent-person set.')
    if not np.allclose(transform(raw, reference['scaler']), scaled, atol=1e-12, rtol=1e-12):
        raise ValueError('Reference scaling is inconsistent.')
    fitted_scaler = fit_scaler(raw, features)
    for key in ('mean', 'scale'):
        if not np.allclose(fitted_scaler[key], reference['scaler'][key], atol=1e-12, rtol=1e-12):
            raise ValueError('Frozen scaler was not fit on the declared reference.')
    if any(not np.isclose(v, reference['scaler']['clock_centers'][k], atol=1e-12, rtol=1e-12) for k, v in fitted_scaler['clock_centers'].items()):
        raise ValueError('Frozen circular centers differ from the declared reference.')
    if discovery['reference_id'] != reference['id'] or discovery['study_id'] != fingerprint(asdict(config)):
        raise ValueError('Discovery was fit for another reference/study.')
    if not config.simulation and not discovery.get('outcome_protocol_id'):
        raise ValueError('Real discovery lacks a frozen endpoint protocol identity.')
    modules = discovery['modules']
    if modules:
        candidate_modules(features, {str(i): m for i, m in enumerate(modules)})
    if len(modules) > config.max_modules or len({tuple(sorted(m)) for m in modules}) != len(modules):
        raise ValueError('Too many frozen modules.')
    reference_people, development_people = set(reference['people']), set(discovery['people'])
    if reference_people & development_people:
        raise ValueError('Reference/development participant overlap.')
    if instant(discovery['cutoff']) < instant(reference['cutoff']):
        raise ValueError('Discovery predates its reference.')
    if calibration is not None:
        if calibration['discovery_id'] != discovery['id'] or calibration['reference_id'] != reference['id']:
            raise ValueError('Calibration belongs to different frozen artifacts.')
        if (reference_people | development_people) & set(calibration['people']):
            raise ValueError('Calibration participant overlap.')
        if instant(calibration['cutoff']) < instant(discovery['cutoff']):
            raise ValueError('Calibration predates discovery.')
        threshold = calibration['threshold']
        if calibration['method'] not in ('single_sample', 'rolling'):
            raise ValueError('Unknown calibrated method.')
        if threshold is not None and (not np.isfinite(threshold) or threshold < 0 or not modules):
            raise ValueError('Invalid calibrated threshold.')
    return config


def save_bundle(bundle, directory):
    # PSEUDOCODE: validate -> create a new content-addressed file exclusively -> return its path.
    check_compatibility(bundle)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory / (bundle['id'] + '.json')
    with path.open('x', encoding='utf-8') as stream:
        stream.write(canonical_json(bundle) + '\n')
    return path


def load_bundle(path):
    # PSEUDOCODE: parse immutable JSON -> check hashes, role separation and numerical consistency.
    bundle = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    check_compatibility(bundle)
    return bundle
