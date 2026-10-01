"""Offline fit orchestration with separate reference, development and calibration people."""

from dataclasses import dataclass
from datetime import datetime
from ..contracts import PredictionRequest, OutcomeLabel, EvaluationDay
from ..api import RhythmPredictor
from ..bundles import make_bundle
from ..dnb.reference import fit_reference
from ..measures.panel import get_panel
from ..timebase import instant
from ..research.discover import discover
from ..research.calibrate import calibrate


@dataclass(frozen=True)
class LabeledCase:
    request: PredictionRequest
    label: OutcomeLabel
    label_available_at: datetime
    outcome_protocol_id: str | None = None


def score_cases(bundle, cases, *, method='single_sample'):
    # PSEUDOCODE: send only requests to inference -> attach outcomes after scoring -> keep per-person state.
    predictor = RhythmPredictor(bundle); states, rows, responses = {}, [], []
    for case in sorted(cases, key=lambda c: (c.request.participant_id, instant(c.request.issued_at))):
        if not case.outcome_protocol_id or case.outcome_protocol_id != bundle['discovery'].get('outcome_protocol_id'):
            raise ValueError('Forecast evaluation and discovery must use the same frozen endpoint definition.')
        request = case.request
        response = predictor.predict(request, states.get(request.participant_id), method=method)
        states[request.participant_id] = response.state
        rows.append(EvaluationDay(request.participant_id, request.issued_at, response.score,
                                   case.label.value, case.label.onset, case.label_available_at, request.timezone))
        responses.append(response)
    return rows, responses


def develop(reference_candidates, pairs, calibration_cases, config, *, reference_cutoff,
            discovery_cutoff, calibration_cutoff, text_model_id=None, method='single_sample',
            minimum_calibration_events=None, calibration_events=None, calibration_monitoring=None,
            text_checkpoint=None):
    # PSEUDOCODE: freeze reference -> discover modules -> score independent calibration -> freeze policy.
    features = get_panel(config.panel_id)
    text_people = []
    if config.panel_id == 'joint12':
        from ..text.checkpoint import inspect_checkpoint
        if text_checkpoint is None:
            raise ValueError('Joint development requires the verified text checkpoint and participant registry.')
        manifest, identity = inspect_checkpoint(text_checkpoint)
        if text_model_id is not None and identity != text_model_id:
            raise ValueError('Declared text identity differs from the supplied checkpoint.')
        text_model_id = identity
        roles = manifest.get('dataset', {}).get('people', {})
        if any(not roles.get(role) for role in ('train', 'validation', 'calibration')):
            raise ValueError('Text checkpoint lacks fitting-participant provenance.')
        text_people = sorted({p for role in ('train', 'validation', 'calibration') for p in roles[role]})
    reference = fit_reference(reference_candidates, features, reference_cutoff,
                              minimum=config.reference_min_people, seed=config.seed)
    discovery = discover(pairs, config, reference, discovery_cutoff)
    provisional = make_bundle(config, reference, discovery, text_model_id=text_model_id, text_fitted_people=text_people)
    rows, _ = score_cases(provisional, calibration_cases, method=method)
    calibration = calibrate(rows, config, reference, discovery, calibration_cutoff, method=method,
                            minimum_events=minimum_calibration_events, events=calibration_events, monitoring=calibration_monitoring)
    return make_bundle(config, reference, discovery, calibration, text_model_id=text_model_id, text_fitted_people=text_people)
