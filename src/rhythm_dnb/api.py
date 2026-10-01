"""Frozen inference entry point. No outcome, discovery, training or database imports."""

from copy import deepcopy
from .bundles import load_bundle, check_compatibility
from .contracts import AlarmState, PredictionResponse
from .timebase import forecast_bounds, instant
from .warning.windows import panel_vector, select_window
from .warning.score import score_modules, aggregate_score
from .warning.policy import advance


class RhythmPredictor:
    def __init__(self, bundle):
        # PSEUDOCODE: detach caller-owned state and validate the frozen artifact graph.
        self._bundle = deepcopy(bundle)
        self.config = check_compatibility(self._bundle)

    @classmethod
    def from_bundle(cls, path):
        # PSEUDOCODE: load a verified artifact and construct a predictor without fitting.
        return cls(load_bundle(path))

    def predict(self, request, state=None, *, method='single_sample'):
        # PSEUDOCODE: verify identity/cutoff -> extract causal features -> score -> advance policy.
        bundle, config = self._bundle, self.config
        if method not in ('single_sample', 'rolling'):
            raise ValueError('Unknown inference method.')
        day, _, as_of = forecast_bounds(request.issued_at, request.timezone)
        reference = bundle['reference']
        roles = reference['people'] + bundle['discovery']['people'] + bundle['text_fitted_people']
        if bundle['calibration']:
            roles += bundle['calibration']['people']
        if request.participant_id in roles:
            raise ValueError('Inference participant was used to fit this bundle.')
        cutoff = max(instant(a['cutoff']) for a in (reference, bundle['discovery'], bundle['calibration']) if a)
        if as_of <= cutoff:
            raise ValueError('Forecast must follow the last fitting cutoff.')
        # PSEUDOCODE: validate all history identities; future panels cannot affect the selected day.
        identities = set()
        for panel in request.history:
            if panel.participant_id != request.participant_id or panel.timezone != request.timezone or panel.day in identities:
                raise ValueError('Mixed or duplicate daily panels.')
            identities.add(panel.day)
            if panel.day <= day:
                for feature in panel.features:
                    if feature.name in reference['features'] and feature.name.startswith('text_') and feature.value is not None and feature.model_id != bundle['text_model_id']:
                        raise ValueError('Text feature was produced by a different model.')
        results, reasons = (), ()
        modules = bundle['discovery']['modules']
        if not modules:
            reasons = ('no_stable_modules',)
        elif method == 'single_sample':
            target = next((p for p in request.history if p.day == day), None)
            if target is None:
                reasons = ('missing_target_day',)
            else:
                values, reasons = panel_vector(target, reference['features'], as_of)
                if not reasons:
                    results = score_modules(values, reference, modules, epsilon=config.epsilon,
                                            pair_convention=config.pair_convention)
        elif method == 'rolling':
            values, reason = select_window(request.history, request.participant_id, request.timezone, day,
                                           reference['features'], as_of, days=config.rolling_days,
                                           minimum=config.rolling_min_days, max_missing_run=config.max_missing_run)
            if reason:
                reasons = (reason,)
            else:
                results = score_modules(values, reference, modules, kind=method, epsilon=config.epsilon,
                                        minimum=config.rolling_min_days)
        else:
            raise ValueError('Unknown inference method.')
        score = aggregate_score(results)
        reasons += tuple(r.reason for r in results if not r.valid)
        calibration = bundle['calibration']
        threshold = calibration['threshold'] if calibration and calibration['method'] == method else None
        warning, status, new_state = advance(state or AlarmState(), request.participant_id, bundle['id'], day,
                                            score, threshold, consecutive=config.alarm_consecutive,
                                            cooldown_days=config.cooldown_days,
                                            context=method + ':' + request.timezone)
        return PredictionResponse(request.participant_id, request.issued_at, bundle['id'], status,
                                  warning, score, results, reasons, new_state)

    def score_text(self, category, text, checkpoint, *, device='auto', base_path=None):
        # PSEUDOCODE: lazy-load a local checkpoint and require the text identity frozen in this bundle.
        from .text.predict import TextPredictor
        predictor = TextPredictor(checkpoint, device=device, base_path=base_path)
        if predictor.model_identity != self._bundle['text_model_id']:
            raise ValueError('Text checkpoint differs from the frozen bundle.')
        return predictor.predict(category, text)
