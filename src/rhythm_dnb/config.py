"""Registered study choices; no data fitting or working-directory path guessing."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class StudyConfig:
    study_id: str = 'behavioral-rhythm'
    panel_id: str = 'joint12'
    baseline_days: int = 14
    outcome_window: int = 7
    outcome_min_days: int = 6
    persistence_days: int = 3
    rolling_days: int = 28
    rolling_min_days: int = 23
    max_missing_run: int = 2
    reference_min_people: int = 100
    endpoint_min_people: int = 200
    module_sizes: tuple[int, ...] = (2, 3, 4)
    module_stability: float = .7
    max_modules: int = 5
    bootstrap_repetitions: int = 10000
    permutation_repetitions: int = 9999
    discovery_alpha: float = .05
    threshold_quantile: float = .95
    alarm_consecutive: int = 2
    cooldown_days: int = 7
    horizon_days: int = 7
    min_lead_hours: int = 24
    max_false_alarms_per_30_days: float = 1.
    calibration_min_events: int = 100
    calibration_min_negative_days: int = 30
    epsilon: float = 1e-8
    seed: int = 20261001
    pair_convention: str = 'paper_k_squared'

    def __post_init__(self):
        # PSEUDOCODE: reject ambiguous types and incoherent window/threshold settings.
        for name in ('baseline_days', 'outcome_window', 'outcome_min_days', 'persistence_days',
                     'rolling_days', 'rolling_min_days', 'reference_min_people', 'max_modules',
                     'bootstrap_repetitions', 'alarm_consecutive', 'cooldown_days', 'horizon_days',
                     'min_lead_hours', 'permutation_repetitions', 'endpoint_min_people',
                     'calibration_min_events', 'calibration_min_negative_days'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if type(self.seed) is not int or self.seed < 0 or type(self.max_missing_run) is not int or self.max_missing_run < 0:
            raise ValueError('Invalid seed or missing-run length.')
        if self.reference_min_people < 9 or self.outcome_min_days > self.outcome_window or self.rolling_min_days > self.rolling_days:
            raise ValueError('Incoherent minimum sample count.')
        if self.baseline_days - self.outcome_window + 1 < self.outcome_min_days or self.outcome_min_days < 2:
            raise ValueError('Baseline cannot supply the required complete endpoint windows.')
        for name in ('module_stability', 'threshold_quantile', 'epsilon', 'max_false_alarms_per_30_days', 'discovery_alpha'):
            if type(getattr(self, name)) not in (int, float) or not math.isfinite(getattr(self, name)):
                raise ValueError(name + ' must be a finite numeric value, not boolean.')
        if not 0 < self.discovery_alpha < 1 or 1 / (self.permutation_repetitions + 1) > self.discovery_alpha:
            raise ValueError('Too few permutations to attain the discovery significance level.')
        if not 0 < self.module_stability <= 1 or not 0 < self.threshold_quantile < 1:
            raise ValueError('Invalid stability or quantile.')
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError('epsilon must be finite and positive.')
        if not math.isfinite(self.max_false_alarms_per_30_days) or self.max_false_alarms_per_30_days < 0:
            raise ValueError('Invalid false alarm budget.')
        if self.min_lead_hours > self.horizon_days * 24:
            raise ValueError('Lead time exceeds the horizon.')
        if not self.module_sizes or len(set(self.module_sizes)) != len(self.module_sizes) or any(type(k) is not int or k < 2 for k in self.module_sizes):
            raise ValueError('Invalid module sizes.')
        if self.pair_convention not in ('unique_pairs', 'paper_k_squared'):
            raise ValueError('Invalid scoring convention or domain.')
        from .measures.panel import get_panel
        if not isinstance(self.study_id, str) or not self.study_id.strip() or max(self.module_sizes) >= len(get_panel(self.panel_id)):
            raise ValueError('Invalid study identity or module larger than the feature universe.')


def load_study(path=None) -> StudyConfig:
    # PSEUDOCODE: parse explicit JSON and let the strict dataclass reject unknown fields.
    if path is None:
        return StudyConfig()
    values = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if 'module_sizes' in values:
        values['module_sizes'] = tuple(values['module_sizes'])
    return StudyConfig(**values)


def resolve_paths(cli=None, local_file=None, environment=None):
    # PSEUDOCODE: resolve CLI over environment over explicit local config; reject relative paths.
    environment = os.environ if environment is None else environment
    local = json.loads(Path(local_file).read_text(encoding='utf-8-sig')) if local_file else {}
    result = {}
    for name in ('data', 'models', 'bundles', 'runs'):
        value = (cli or {}).get(name) or environment.get('RHYTHM_DNB_' + name.upper()) or local.get(name)
        if value is not None:
            path = Path(value).expanduser()
            if not path.is_absolute():
                raise ValueError(f'{name} must be absolute.')
            result[name] = path.resolve()
    return result
