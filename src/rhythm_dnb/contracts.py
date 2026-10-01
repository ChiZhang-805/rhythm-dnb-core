"""Immutable scientific contracts. Forecast inputs deliberately contain no truth.

Times must be offset-aware. A missing value carries a reason; it is never zero.
Scores are continuous statistics, not probabilities or clinical diagnoses.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal
from .definitions import MEASUREMENT_ID


@dataclass(frozen=True)
class Provenance:
    kind: Literal['observed', 'derived', 'constructed', 'synthetic', 'unknown']
    source_id: str
    source_hash: str = ''
    parent_ids: tuple[str, ...] = ()
    method: str = ''
    independent: bool = True


@dataclass(frozen=True)
class Observation:
    observation_id: str
    participant_id: str
    variable: str
    value: float | str | None
    unit: str
    start: datetime
    end: datetime
    available_at: datetime
    timezone: str
    provenance: Provenance


@dataclass(frozen=True)
class FeatureValue:
    name: str
    value: float | None
    unit: str
    available_at: datetime
    provenance: Provenance
    reason: str | None = None
    coverage: float = 1.0
    measurement_id: str = MEASUREMENT_ID
    measured_until: datetime | None = None
    model_id: str | None = None


@dataclass(frozen=True)
class DailyPanel:
    participant_id: str
    day: date
    timezone: str
    features: tuple[FeatureValue, ...]


@dataclass(frozen=True)
class PredictionRequest:
    participant_id: str
    issued_at: datetime
    timezone: str
    history: tuple[DailyPanel, ...]


@dataclass(frozen=True)
class DNBResult:
    valid: bool
    score: float | None
    amplitude: float | None
    internal: float | None
    external: float | None
    module: tuple[str, ...]
    n_valid: int
    reason: str | None = None


@dataclass(frozen=True)
class AlarmState:
    participant_id: str = ''
    bundle_id: str = ''
    last_day: date | None = None
    last_alarm_day: date | None = None
    consecutive: int = 0
    context: str = ''


@dataclass(frozen=True)
class PredictionResponse:
    participant_id: str
    issued_at: datetime
    bundle_id: str
    status: str
    warning: int | None
    score: float | None
    modules: tuple[DNBResult, ...]
    reasons: tuple[str, ...]
    state: AlarmState
    interpretation: str = 'research_early_warning_not_diagnosis'


@dataclass(frozen=True)
class OutcomeAssessment:
    participant_id: str
    day: date
    abnormal_domains: tuple[str, ...]
    evaluable: bool
    available_at: datetime


@dataclass(frozen=True)
class OutcomeEvent:
    participant_id: str
    onset: datetime
    confirmed_at: datetime
    definition: str = 'sleep-eating-activity'


@dataclass(frozen=True)
class OutcomeLabel:
    value: int | None
    reason: str
    onset: datetime | None = None


@dataclass(frozen=True)
class EvaluationDay:
    participant_id: str
    issued_at: datetime
    score: float | None
    label: int | None
    onset: datetime | None = None
    label_available_at: datetime | None = None
    timezone: str = 'UTC'


@dataclass(frozen=True)
class MonitoringPeriod:
    participant_id: str
    first_issue_day: date
    last_issue_day: date
    timezone: str


def parse_panel(payload):
    # PSEUDOCODE: reconstruct nested immutable contracts and reject unknown fields through constructors.
    values = dict(payload)
    values['day'] = date.fromisoformat(values['day'])
    features = []
    for raw in values['features']:
        if not raw.get('measurement_id'):
            raise ValueError('Serialized features require their original measurement identity; rebuild missing identities from source.')
        item = dict(raw); provenance = dict(item['provenance'])
        provenance['parent_ids'] = tuple(provenance.get('parent_ids', ()))
        item['provenance'] = Provenance(**provenance)
        item['available_at'] = datetime.fromisoformat(item['available_at'])
        if item.get('measured_until'):
            item['measured_until'] = datetime.fromisoformat(item['measured_until'])
        features.append(FeatureValue(**item))
    values['features'] = tuple(features)
    return DailyPanel(**values)


def parse_request(payload):
    # PSEUDOCODE: decode a forecast-only object; truth or split keys are rejected as unknown fields.
    values = dict(payload)
    values['issued_at'] = datetime.fromisoformat(values['issued_at'])
    values['history'] = tuple(parse_panel(p) for p in values['history'])
    return PredictionRequest(**values)


def parse_observation(payload):
    # PSEUDOCODE: reconstruct explicit timestamps and provenance without inferring missing metadata.
    values = dict(payload); provenance = dict(values['provenance'])
    provenance['parent_ids'] = tuple(provenance.get('parent_ids', ()))
    values['provenance'] = Provenance(**provenance)
    for key in ('start', 'end', 'available_at'):
        values[key] = datetime.fromisoformat(values[key])
    return Observation(**values)
