"""Acquisition validation, distinct from endpoint thresholds and imputation."""

from zoneinfo import ZoneInfo
import math
from ..timebase import instant
from ..contracts import Observation


def validate_observations(observations):
    # PSEUDOCODE: check identifiers/time ordering/units and preserve all explicit missing values.
    observations = tuple(observations); ids = set()
    for row in observations:
        if not isinstance(row, Observation) or not all(isinstance(v, str) and v.strip() for v in (row.observation_id, row.participant_id, row.variable, row.unit)):
            raise ValueError('Incomplete observation identity.')
        if row.observation_id in ids:
            raise ValueError('Duplicate observation ID.')
        ids.add(row.observation_id)
        ZoneInfo(row.timezone)
        if instant(row.start) > instant(row.end) or instant(row.available_at) < instant(row.end):
            raise ValueError('Observation arrival precedes its completed measurement.')
        if row.provenance.kind not in ('observed', 'derived', 'constructed', 'synthetic', 'unknown'):
            raise ValueError('Unknown provenance class.')
        if isinstance(row.value, bool):
            raise ValueError('Boolean observation values must have an explicit categorical encoding.')
        if row.value is not None and row.unit != 'text' and (type(row.value) not in (int, float) or not math.isfinite(row.value)):
            raise ValueError('Numeric observations must be finite numbers or explicit missing values.')
        if row.unit == 'text' and row.value is not None and not isinstance(row.value, str):
            raise ValueError('Text observations require strings.')
    return observations
