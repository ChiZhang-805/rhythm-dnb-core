"""Caloric timing requires confirmed complete logs; fasting is not missingness."""

from zoneinfo import ZoneInfo
import numpy as np
from ..timebase import instant
from .scaling import circular_summary


def daily_eating(events, zone, *, complete):
    """events: aware timestamp, energy kcal. Zero-calorie drinks are excluded."""
    # PSEUDOCODE: require complete logging -> reject invalid calories -> order caloric events.
    if type(complete) is not bool:
        raise ValueError('Logging completeness must be explicit.')
    parsed = []
    for timestamp, kcal in events:
        if isinstance(kcal, bool) or not np.isfinite(kcal) or kcal < 0:
            raise ValueError('Invalid energy value.')
        if kcal > 0:
            parsed.append(instant(timestamp))
    if not complete or not parsed:
        return {'first_caloric_h': None, 'last_caloric_h': None}
    clocks = []
    for timestamp in (min(parsed), max(parsed)):
        local = timestamp.astimezone(ZoneInfo(zone))
        clocks.append(local.hour + local.minute / 60 + local.second / 3600)
    return dict(zip(('first_caloric_h', 'last_caloric_h'), clocks))


def eating_regularity(first, last, minimum=6):
    # PSEUDOCODE: use paired complete days -> take the larger first/last circular spread.
    x = np.asarray([first, last], dtype=float).T
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError('First and last meal series must align.')
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < minimum:
        return None
    values = [circular_summary(x[:, j], minimum=minimum)[1] for j in (0, 1)]
    return None if any(v is None for v in values) else max(values)
