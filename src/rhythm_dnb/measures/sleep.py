"""Sleep measurements, not disorder labels. SRI needs a complete state grid."""

from zoneinfo import ZoneInfo
import numpy as np
from ..timebase import instant
from .scaling import circular_summary


def daily_sleep(episodes, zone, *, main_period=None):
    """Episodes are actual nonoverlapping asleep intervals (aware start/end).

    Duration sums actual asleep intervals. When fragmentation/naps produce
    multiple intervals, main_period=(sleep onset, final awakening) is required
    for midpoint; guessing the longest uninterrupted segment would bias S1.
    A caller must select the completed research day and explicitly confirm logging.
    """
    # PSEUDOCODE: validate ordered intervals -> sum elapsed sleep -> use the identified main-period midpoint.
    periods = sorted((instant(a), instant(b)) for a, b in episodes)
    if not periods:
        return {'sleep_midpoint_h': None, 'sleep_duration_h': None}
    for i, (a, b) in enumerate(periods):
        if b <= a or (b - a).total_seconds() > 86400 or i and a < periods[i - 1][1]:
            raise ValueError('Sleep episodes overlap or have invalid duration.')
    midpoint_h = None
    if main_period is not None:
        a, b = map(instant, main_period)
        if b <= a or (b - a).total_seconds() > 86400 or not any(a <= start < end <= b for start, end in periods):
            raise ValueError('Main sleep period must contain measured sleep and last at most 24 hours.')
    elif len(periods) == 1:
        a, b = periods[0]
    else:
        a = b = None
    if a is not None:
        midpoint = (a + (b - a) / 2).astimezone(ZoneInfo(zone))
        midpoint_h = midpoint.hour + midpoint.minute / 60 + midpoint.second / 3600
    return {'sleep_midpoint_h': midpoint_h,
            'sleep_duration_h': sum((b - a).total_seconds() for a, b in periods) / 3600}


def sleep_regularity(midpoints, minimum=6):
    # PSEUDOCODE: compute S1 circular spread without treating midnight as a discontinuity.
    return circular_summary(midpoints, minimum=minimum)[1]


def sleep_regularity_index(states, epochs_per_day):
    """Phillips 2017 SRI = 200*P(same state 24h apart)-100, range [-100,100].

    Input is an equally spaced complete UTC epoch grid of sleep=1/wake=0.
    Missing wear cannot be classified as wake. UTC 24h differs from wall day at DST.
    """
    # PSEUDOCODE: validate complete binary grid -> compare exact 24-hour-lag pairs.
    x = np.asarray(states, dtype=float)
    if type(epochs_per_day) is not int or epochs_per_day <= 0 or x.ndim != 1:
        raise ValueError('Invalid state grid.')
    if len(x) <= epochs_per_day or not np.isfinite(x).all():
        return None
    if not np.isin(x, [0, 1]).all():
        raise ValueError('Sleep state must be binary.')
    return float(200 * np.mean(x[epochs_per_day:] == x[:-epochs_per_day]) - 100)
