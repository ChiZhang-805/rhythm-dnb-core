"""Nonparametric actigraphy: M10/L5, relative amplitude, IS and IV.

Hourly inputs are a comparable 24-bin local-clock profile. DST days require an
explicit upstream resampling policy; 23/25-hour arrays are rejected. No-wear is
NaN, never zero. Metrics require a complete accepted profile; >=20h wear alone
does not justify fabricating its missing hours.
"""

import numpy as np


def hourly_profile(hourly, wear_minutes=None, *, minimum_hour_wear=45., minimum_day_wear=1200.):
    """Convert observed counts to a 60-minute exposure rate per accepted hour.

    This explicit rate normalization is a derived measurement, not missing-hour
    imputation. An hour with less than 45 minutes wear remains missing.
    """
    # PSEUDOCODE: validate counts/wear -> normalize observed exposure -> preserve invalid hours as NaN.
    if any(type(v) not in (int, float) or not np.isfinite(v) for v in (minimum_hour_wear, minimum_day_wear)) or not 0 < minimum_hour_wear <= 60 or not 0 < minimum_day_wear <= 1440:
        raise ValueError('Invalid actigraphy exposure thresholds.')
    x = np.asarray(hourly, dtype=float)
    if x.shape != (24,) or np.isinf(x).any() or np.any(x[np.isfinite(x)] < 0):
        raise ValueError('Activity requires 24 nonnegative hourly bins.')
    if wear_minutes is not None:
        wear = np.asarray(wear_minutes, dtype=float)
        if wear.shape != (24,) or not np.isfinite(wear).all() or np.any((wear < 0) | (wear > 60)):
            raise ValueError('Invalid wear grid.')
        x = np.divide(x * 60, wear, out=np.full(24, np.nan), where=wear >= minimum_hour_wear)
        if wear.sum() < minimum_day_wear:
            x[:] = np.nan
    return x.copy()


def daily_activity(hourly, wear_minutes=None, *, minimum_hour_wear=45., minimum_day_wear=1200.):
    # PSEUDOCODE: normalize observed exposure -> compute cyclic rolling means -> summarize the complete profile.
    x = hourly_profile(hourly, wear_minutes, minimum_hour_wear=minimum_hour_wear, minimum_day_wear=minimum_day_wear)
    empty = {k: None for k in ('activity_m10_start_h', 'activity_ra', 'activity_total', 'm10', 'l5')}
    if not np.isfinite(x).all():
        return empty
    doubled = np.r_[x, x]
    means10 = np.array([doubled[i:i + 10].mean() for i in range(24)])
    means5 = np.array([doubled[i:i + 5].mean() for i in range(24)])
    m10, l5 = float(means10.max()), float(means5.min())
    maxima = np.flatnonzero(np.isclose(means10, m10, rtol=1e-12, atol=0))
    return {'activity_m10_start_h': float(maxima[0]) if m10 > l5 and len(maxima) == 1 else None,
            'activity_ra': (m10 - l5) / (m10 + l5) if m10 + l5 > 0 else None,
            'activity_total': float(x.sum()), 'm10': m10, 'l5': l5}


def activity_regularity(days, minimum=6):
    """A1=1-IS; IS=n*sum_h(mean_h-mean)^2/(24*sum_t(x_t-mean)^2)."""
    # PSEUDOCODE: select complete days -> compare clock-hour means with total variance.
    if type(minimum) is not int or minimum < 1:
        raise ValueError('Activity regularity needs a positive integer minimum.')
    x = np.asarray(days, dtype=float)
    if x.ndim != 2 or x.shape[1] != 24 or np.isinf(x).any() or np.any(x[np.isfinite(x)] < 0):
        raise ValueError('Activity history must be days by 24 hours.')
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < minimum:
        return None
    denominator = np.sum((x - x.mean()) ** 2)
    if denominator <= 0:
        return None
    stability = x.size * np.sum((x.mean(axis=0) - x.mean()) ** 2) / (24 * denominator)
    return float(1 - np.clip(stability, 0, 1))


def intradaily_variability(hourly):
    """IV for a contiguous equally spaced series; missing samples invalidate it."""
    # PSEUDOCODE: compare successive squared changes against total squared deviation.
    x = np.asarray(hourly, dtype=float)
    if x.ndim != 1 or np.isinf(x).any() or np.any(x[np.isfinite(x)] < 0):
        raise ValueError('Intradaily variability requires nonnegative activity without infinities.')
    if len(x) < 2 or not np.isfinite(x).all():
        return None
    denominator = (len(x) - 1) * np.sum((x - x.mean()) ** 2)
    return float(len(x) * np.sum(np.diff(x) ** 2) / denominator) if denominator > 0 else None
