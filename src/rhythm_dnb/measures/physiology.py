"""Measured physiological summaries; an arbitrary minimum is not resting HR."""

import numpy as np


def daily_physiology(values, *, resting_mask=None, device_resting_hr=None):
    # PSEUDOCODE: validate observed HR -> summarize -> use only identified resting epochs.
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or np.isinf(x).any() or np.any((x[np.isfinite(x)] < 20) | (x[np.isfinite(x)] > 250)):
        raise ValueError('Heart-rate input is outside the acquisition range.')
    valid = x[np.isfinite(x)]
    resting = None
    if device_resting_hr is not None:
        if isinstance(device_resting_hr, bool) or not np.isfinite(device_resting_hr) or not 20 <= device_resting_hr <= 250:
            raise ValueError('Invalid device resting HR.')
        resting = float(device_resting_hr)
    elif resting_mask is not None:
        mask = np.asarray(resting_mask)
        if mask.shape != x.shape or mask.dtype != bool:
            raise ValueError('Resting mask must be an aligned boolean array.')
        selected = x[mask & np.isfinite(x)]
        resting = float(np.median(selected)) if len(selected) else None
    return {'resting_hr_bpm': resting, 'heart_rate_bpm': float(valid.mean()) if len(valid) else None,
            'hr_sd_bpm': float(valid.std(ddof=1)) if len(valid) >= 2 else None}
