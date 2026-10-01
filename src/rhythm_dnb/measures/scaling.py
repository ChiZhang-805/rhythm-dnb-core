"""Reference-only scaling, preserving daily dispersion and clock geometry."""

import numpy as np
from .panel import CLOCKS


def circular_summary(values, *, period=24., minimum=1):
    """Return circular mean and circular SD in the input unit; undefined for R≈0."""
    # PSEUDOCODE: retain finite values -> map to unit circle -> convert resultant to SD.
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < minimum or period <= 0:
        return None, None
    angles = x * 2 * np.pi / period
    vector = np.mean(np.exp(1j * angles))
    length = abs(vector)
    if length <= 1e-12:
        return None, None
    mean = float(np.angle(vector) * period / (2 * np.pi) % period)
    mean %= period  # Round-off of a tiny negative angle can produce exactly period.
    sd = float(np.sqrt(-2 * np.log(min(1., length))) * period / (2 * np.pi))
    return mean, sd


def fit_scaler(matrix, features):
    """Fit on complete independent reference rows; never standardize a target window."""
    # PSEUDOCODE: fit fixed circular anchors -> unwrap -> estimate reference mean and SD.
    x = np.asarray(matrix, dtype=float).copy()
    if x.ndim != 2 or x.shape[1] != len(features) or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Scaler requires a complete reference matrix.')
    clocks = {}
    for j, name in enumerate(features):
        if name in CLOCKS:
            center, _ = circular_summary(x[:, j])
            if center is None:
                raise ValueError('Undefined circular reference center: ' + name)
            clocks[name] = center
            x[:, j] = (x[:, j] - center + 12) % 24 - 12
    scale = np.std(x, axis=0, ddof=1)
    if not np.isfinite(scale).all() or np.any(scale <= 1e-12):
        raise ValueError('Constant or undefined reference feature.')
    return {'features': list(features), 'clock_centers': clocks,
            'mean': np.mean(x, axis=0).tolist(), 'scale': scale.tolist()}


def transform(matrix, scaler):
    # PSEUDOCODE: copy input -> apply frozen clock centers and scale -> keep missing values.
    x = np.asarray(matrix, dtype=float).copy()
    if x.ndim not in (1, 2) or x.shape[-1] != len(scaler['features']):
        raise ValueError('Feature dimension differs from the frozen scaler.')
    scale, mean = np.asarray(scaler['scale']), np.asarray(scaler['mean'])
    if scale.shape != (x.shape[-1],) or mean.shape != scale.shape or not np.isfinite(scale).all() or not np.isfinite(mean).all() or np.any(scale <= 1e-12):
        raise ValueError('Invalid frozen scale or center.')
    if set(scaler['clock_centers']) != set(scaler['features']) & set(CLOCKS) or any(not np.isfinite(v) or not 0 <= v < 24 for v in scaler['clock_centers'].values()):
        raise ValueError('Invalid frozen clock centers.')
    for j, name in enumerate(scaler['features']):
        if name in scaler['clock_centers']:
            x[..., j] = (x[..., j] - scaler['clock_centers'][name] + 12) % 24 - 12
    return (x - np.asarray(scaler['mean'])) / np.asarray(scaler['scale'])
