"""Fixed-universe Pearson DNB; Chen 2012 / Liu 2017, DOI 10.1371/journal.pcbi.1005633.

Rows are observations; ddof=1; missing rows are jointly excluded.
Ported from the existing audited project kernel without changing pair semantics.
"""

from collections.abc import Mapping
import math
import numpy as np
_COMPONENTS = {"dnb": ("sd_in", "pcc_in", "pcc_out"), "sdnb": ("sed_in", "spcc_in", "spcc_out")}


class _DataError(ValueError):
    pass


def _integer(value, name, minimum=1):
    # PSEUDOCODE: reject boolean/noninteger settings and values below the required minimum.
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return int(value)


def _settings(min_samples, epsilon):
    # PSEUDOCODE: validate sample count -> require a finite positive denominator tolerance.
    minimum = _integer(min_samples, 'min_samples', 2)
    if (isinstance(epsilon, (bool, np.bool_))
            or not isinstance(epsilon, (int, float, np.integer, np.floating))):
        raise ValueError('epsilon must be finite and positive.')
    try:
        epsilon = float(epsilon)
    except (ValueError, OverflowError) as exc:
        raise ValueError('epsilon must be finite and positive.') from exc
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be finite and positive.')
    return minimum, epsilon


def _sequence(value, name):
    # PSEUDOCODE: require an ordered sequence without accepting strings, mappings or sets.
    if isinstance(value, (str, bytes, Mapping, set, frozenset)):
        raise ValueError(f'{name} must be an ordered sequence.')
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be an ordered sequence.') from exc


def _names(value, name):
    # PSEUDOCODE: require nonempty unique feature names while preserving input order.
    names = _sequence(value, name)
    if (not names or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names)):
        raise ValueError(f'{name} must contain unique nonempty strings.')
    return names


def _numeric(value, ndim, name):
    # PSEUDOCODE: validate dimensions and real numeric types before float conversion can conceal invalid inputs.
    if np.ma.isMaskedArray(value):
        raise ValueError(f'{name} must use None or NaN for missing entries, not a mask.')
    try:
        objects = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be rectangular and {ndim}-dimensional.') from exc
    if objects.ndim != ndim:
        raise ValueError(f'{name} must be {ndim}-dimensional.')
    # Check before float coercion can conceal booleans, strings, or complex values.
    if any(item is not None and (isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, float, np.integer, np.floating)))
            for item in objects.flat):
        raise ValueError(f'{name} must contain real numbers or None/NaN.')
    try:
        with np.errstate(over='ignore', invalid='ignore'):
            return objects.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} contains values not representable as float64.') from exc


def _matrix(matrix, feature_names):
    # PSEUDOCODE: match names to matrix columns -> sort both into one canonical feature order.
    x = _numeric(matrix, 2, 'matrix')
    names = _names(feature_names, 'feature_names')
    if len(names) < 3 or len(names) != x.shape[1]:
        raise ValueError('feature_names must match at least three matrix columns.')
    order = sorted(range(len(names)), key=lambda i: names[i])
    return x[:, order], tuple(names[i] for i in order), order


def _indices(module, names):
    # PSEUDOCODE: resolve module members -> require at least two nodes and a nonempty external complement.
    selected = _names(module, 'module')
    if not set(selected) <= set(names):
        raise ValueError('module contains unknown features.')
    if not 2 <= len(selected) < len(names):
        raise ValueError('module needs >=2 features and a nonempty outside complement.')
    return tuple(i for i, name in enumerate(names) if name in selected)


def _empty_result(matrix, names, indices, kind):
    # PSEUDOCODE: initialize an invalid result with actual row counts and explicit missing components.
    return {
        'valid': False, 'reason': None, 'score': None,
        **dict.fromkeys(_COMPONENTS[kind]),
        'n_total': len(matrix),
        'n_valid': int(np.isfinite(matrix).all(axis=1).sum()),
        'module': [names[i] for i in indices],
    }


def _complete_rows(matrix, min_samples, reference=False):
    # PSEUDOCODE: reject infinities -> retain jointly complete rows -> enforce the sample-count gate.
    if np.isinf(matrix).any():
        raise _DataError('infinite_reference' if reference else 'infinite_input')
    if not reference and np.isnan(matrix).all(axis=0).any():
        raise _DataError('all_missing_features')
    rows = matrix[np.isfinite(matrix).all(axis=1)]
    if len(rows) < min_samples:
        raise _DataError('insufficient_complete_reference_samples' if reference
                         else 'insufficient_complete_observations')
    return rows


def _statistics(rows, reference=False):
    # PSEUDOCODE: compute sample SD and Pearson correlations -> reject constant or nonfinite statistics.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        sd = np.std(rows, axis=0, ddof=1)
        if np.any(sd == 0) or np.any(np.all(rows == rows[0], axis=0)):
            raise _DataError('constant_reference_features' if reference else 'constant_features')
        if not np.isfinite(sd).all():
            raise _DataError('nonfinite_reference_statistics' if reference
                             else 'nonfinite_standard_deviation')
        correlation = np.corrcoef(rows, rowvar=False)
    if not np.isfinite(correlation).all():
        raise _DataError('nonfinite_reference_statistics' if reference else 'nonfinite_correlation')
    return sd, correlation


def _score(result, amplitude, correlation, indices, kind, epsilon, pair_convention='unique_pairs'):
    # PSEUDOCODE: average internal amplitudes and absolute correlations -> apply pair convention -> reject unstable denominators.
    inside = np.asarray(indices, dtype=int)
    outside = [i for i in range(len(amplitude)) if i not in indices]
    internal = correlation[np.ix_(inside, inside)][np.triu_indices(len(inside), k=1)]
    external = correlation[np.ix_(inside, outside)]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        internal_mean=float(np.mean(np.abs(internal)))
        if pair_convention=='paper_k_squared':
            internal_mean*=(len(inside)-1)/len(inside)
        values = (float(np.sum(amplitude[inside] / len(inside))),
                  internal_mean, float(np.mean(np.abs(external))))
    if not all(math.isfinite(value) for value in values):
        result['reason'] = 'nonfinite_components'
        return result
    result.update(zip(_COMPONENTS[kind], values))
    if values[2] <= epsilon:
        result['reason'] = 'near_zero_spcc_out' if kind == 'sdnb' else 'near_zero_external_correlation'
        return result
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        score = np.float64(values[0]) * values[1] / values[2]
    if not np.isfinite(score):
        result['reason'] = 'nonfinite_score'
        return result
    result.update(valid=True, score=float(score))
    return result
