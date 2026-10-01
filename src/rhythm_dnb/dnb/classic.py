"""Fixed-universe Pearson DNB; Chen 2012 / Liu 2017, DOI 10.1371/journal.pcbi.1005633.

Rows are observations; ddof=1; missing rows are jointly excluded.
Ported from the existing audited project kernel without changing pair semantics.
"""

from .statistics import _settings, _matrix, _indices, _empty_result, _statistics, _complete_rows, _DataError, _score


def dnb_components(matrix, feature_names, module, min_samples=8, epsilon=1e-8):
    """Return SD_in * PCC_in / PCC_out and its components for one named module.

    Counts describe all supplied rows and shared complete rows. Invalid scores
    are None; a denominator <= epsilon is withheld, never clipped. Modules and
    columns are sorted by name together for deterministic pair arithmetic.
    """
    # PSEUDOCODE: validate fixed dimensions -> compute dnb components -> retain invalid-data reasons.
    min_samples, epsilon = _settings(min_samples, epsilon)
    x, names, _ = _matrix(matrix, feature_names)
    indices = _indices(module, names)
    result = _empty_result(x, names, indices, 'dnb')
    try:
        sd, correlation = _statistics(_complete_rows(x, min_samples))
    except _DataError as exc:
        result['reason'] = str(exc)
        return result
    return _score(result, sd, correlation, indices, 'dnb', epsilon)
