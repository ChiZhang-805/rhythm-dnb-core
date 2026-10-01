"""Fixed-universe Pearson DNB; Chen 2012 / Liu 2017, DOI 10.1371/journal.pcbi.1005633.

Rows are observations; ddof=1; missing rows are jointly excluded.
Ported from the existing audited project kernel without changing pair semantics.
"""

import numpy as np
from .statistics import _settings, _matrix, _numeric, _indices, _empty_result, _complete_rows, _statistics, _DataError, _score


def sdnb_components(sample, reference, feature_names, module, min_samples=9, epsilon=1e-8,
                    pair_convention='unique_pairs'):
    """Score one target against an unchanged reference in the same units/order.

    min_samples, n_total and n_valid refer to reference rows, excluding the
    target. The target must be a complete 1D vector. Reference rows are dropped
    across the fixed universe before appending exactly one target to a new array.
    The caller selects the reference and excludes self/future/held-out records.
    The low-level default retains legacy unique pairs for numerical compatibility;
    the project pipeline explicitly selects paper_k_squared for Eq. (6).
    """
    # PSEUDOCODE: compare unchanged reference correlations with reference-plus-target -> score target deviation and network perturbation.
    min_samples, epsilon = _settings(min_samples, epsilon)
    if pair_convention not in ('unique_pairs','paper_k_squared'):
        raise ValueError('Unknown sDNB pair convention.')
    r, names, order = _matrix(reference, feature_names)
    x = _numeric(sample, 1, 'sample')
    if x.shape != (len(names),):
        raise ValueError('sample must match the fixed reference feature dimension.')
    x = x[order]
    indices = _indices(module, names)
    result = _empty_result(r, names, indices, 'sdnb')
    result['pair_convention']=pair_convention
    try:
        rows = _complete_rows(r, min_samples, reference=True)
        _, base = _statistics(rows, reference=True)
        if np.isinf(x).any():
            raise _DataError('infinite_sample')
        if np.isnan(x).any():
            raise _DataError('missing_sample')
        with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
            sed = np.abs(x - np.mean(rows, axis=0))
            delta = np.corrcoef(np.vstack((rows, x)), rowvar=False) - base
        if not np.isfinite(sed).all() or not np.isfinite(delta).all():
            raise _DataError('nonfinite_sample_statistics')
    except _DataError as exc:
        result['reason'] = str(exc)
        return result
    return _score(result, sed, delta, indices, 'sdnb', epsilon,pair_convention)


def sample_network(sample, reference, feature_names, *, deviation_quantile=.75, alpha=.05,
                   correction='bh', epsilon=1e-8):
    """Exploratory Liu-2017 network screening with explicit choices left open by the paper.

    Z=deltaPCC*(n-1)/(1-PCC_n^2); two-sided normal approximation. BH correction is
    a conservative project choice, not a claim of exact reproduction of every
    paper setting. This target-adaptive network is not the frozen primary model.
    """
    # PSEUDOCODE: form single-sample perturbations -> test high-deviation edges -> return signed network.
    from math import erfc, sqrt
    if not 0 < deviation_quantile < 1 or not 0 < alpha < 1 or correction not in ('bh', 'none'):
        raise ValueError('Invalid network screening settings.')
    rows, names, order = _matrix(reference, feature_names)
    rows = _complete_rows(rows, 9, reference=True)
    x = _numeric(sample, 1, 'sample')
    if x.shape != (len(names),) or not np.isfinite(x).all():
        raise ValueError('A complete target sample is required.')
    x = x[order]
    _, corr = _statistics(rows, reference=True)
    delta = np.corrcoef(np.vstack((rows, x)), rowvar=False) - corr
    deviation = np.abs(x - rows.mean(axis=0))
    selected = np.flatnonzero(deviation >= np.quantile(deviation, deviation_quantile))
    edges = []
    for offset, i in enumerate(selected):
        for j in selected[offset + 1:]:
            denominator = 1 - corr[i, j] ** 2
            if denominator <= epsilon:
                continue
            z = float(delta[i, j] * (len(rows) - 1) / denominator)
            edges.append({'source': names[i], 'target': names[j], 'delta': float(delta[i, j]),
                          'z': z, 'p': erfc(abs(z) / sqrt(2))})
    order_p = sorted(range(len(edges)), key=lambda i: edges[i]['p'])
    adjusted = 1.
    for rank in range(len(order_p), 0, -1):
        edge = edges[order_p[rank - 1]]
        adjusted = min(adjusted, edge['p'] * len(edges) / rank)
        edge['q'] = adjusted if correction == 'bh' else edge['p']
        edge['significant'] = edge['q'] <= alpha
    return {'features': list(names), 'selected': [names[i] for i in selected],
            'deviation': deviation.tolist(), 'delta': delta.tolist(), 'edges': edges,
            'method': 'Liu2017_normal_approximation', 'correction': correction}


def discover_sample_modules(sample, reference, feature_names, *, deviation_quantile=.75,
                            alpha=.05, correction='bh', distance_cut=1.9):
    """Paper-style single-linkage exploration; cut and screening thresholds are explicit."""
    # PSEUDOCODE: screen edges -> cluster distance 2-|deltaPCC| -> score nontrivial clusters.
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    if not 0 <= distance_cut < 2:
        raise ValueError('distance_cut must be in [0,2).')
    network = sample_network(sample, reference, feature_names, deviation_quantile=deviation_quantile,
                             alpha=alpha, correction=correction)
    selected = network['selected']; size = len(selected)
    if size < 2:
        return {'network': network, 'modules': [], 'best': None}
    distances = np.full((size, size), 2.); np.fill_diagonal(distances, 0.)
    for edge in network['edges']:
        if edge['significant']:
            i, j = selected.index(edge['source']), selected.index(edge['target'])
            distances[i, j] = distances[j, i] = max(0., 2 - abs(edge['delta']))
    clusters = fcluster(linkage(squareform(distances), method='single'), distance_cut, criterion='distance')
    results = []
    for label in sorted(set(clusters)):
        module = [name for name, group in zip(selected, clusters) if group == label]
        if 2 <= len(module) < len(feature_names):
            results.append(sdnb_components(sample, reference, feature_names, module, pair_convention='paper_k_squared'))
    valid = [r for r in results if r['valid']]
    return {'network': network, 'modules': results, 'best': max(valid, key=lambda r: r['score']) if valid else None}
