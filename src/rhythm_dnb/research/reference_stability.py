"""Reference-person resampling of the unchanged personal DNB formula and matched deviation controls."""

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def network_changes(reference, targets):
    # PSEUDOCODE: update centered cross-products with one target; match reference-plus-target Pearson exactly.
    r, x = np.asarray(reference, float), np.asarray(targets, float)
    if r.ndim != 2 or x.ndim != 2 or r.shape[1] != x.shape[1] or len(r) < 9 or r.shape[1] < 3:
        raise ValueError('Invalid reference/target shape.')
    if not np.isfinite(r).all() or not np.isfinite(x).all():
        raise ValueError('Reference resampling requires complete finite observations.')
    mean = r.mean(0); centered = r - mean; scatter = centered.T @ centered
    diagonal = np.diag(scatter)
    if np.any(diagonal <= 1e-12):
        raise ValueError('Degenerate resampled reference.')
    base = scatter / np.sqrt(diagonal[:, None] * diagonal[None, :])
    difference = x - mean
    updated = scatter + len(r) / (len(r)+1) * difference[:, :, None] * difference[:, None, :]
    variance = np.diagonal(updated, axis1=1, axis2=2)
    correlation = updated / np.sqrt(variance[:, :, None] * variance[:, None, :])
    delta = correlation - base
    for matrix in delta:
        np.fill_diagonal(matrix, 0.)
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite reference perturbations.')
    return np.abs(difference), delta


def personal_components(deviation, delta, epsilon):
    # PSEUDOCODE: score all proper single-linkage branches using K-squared means; preserve invalid-module abstention.
    deviation, delta = np.asarray(deviation, float), np.asarray(delta, float)
    n = len(deviation)
    if (deviation.ndim != 1 or delta.shape != (n, n) or n < 3 or not np.isfinite(deviation).all()
            or not np.isfinite(delta).all() or np.any(deviation < 0) or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError('Invalid DNB components.')
    if not np.allclose(delta, delta.T, atol=1e-12, rtol=0) or np.any(np.diag(delta) != 0):
        raise ValueError('Expected symmetric perturbation with zero diagonal.')
    distance = 2 - np.abs((delta + delta.T)/2); np.fill_diagonal(distance, 0)
    if np.any(distance < -1e-12):
        raise ValueError('Pearson perturbation exceeds its range.')
    tree = linkage(squareform(np.maximum(distance, 0), checks=True), method='single')
    branches = [[i] for i in range(n)]; best = None
    for a, b, _, _ in tree[:-1]:
        inside = sorted(branches[int(a)] + branches[int(b)]); branches.append(inside)
        outside = [i for i in range(n) if i not in inside]
        amplitude = float(deviation[inside].mean())
        within = float(np.abs(delta[np.ix_(inside, inside)]).mean())
        between = float(np.abs(delta[np.ix_(inside, outside)]).mean())
        if between <= epsilon:
            return None
        ratio = within / between; score = amplitude * ratio
        if not np.isfinite(score):
            return None
        if best is None or score > best['score']:
            best = {'score': score, 'ratio': ratio, 'module': inside}
    return best


def reference_draws(people, replicates, seed):
    # PSEUDOCODE: sample whole stable people once per draw and reuse the same draw across all targets.
    if len(people) < 9 or len(set(people)) != len(people) or type(replicates) is not int or replicates < 2:
        raise ValueError('Reference draws require independent people and at least two replicates.')
    draws = np.random.default_rng(seed).integers(len(people), size=(replicates, len(people)))
    if any(len(set(indices)) < 9 for indices in draws):
        raise ValueError('A reference draw has too few distinct people; do not silently replace it.')
    return draws


def resample_scores(reference, targets, draws, epsilon, *, progress=None):
    # PSEUDOCODE: recompute unchanged personal DNB and mean deviation for each shared reference draw.
    r, x, draws = np.asarray(reference, float), np.asarray(targets, float), np.asarray(draws)
    if (r.ndim != 2 or x.ndim != 2 or draws.ndim != 2 or draws.shape[1] != len(r)
            or not np.issubdtype(draws.dtype, np.integer) or np.any(draws < 0) or np.any(draws >= len(r))):
        raise ValueError('Invalid resampling indices or matrices.')
    output = {k: np.full((len(draws), len(x)), np.nan) for k in ('dnb', 'network_ratio', 'deviation')}
    memberships = np.zeros((len(x), r.shape[1]), dtype=int)
    for b, indices in enumerate(draws):
        if len(set(indices)) < 9:
            raise ValueError('Too few distinct reference people.')
        deviation, delta = network_changes(r[indices], x)
        output['deviation'][b] = deviation.mean(axis=1)
        for j, (distance, matrix) in enumerate(zip(deviation, delta)):
            parts = personal_components(distance, matrix, epsilon)
            if parts is not None:
                output['dnb'][b, j] = parts['score']; output['network_ratio'][b, j] = parts['ratio']
                memberships[j, parts['module']] += 1
        if progress is not None:
            progress(b+1, len(draws))
    return output, memberships


def summarize_draws(values):
    # PSEUDOCODE: report median and interquartile spread only when every prescribed draw was evaluable.
    x = np.asarray(values, float)
    if x.ndim != 2 or len(x) < 2 or np.isinf(x).any() or np.any(x[np.isfinite(x)] < 0):
        raise ValueError('Invalid reference sensitivity values.')
    valid = np.isfinite(x).all(axis=0)
    median = np.full(x.shape[1], np.nan); spread = median.copy()
    if valid.any():
        lower, middle, upper = np.quantile(x[:, valid], [.25, .5, .75], axis=0)
        median[valid] = middle; spread[valid] = upper-lower
    return median, spread, valid
