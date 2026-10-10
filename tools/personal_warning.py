"""Audit personal modules and fit nonnegative combination weights using development people only."""

from collections import defaultdict
import csv

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.special import expit
from sklearn.model_selection import StratifiedGroupKFold

from rhythm_dnb.dnb.single_sample import sdnb_components
from tools.run_dnbr_ablation import matrix
from tools.run_dnbr_warning import read_json, read_scores


def branches(delta, names):
    # PSEUDOCODE: independently enumerate the proper single-linkage subtrees in canonical feature order.
    distance = 2 - np.abs((delta + delta.T) / 2); np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=True), method='single')
    groups = [[name] for name in names]
    for a, b, _, _ in tree:
        groups.append(sorted(groups[int(a)] + groups[int(b)]))
    return groups[len(names):-1]


def audit_personal(inputs, scored, rows):
    # PSEUDOCODE: verify each target-specific topology, each R arithmetic component and the selected maximum.
    settings = read_json(inputs / 'manifest.json')
    names, _, reference = matrix(inputs / 'reference.csv')
    target_names, records, targets = matrix(inputs / 'targets.csv')
    if names != target_names or records != [r['record_id'] for r in rows]:
        raise ValueError('Personal score inputs are not aligned.')
    order = np.argsort(names); names = [names[i] for i in order]
    reference = reference[:, order]; targets = targets[:, order]
    base = np.corrcoef(reference, rowvar=False)
    parts = defaultdict(list)
    with (scored / 'components.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            parts[row['record_id']].append(row)
    if set(parts) != set(records):
        raise ValueError('Missing or extra component records.')
    scores = read_scores(scored / 'scores.csv', records)
    checked, error = 0, 0.
    for record, target in zip(records, targets):
        delta = np.corrcoef(np.vstack([reference, target]), rowvar=False) - base
        expected_modules = {tuple(m) for m in branches(delta, names)}
        observed_modules = [tuple(r['genes'].split(',')) for r in parts[record]]
        if set(observed_modules) != expected_modules or len(observed_modules) != len(expected_modules):
            raise ValueError('R/Python personal tree topology differs.')
        values = []
        for row in parts[record]:
            expected = sdnb_components(target, reference, names, row['genes'].split(','),
                epsilon=settings['epsilon'], pair_convention='paper_k_squared')
            if (row['valid'].upper() == 'TRUE') != expected['valid']:
                raise ValueError('Personal component validity differs.')
            for name in ('sED_in', 'sPCC_in', 'sPCC_out', 'score'):
                actual = float(row[name]) if row[name] else None
                value = expected[name.lower()]
                if value is None:
                    if actual is not None:
                        raise ValueError('Undefined component was filled.')
                else:
                    np.testing.assert_allclose(actual, value, rtol=1e-9, atol=1e-10)
                    error = max(error, abs(actual - value))
            values.append(expected['score'] if expected['valid'] else None)
            checked += 1
        best = max(values) if values and all(v is not None for v in values) else None
        if best is None:
            if scores[record] is not None:
                raise ValueError('Invalid personal module must abstain.')
        else:
            np.testing.assert_allclose(scores[record], best, rtol=1e-9, atol=1e-10)
    return scores, {'checked': checked, 'maximum_absolute_error': error, 'topologies_checked': len(records)}


def weights(groups):
    # PSEUDOCODE: give each person equal total influence regardless of their number of recorded days.
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1 / counts[inverse] / len(unique)


def fit_model(x, y, groups, penalty):
    # PSEUDOCODE: fit training-only log scale and bounded ridge-logistic slopes, then check optimization convergence.
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    if (x.ndim != 2 or len(y) != len(x) or len(groups) != len(x) or not np.isfinite(x).all() or
            np.any(x < 0) or set(y) != {0., 1.} or not np.isfinite(penalty) or penalty <= 0):
        raise ValueError('Invalid combination-model training inputs.')
    w = weights(groups); transformed = np.log1p(x)
    center = (w[:, None] * transformed).sum(axis=0)
    scale = np.sqrt((w[:, None] * (transformed - center)**2).sum(axis=0))
    constant = scale <= 1e-12; scale[constant] = 1
    z = np.column_stack([np.ones(len(x)), (transformed - center) / scale])

    def objective(beta):
        # PSEUDOCODE: evaluate stable weighted logistic loss and its analytic gradient plus slope shrinkage.
        eta = z @ beta
        loss = w @ (np.logaddexp(0, eta) - y * eta) + penalty / 2 * (beta[1:] @ beta[1:])
        gradient = z.T @ (w * (expit(eta) - y)) + np.r_[0., penalty * beta[1:]]
        return float(loss), gradient

    initial = np.zeros(z.shape[1]); prevalence = w @ y
    initial[0] = np.log(prevalence / (1 - prevalence))
    bounds = [(None, None)] + [(0, 0) if c else (0, None) for c in constant]
    fitted = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError('Combination fit failed: ' + str(fitted.message))
    return {'center': center.tolist(), 'scale': scale.tolist(), 'intercept': float(fitted.x[0]),
            'coefficients': fitted.x[1:].tolist(), 'penalty': float(penalty),
            'training_people': sorted(set(groups.tolist())), 'iterations': int(fitted.nit),
            'objective': float(fitted.fun)}


def predict_model(model, x):
    # PSEUDOCODE: apply the frozen development transform and slopes; leave incomplete rows unknown.
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(model['coefficients']) or np.isinf(x).any() or np.any(x < 0):
        raise ValueError('Invalid combination prediction inputs.')
    complete = np.isfinite(x).all(axis=1)
    values = np.full(len(x), np.nan)
    transformed = (np.log1p(x[complete]) - np.asarray(model['center'])) / np.asarray(model['scale'])
    values[complete] = expit(model['intercept'] + transformed @ np.asarray(model['coefficients']))
    return values


def select_model(x, y, groups, config, seed):
    # PSEUDOCODE: isolate people in inner folds; fit preprocessing inside each fold; choose only by development loss.
    x, y, groups = np.asarray(x, float), np.asarray(y, int), np.asarray(groups)
    if not np.isfinite(x).all():
        raise ValueError('Development predictors incomplete; refuse selective training.')
    splitter = StratifiedGroupKFold(n_splits=config['inner_folds'], shuffle=True, random_state=seed)
    splits = list(splitter.split(x, y, groups))
    candidates, identities = [], []
    for a, b in splits:
        if set(groups[a]) & set(groups[b]):
            raise ValueError('Inner development people overlap.')
        identities.append({'fit_people': sorted(set(groups[a])), 'validation_people': sorted(set(groups[b]))})
    for penalty in config['penalties']:
        loss_parts = []
        for a, b in splits:
            model = fit_model(x[a], y[a], groups[a], penalty)
            p = np.clip(predict_model(model, x[b]), np.finfo(float).eps, 1 - np.finfo(float).eps)
            loss_parts.append(float(weights(groups[b]) @ (-y[b] * np.log(p) - (1-y[b]) * np.log1p(-p))))
        candidates.append({'penalty': penalty, 'mean_loss': float(np.mean(loss_parts)), 'fold_losses': loss_parts})
    chosen = min(candidates, key=lambda r: (round(r['mean_loss'], 12), -r['penalty']))
    return {**fit_model(x, y, groups, chosen['penalty']), 'selection': candidates, 'inner_splits': identities}
