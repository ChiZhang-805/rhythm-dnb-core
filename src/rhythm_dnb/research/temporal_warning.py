"""Causal history and matched warning learners; outcomes never enter feature construction."""

from collections import defaultdict
from datetime import timedelta

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..timebase import instant


def person_weights(groups):
    # PSEUDOCODE: distribute equal total weight to each person, not each visit.
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1. / counts[inverse] / len(counts)


def causal_history(records, values, names, history_days):
    # PSEUDOCODE: order each person's available records; summarize only the trailing calendar window.
    if type(history_days) is not int or history_days < 2 or len(set(names)) != len(names):
        raise ValueError('Invalid temporal feature settings.')
    ids = [r['record_id'] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(values):
        raise ValueError('Repeated, missing or extra records.')
    groups = defaultdict(list)
    for row in records:
        if instant(row['observed_at']) > instant(row['issued_at']):
            raise ValueError('Observation was not available at issuance.')
        v = np.asarray(values[row['record_id']], float)
        if v.shape != (len(names),) or np.isinf(v).any():
            raise ValueError('Invalid temporal feature vector.')
        groups[row['participant_id']].append(row)
    output, lineage = {}, {}
    for rows in groups.values():
        rows.sort(key=lambda r: instant(r['issued_at']))
        times = [instant(r['issued_at']) for r in rows]
        if len(times) != len(set(times)):
            raise ValueError('Repeated person/issuance timestamp.')
        for i, row in enumerate(rows):
            t = times[i]
            past = [r for r in rows[:i + 1] if instant(r['issued_at']) > t - timedelta(days=history_days)]
            x = np.array([values[r['record_id']] for r in past], float)
            mean = np.full(len(names), np.nan); sd = mean.copy(); change = mean.copy()
            for j in range(len(names)):
                observed = x[np.isfinite(x[:, j]), j]
                if len(observed):
                    mean[j] = observed.mean()
                if len(observed) > 1:
                    sd[j] = observed.std(ddof=1)
            if i and t - times[i - 1] == timedelta(days=1):
                change = np.asarray(values[row['record_id']]) - np.asarray(values[rows[i - 1]['record_id']])
            output[row['record_id']] = np.r_[x[-1], mean, sd, change]
            lineage[row['record_id']] = [r['record_id'] for r in past]
    columns = [f'{kind}:{name}' for kind in ('current', 'mean', 'sd', 'change') for name in names]
    return output, columns, lineage


def causal_smooth(records, scores, half_life):
    # PSEUDOCODE: decay past observations by actual elapsed days; missing current scores still abstain.
    if not np.isfinite(half_life) or half_life < 0:
        raise ValueError('Invalid half-life.')
    if set(scores) != {r['record_id'] for r in records}:
        raise ValueError('Smoothing scores do not match records.')
    groups = defaultdict(list); result = {}
    for row in records:
        if instant(row['observed_at']) > instant(row['issued_at']):
            raise ValueError('Future observation used for smoothing.')
        groups[row['participant_id']].append(row)
    seen_ids = set()
    for rows in groups.values():
        rows.sort(key=lambda r: instant(r['issued_at']))
        previous = None; numerator = denominator = 0.
        for row in rows:
            rid = row['record_id']; t = instant(row['issued_at']); score = scores[rid]
            if rid in seen_ids or previous is not None and t <= previous:
                raise ValueError('Repeated record or issuance timestamp.')
            seen_ids.add(rid)
            if score is not None and (not np.isfinite(score) or score < 0):
                raise ValueError('Risk scores must be nonnegative and finite.')
            decay = 0. if half_life == 0 or previous is None else 2. ** (-(t - previous).total_seconds() / 86400 / half_life)
            numerator *= decay; denominator *= decay
            if score is not None:
                numerator += score; denominator += 1
            result[rid] = numerator / denominator if score is not None else None
            previous = t
    return result


def grouped_splits(x, y, groups, folds, seed):
    # PSEUDOCODE: keep every person's visits together; require both outcomes on each side.
    y, groups = np.asarray(y), np.asarray(groups)
    if set(y) != {0, 1} or len(x) != len(y) or len(groups) != len(y):
        raise ValueError('Invalid development outcomes or alignment.')
    splits = list(StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(x, y, groups))
    for a, b in splits:
        if set(groups[a]) & set(groups[b]) or set(y[a]) != {0, 1} or set(y[b]) != {0, 1}:
            raise ValueError('Invalid person-isolated development fold.')
    return splits


def candidates(config):
    # PSEUDOCODE: keep the same small, frozen capacity grid for both matched feature arms.
    result = [{'family': 'ridge_logistic', 'c': c} for c in config['logistic_c']]
    result += [{'family': 'boosted_tree', 'leaves': leaves, 'l2': l2}
               for leaves in config['tree_leaves'] for l2 in config['tree_l2']]
    return result


def fit_candidate(x, y, groups, spec, config, seed):
    # PSEUDOCODE: fit all learned preprocessing inside the supplied development split.
    if np.isinf(x).any() or len(x) != len(y) or set(y) != {0, 1}:
        raise ValueError('Invalid learner input.')
    w = person_weights(groups) * len(groups)
    if spec['family'] == 'ridge_logistic':
        model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                              StandardScaler(), LogisticRegression(C=spec['c'], max_iter=2000, random_state=seed))
        model.fit(x, y, logisticregression__sample_weight=w)
        if np.any(model[-1].n_iter_ >= model[-1].max_iter):
            raise ValueError('Logistic learner did not converge.')
    elif spec['family'] == 'boosted_tree':
        model = HistGradientBoostingClassifier(max_leaf_nodes=spec['leaves'], l2_regularization=spec['l2'],
            max_iter=config['tree_iterations'], learning_rate=config['tree_learning_rate'],
            min_samples_leaf=config['tree_min_samples_leaf'], early_stopping=False, random_state=seed)
        model.fit(x, y, sample_weight=w)
    else:
        raise ValueError('Unknown learner family.')
    return model


def select_learner(x, y, groups, config, seed):
    # PSEUDOCODE: choose capacity by person-weighted out-of-fold development loss, never test performance.
    x, y, groups = np.asarray(x, float), np.asarray(y, int), np.asarray(groups)
    splits = grouped_splits(x, y, groups, config['inner_folds'], seed)
    trials = []
    for spec in candidates(config):
        predictions = np.full(len(y), np.nan); losses = []
        for a, b in splits:
            fitted = fit_candidate(x[a], y[a], groups[a], spec, config, seed)
            p = np.clip(fitted.predict_proba(x[b])[:, 1], np.finfo(float).eps, 1 - np.finfo(float).eps)
            predictions[b] = p
            losses.append(float(person_weights(groups[b]) @ (-y[b] * np.log(p) - (1-y[b]) * np.log1p(-p))))
        if not np.isfinite(predictions).all():
            raise ValueError('Incomplete development predictions.')
        trials.append({'spec': spec, 'mean_log_loss': float(np.mean(losses)), 'fold_log_loss': losses})
    selected = min(range(len(trials)), key=lambda i: (round(trials[i]['mean_log_loss'], 12), i))
    spec = trials[selected]['spec']
    fitted = fit_candidate(x, y, groups, spec, config, seed)
    receipt = {'selected': spec, 'trials': trials, 'training_people': sorted(set(groups.tolist())),
               'inner_splits': [{'fit_people': sorted(set(groups[a])), 'validation_people': sorted(set(groups[b]))}
                                for a, b in splits]}
    return fitted, receipt


def select_smoothing(vectors, y, groups, config, seed):
    # PSEUDOCODE: compare only development ranking across the same person folds; ties retain shorter memory.
    y, groups = np.asarray(y), np.asarray(groups)
    first = next(iter(vectors.values()))
    splits = grouped_splits(first, y, groups, config['inner_folds'], seed)
    trials = []
    for half_life in config['half_lives_days']:
        scores = np.asarray(vectors[half_life], float)
        if not np.isfinite(scores).all():
            raise ValueError('Incomplete development smoothing scores.')
        areas = [float(roc_auc_score(y[b], scores[b], sample_weight=person_weights(groups[b]))) for _, b in splits]
        trials.append({'half_life': half_life, 'mean_auc': float(np.mean(areas)), 'fold_auc': areas})
    chosen = max(trials, key=lambda t: (round(t['mean_auc'], 12), -t['half_life']))
    return {'half_life': chosen['half_life'], 'trials': trials, 'training_people': sorted(set(groups.tolist()))}
