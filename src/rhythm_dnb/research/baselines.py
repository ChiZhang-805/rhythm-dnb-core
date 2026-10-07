"""Development-only choice between linear and nonlinear comparators, grouped by person."""

import numpy as np


def fit_baselines(x, y, people, *, seed=20261001, folds=5):
    """Select hyperparameters by grouped CV average precision; held-out test is absent.

    Input columns are prespecified reference deviations/trends, never future
    labels or window-standardized DNB features. Missing values are training-fold
    median imputed with indicators. All transformations fit within each fold.
    """
    # PSEUDOCODE: build comparable pipelines -> grouped CV -> tune linear/spline/tree models on development only.
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, SplineTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
    x, y, people = np.asarray(x, dtype=float), np.asarray(y), np.asarray(people)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(people) or set(y) != {0, 1} or len(set(people)) < folds:
        raise ValueError('Insufficient grouped two-class development data.')
    cv = list(StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(x, y, people))
    if any(len(set(y[a])) != 2 or len(set(y[b])) != 2 for a, b in cv):
        raise ValueError('Each grouped fold must contain both outcome classes.')
    candidates = {
        'logistic': (Pipeline([('impute', SimpleImputer(add_indicator=True)), ('scale', StandardScaler()),
                               ('model', LogisticRegression(max_iter=3000, random_state=seed))]), {'model__C': [.01, .1, 1., 10.]}),
        'spline_logistic': (Pipeline([('impute', SimpleImputer(add_indicator=True)),
            ('spline', SplineTransformer(include_bias=False)), ('scale', StandardScaler()),
            ('model', LogisticRegression(max_iter=3000, random_state=seed))]),
            {'spline__n_knots': [3, 5], 'spline__degree': [2, 3], 'model__C': [.1, 1.]}),
        'boosted_trees': (Pipeline([('model', HistGradientBoostingClassifier(early_stopping=False, random_state=seed))]),
                          {'model__max_leaf_nodes': [3, 7], 'model__l2_regularization': [1., 10.]})}
    fitted, summary = {}, {}
    for name, (model, grid) in candidates.items():
        search = GridSearchCV(model, grid, scoring='average_precision', cv=cv, n_jobs=1, error_score='raise')
        search.fit(x, y)
        fitted[name] = search.best_estimator_
        summary[name] = {'grouped_cv_average_precision': float(search.best_score_), 'parameters': search.best_params_}
    winner = max(summary, key=lambda name: summary[name]['grouped_cv_average_precision'])
    return fitted, {'selected': winner, 'candidates': summary, 'test_used': False}


def deviation_and_trend(window):
    # PSEUDOCODE: summarize fixed-reference z values with level, absolute deviation and per-day slope.
    x = np.asarray(window, dtype=float)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] == 0 or np.isinf(x).any():
        raise ValueError('A daily feature matrix is required.')
    result = []
    for j in range(x.shape[1]):
        valid = np.isfinite(x[:, j]); t = np.arange(len(x))[valid]
        y = x[valid, j]
        result.extend([float(y[-1]) if len(y) else np.nan, float(np.mean(np.abs(y))) if len(y) else np.nan,
                       float(np.polyfit(t, y, 1)[0]) if len(y) >= 2 else np.nan])
    return np.asarray(result)
