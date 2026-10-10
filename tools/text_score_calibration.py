"""Fit bounded monotone score corrections using text development people, never DNB outcomes."""

import numpy as np
from sklearn.isotonic import IsotonicRegression

from rhythm_dnb.research.experiment_data import split_text_groups, TEXT
from rhythm_dnb.provenance import fingerprint
from tools.personal_warning import weights


STRENGTHS = (0., .25, .5, .75)
SEED = 20261010


def development_rows(data):
    # PSEUDOCODE: split original text-validation identities into two intact connected groups for cross-fitting.
    rows = [r for r in data['text-development']['rows'] if r['split'] == 'validation'
            and any(r['scores'].get(k[5:]) is not None for k in TEXT)]
    split = split_text_groups(rows, SEED, validation_fraction=.5)
    for row in split:
        row['calibration_fold'] = int(row['split'] == 'validation')
    if {r['calibration_fold'] for r in split} != {0, 1}:
        raise ValueError('Insufficient disconnected text development groups.')
    return split


def fit_curve(x, y, groups):
    # PSEUDOCODE: fit an increasing bounded least-squares curve with equal total weight per person.
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (x.ndim != 1 or x.shape != y.shape or len(groups) != len(x) or len(x) < 2 or
            not np.isfinite(x).all() or not np.isfinite(y).all() or
            np.any(x < 0) or np.any(x > 100) or np.any(y < 0) or np.any(y > 100)):
        raise ValueError('Invalid calibration observations.')
    model = IsotonicRegression(y_min=0, y_max=100, increasing=True, out_of_bounds='clip')
    model.fit(x, y, sample_weight=weights(groups))
    return {'x': model.X_thresholds_.tolist(), 'y': model.y_thresholds_.tolist()}


def apply_curve(curve, values, strength):
    # PSEUDOCODE: blend monotone correction with the original score; preserve missingness and strict ranking.
    if not np.isfinite(strength) or not 0 <= strength < 1:
        raise ValueError('Correction must retain a positive original-score component.')
    values = np.asarray(values, float)
    if np.isinf(values).any() or np.any(values < 0) or np.any(values > 100):
        raise ValueError('Invalid intensity input.')
    x, y = np.asarray(curve['x'], float), np.asarray(curve['y'], float)
    if (not len(x) or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or
            np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0) or np.any(x < 0) or np.any(x > 100) or
            np.any(y < 0) or np.any(y > 100)):
        raise ValueError('Invalid monotone calibration knots.')
    if strength == 0:
        return values.copy()
    return (1-strength)*values + strength*np.interp(values, x, y)


def select_curves(rows, predictions):
    # PSEUDOCODE: select correction strength from group-held-out errors, then refit knots on text development only.
    fitted = {}
    for feature in TEXT:
        metric = feature[5:]
        selected = [r for r in rows if r['scores'].get(metric) is not None]
        x = np.array([predictions[fingerprint([r['category'], r['text']])][metric] for r in selected])
        y = np.array([r['scores'][metric] for r in selected], float)
        groups = np.array([r.get('participant_id') or r.get('family_id') or r.get('group_id') or r['example_id']
                           for r in selected])
        folds = np.array([r['calibration_fold'] for r in selected])
        if set(folds) != {0, 1}:
            raise ValueError('Metric lacks both calibration folds: ' + metric)
        transformed = {a: np.full(len(x), np.nan) for a in STRENGTHS}
        split_receipts = []
        for fold in (0, 1):
            train, validation = folds != fold, folds == fold
            if set(groups[train]) & set(groups[validation]):
                raise ValueError('Calibration person/group leakage.')
            curve = fit_curve(x[train], y[train], groups[train])
            for alpha in STRENGTHS:
                transformed[alpha][validation] = apply_curve(curve, x[validation], alpha)
            split_receipts.append({'fit_examples': [r['example_id'] for r, ok in zip(selected, train) if ok],
                                   'validation_examples': [r['example_id'] for r, ok in zip(selected, validation) if ok]})
        choices = [{'strength': a, 'person_weighted_oof_mae': float(weights(groups) @ np.abs(y-p))}
                   for a, p in transformed.items()]
        if any(not np.isfinite(c['person_weighted_oof_mae']) for c in choices):
            raise ValueError('Incomplete cross-fitted predictions.')
        best = min(choices, key=lambda c: (round(c['person_weighted_oof_mae'], 12), c['strength']))
        fitted[feature] = {'curve': fit_curve(x, y, groups), **best, 'candidates': choices,
                           'n': len(x), 'people_or_groups': len(set(groups)), 'splits': split_receipts}
    return fitted
