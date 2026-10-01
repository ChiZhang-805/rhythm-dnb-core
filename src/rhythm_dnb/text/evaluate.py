"""Intensity error, dispersion, evidence coverage and optional within-person change evaluation."""

from collections import defaultdict
import math
import numpy as np
from .schema import CATEGORIES, METRICS, FORMAL_CATEGORIES
from .evidence import qualified_scores


def average_ranks(values):
    # PSEUDOCODE: stable-sort values -> give tied observations their average rank.
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(order), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def errors(truth, predicted):
    # PSEUDOCODE: validate paired scores -> report error, ordering, agreement and retained variation.
    truth, predicted = np.asarray(truth, dtype=float), np.asarray(predicted, dtype=float)
    if not len(truth) or truth.shape != predicted.shape or not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError('Evaluation needs matching finite nonempty values.')
    residual = predicted - truth
    a, b = average_ranks(truth), average_ranks(predicted)
    truth_sd, pred_sd = float(truth.std()), float(predicted.std())
    denominator = truth_sd ** 2 + pred_sd ** 2 + float(residual.mean()) ** 2
    covariance = float(np.mean((truth - truth.mean()) * (predicted - predicted.mean())))
    return {'n': len(truth), 'mae': float(np.abs(residual).mean()), 'rmse': float(np.sqrt(np.mean(residual ** 2))),
            'bias': float(residual.mean()), 'within_5': float(np.mean(np.abs(residual) <= 5)),
            'within_10': float(np.mean(np.abs(residual) <= 10)),
            'spearman': float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None,
            'pearson': covariance / (truth_sd * pred_sd) if truth_sd > 0 and pred_sd > 0 else None,
            'concordance': 2 * covariance / denominator if denominator > 0 else None,
            'truth_sd': truth_sd, 'prediction_sd': pred_sd,
            'sd_ratio': pred_sd / truth_sd if truth_sd > 0 else None}


def mean_baseline(train_rows):
    # PSEUDOCODE: fit a constant mean from known training labels only.
    values = defaultdict(list)
    for row in train_rows:
        for key, value in row['scores'].items():
            if value is not None:
                values[key].append(value)
    return {key: float(np.mean(values[key])) if values[key] else None for key, *_ in METRICS}


def median_baseline(train_rows):
    # PSEUDOCODE: fit the constant MAE-optimal predictor from known training labels only.
    values = defaultdict(list)
    for row in train_rows:
        for key, value in row['scores'].items():
            if value is not None:
                values[key].append(value)
    return {key: float(np.median(values[key])) if values[key] else None for key, *_ in METRICS}


def longitudinal_report(rows, predictions):
    # PSEUDOCODE: group genuine repeated observations -> compare successive labeled changes in actual time order.
    from ..timebase import instant
    grouped = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        if row.get('observed_at') is not None:
            for key, value in row['scores'].items():
                if value is not None:
                    grouped[(row['participant_id'], key)].append((instant(row['observed_at']), value, prediction['scores'][key]))
    true_changes, predicted_changes, people = defaultdict(list), defaultdict(list), defaultdict(set)
    for (person, key), values in grouped.items():
        values.sort()
        if len({v[0] for v in values}) != len(values):
            raise ValueError('Longitudinal evaluation requires distinct observation times per person and metric.')
        for previous, current in zip(values, values[1:]):
            true_changes[key].append(current[1] - previous[1])
            predicted_changes[key].append(current[2] - previous[2])
            people[key].add(person)
    return {key: {'people': len(people[key]), **errors(values, predicted_changes[key])} for key, values in true_changes.items()}


def report(rows, predictions, baseline, median_reference=None, calibration=None):
    # PSEUDOCODE: retain unknown labels -> assess raw errors and accepted coverage -> expose common residual patterns.
    if len(rows) != len(predictions) or not rows:
        raise ValueError('Every evaluation row must have one prediction.')
    truth, pred, accepted_truth, accepted_pred = (defaultdict(list) for _ in range(4))
    evidence_counts, residual_pairs = defaultdict(lambda: [0, 0, 0]), defaultdict(list)
    for row, prediction in zip(rows, predictions):
        values = prediction['scores']
        keys = CATEGORIES[row['category']][1]
        if set(values) != set(keys) or any(not math.isfinite(v) or not 0 <= v <= 100 for v in values.values()):
            raise ValueError('Malformed or out-of-range prediction.')
        qualified, _ = qualified_scores(values, prediction['evidence'], calibration)
        for key in keys:
            known = row['scores'][key] is not None
            kept = qualified[key] is not None
            evidence_counts[key][0] += 1
            evidence_counts[key][1] += kept
            evidence_counts[key][2] += kept and known
            if known:
                truth[key].append(row['scores'][key]); pred[key].append(values[key])
                if kept:
                    accepted_truth[key].append(row['scores'][key]); accepted_pred[key].append(values[key])
        for i, left in enumerate(keys):
            for right in keys[i + 1:]:
                if row['scores'][left] is not None and row['scores'][right] is not None:
                    residual_pairs[(left, right)].append((values[left] - row['scores'][left], values[right] - row['scores'][right]))
    per_metric = {key: errors(truth[key], pred[key]) for key in truth}
    per_category = {c: {'n': sum(r['category'] == c for r in rows),
                    'mae': float(np.mean([per_metric[k]['mae'] for k in keys if k in per_metric]))}
                    for c, (_, keys) in CATEGORIES.items() if any(k in per_metric for k in keys)}
    formal = [v['mae'] for c, v in per_category.items() if c in FORMAL_CATEGORIES]
    residuals = []
    for (left, right), pairs in residual_pairs.items():
        a, b = np.asarray(pairs, dtype=float).T
        residuals.append({'left': left, 'right': right, 'n': len(pairs),
            'correlation': float(np.corrcoef(a, b)[0, 1]) if len(pairs) >= 3 and a.std() > 0 and b.std() > 0 else None})
    return {'records': len(rows), 'per_metric': per_metric, 'per_category': per_category,
            'macro_category_mae': float(np.mean([v['mae'] for v in per_category.values()])) if per_category else None,
            'formal_category_mae': float(np.mean(formal)) if formal else None,
            'mean_baseline': {k: errors(truth[k], [baseline[k]] * len(truth[k])) for k in truth if baseline[k] is not None},
            'median_baseline': {k: errors(truth[k], [median_reference[k]] * len(truth[k])) for k in truth if median_reference[k] is not None} if median_reference else None,
            'accepted_errors': {k: errors(v, accepted_pred[k]) for k, v in accepted_truth.items()},
            'evidence': {k: {'n': n, 'accepted': accepted, 'coverage': accepted / n,
                         'accepted_supported_fraction': known / accepted if accepted else None}
                         for k, (n, accepted, known) in evidence_counts.items()},
            'within_person_changes': longitudinal_report(rows, predictions), 'residual_correlations': residuals,
            'reference': 'semantic_reference_scores_not_clinical_measurements', 'independent_human_gold': False}
