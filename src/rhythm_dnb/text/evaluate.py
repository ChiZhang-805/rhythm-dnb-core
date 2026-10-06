"""Intensity error, dispersion, evidence coverage and optional within-person change evaluation."""

from collections import defaultdict
import math
import numpy as np
from .schema import CATEGORIES, METRICS, FORMAL_CATEGORIES
from .evidence import qualified_scores
from .labels import evidence_target


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
    if len(rows) != len(predictions):
        raise ValueError('Every longitudinal row must have one prediction.')
    grouped = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        if row.get('observed_at') is not None:
            for key, value in row['scores'].items():
                if value is not None:
                    grouped[(row['participant_id'], key)].append((instant(row['observed_at']), value, prediction['scores'][key]))
    true_changes, predicted_changes, people = defaultdict(list), defaultdict(list), defaultdict(set)
    per_person, gaps = defaultdict(list), defaultdict(list)
    for (person, key), values in grouped.items():
        values.sort()
        if len({v[0] for v in values}) != len(values):
            raise ValueError('Longitudinal evaluation requires distinct observation times per person and metric.')
        if len(values) >= 2:
            per_person[key].append({'participant_id': person, **errors([v[1] for v in values], [v[2] for v in values])})
        for previous, current in zip(values, values[1:]):
            true_changes[key].append(current[1] - previous[1])
            predicted_changes[key].append(current[2] - previous[2])
            people[key].add(person)
            gaps[key].append((current[0] - previous[0]).total_seconds() / 3600)
    result = {}
    for key, values in true_changes.items():
        reference, predicted = np.asarray(values), np.asarray(predicted_changes[key])
        changing = reference != 0
        result[key] = {'people': len(people[key]), **errors(values, predicted),
            'direction_agreement_on_nonzero_reference_changes': float(np.mean(np.sign(predicted[changing]) == np.sign(reference[changing]))) if changing.any() else None,
            'nonzero_reference_changes': int(changing.sum()), 'gap_hours_minimum': min(gaps[key]), 'gap_hours_maximum': max(gaps[key]),
            'per_person': per_person[key], 'person_macro_mae': float(np.mean([p['mae'] for p in per_person[key]])),
            'pair_definition': 'successive labeled observations; unequal gaps are reported, not assumed daily',
            'temporal_bases': sorted({r.get('temporal_basis', 'unspecified') for r in rows if r.get('observed_at')})}
    return result


def within_person_network(rows, predictions):
    # PSEUDOCODE: align jointly labeled metrics at the same instant -> compare within-person correlations without interpolation.
    from itertools import combinations
    from ..timebase import instant
    if len(rows) != len(predictions):
        raise ValueError('Every network observation must have one prediction.')
    timelines = defaultdict(dict)
    for row, prediction in zip(rows, predictions):
        if row.get('observed_at') is None:
            continue
        observed = timelines[row['participant_id']].setdefault(instant(row['observed_at']), {})
        for key, reference in row['scores'].items():
            if reference is None:
                continue
            if key in observed:
                raise ValueError('Network evaluation requires one reference per person, instant and metric.')
            estimate = prediction['scores'][key]
            if not math.isfinite(reference) or not math.isfinite(estimate):
                raise ValueError('Network evaluation requires finite paired reference estimates.')
            observed[key] = (reference, estimate)
    diagnostics = []
    for person, timeline in sorted(timelines.items()):
        pairs = defaultdict(list)
        for observed in timeline.values():
            for left, right in combinations(sorted(observed), 2):
                pairs[(left, right)].append((*observed[left], *observed[right]))
        for (left, right), values in sorted(pairs.items()):
            if len(values) < 3:
                continue
            reference_left, estimate_left, reference_right, estimate_right = np.asarray(values).T
            reference = float(np.corrcoef(reference_left, reference_right)[0, 1]) if reference_left.std() > 0 and reference_right.std() > 0 else None
            estimate = float(np.corrcoef(estimate_left, estimate_right)[0, 1]) if estimate_left.std() > 0 and estimate_right.std() > 0 else None
            diagnostics.append({'participant_id': person, 'left': left, 'right': right, 'paired_instants': len(values),
                'reference_correlation': reference, 'estimate_correlation': estimate,
                'correlation_error': estimate - reference if reference is not None and estimate is not None else None})
    per_person = defaultdict(list)
    for item in diagnostics:
        if item['correlation_error'] is not None:
            per_person[item['participant_id']].append(abs(item['correlation_error']))
    return {'pairs': diagnostics, 'people_with_estimable_pairs': len(per_person),
        'person_macro_absolute_correlation_error': float(np.mean([np.mean(v) for v in per_person.values()])) if per_person else None,
        'minimum_instants': 3, 'interpolation': False,
        'interpretation': 'descriptive check on jointly labeled source instants; three points only permit calculation, not reliable DNB validation'}


def report(rows, predictions, baseline, median_reference=None, calibration=None):
    # PSEUDOCODE: retain unknown labels -> assess raw errors and accepted coverage -> expose common residual patterns.
    if len(rows) != len(predictions) or not rows:
        raise ValueError('Every evaluation row must have one prediction.')
    truth, pred, accepted_truth, accepted_pred = (defaultdict(list) for _ in range(4))
    evidence_counts, residual_pairs = defaultdict(lambda: [0, 0, 0, 0]), defaultdict(list)
    evidence_pairs = defaultdict(list)
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
            target = evidence_target(row, key)
            if target is not None and row.get('label_states'):
                evidence_pairs[key].append((target, prediction['evidence'][key]))
            evidence_counts[key][2] += kept and target == 1
            evidence_counts[key][3] += kept and target is not None
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
    scopes = [p['scope'] for p in predictions if p.get('scope')]
    counts = {k: sum(s[k] for s in scopes) for k in ('tokens', 'true_positive', 'false_positive', 'false_negative')}
    denominator = 2 * counts['true_positive'] + counts['false_positive'] + counts['false_negative']
    evidence_prediction = {}
    for key, pairs in evidence_pairs.items():
        y, probability = np.asarray(pairs).T
        probability = probability.clip(np.finfo(float).eps, 1 - np.finfo(float).eps)
        positive, negative = int(y.sum()), int(len(y) - y.sum())
        evidence_prediction[key] = {'n': len(y), 'positive': positive, 'negative': negative,
            'brier': float(np.mean((probability - y) ** 2)),
            'log_loss': float(-np.mean(y * np.log(probability) + (1 - y) * np.log1p(-probability))),
            'roc_auc': float((average_ranks(probability)[y == 1].sum() + positive - positive * (positive + 1) / 2) / (positive * negative)) if positive and negative else None}
    return {'records': len(rows), 'per_metric': per_metric, 'per_category': per_category,
            'macro_category_mae': float(np.mean([v['mae'] for v in per_category.values()])) if per_category else None,
            'formal_category_mae': float(np.mean(formal)) if formal else None,
            'mean_baseline': {k: errors(truth[k], [baseline[k]] * len(truth[k])) for k in truth if baseline[k] is not None},
            'median_baseline': {k: errors(truth[k], [median_reference[k]] * len(truth[k])) for k in truth if median_reference[k] is not None} if median_reference else None,
            'accepted_errors': {k: errors(v, accepted_pred[k]) for k, v in accepted_truth.items()},
            'evidence': {k: {'n': n, 'accepted': accepted, 'coverage': accepted / n,
                         'accepted_with_reviewed_evidence': reviewed,
                         'accepted_supported_fraction': known / reviewed if reviewed else None}
                         for k, (n, accepted, known, reviewed) in evidence_counts.items()},
            'metric_support': {key: {'known': len(truth[key]), 'unknown': sum(r['scores'].get(key) is None for r in rows if r['category'] == category),
                'minimum_reference': min(truth[key]) if truth[key] else None, 'maximum_reference': max(truth[key]) if truth[key] else None,
                'status': 'evaluated' if truth[key] else 'not_evaluated'} for key, category, *_ in METRICS},
            'within_person_changes': longitudinal_report(rows, predictions), 'within_person_network': within_person_network(rows, predictions),
            'residual_correlations': residuals,
            'scope_supervision': {**counts, 'rows': len(scopes), 'token_f1': 2 * counts['true_positive'] / denominator if denominator else None,
                'mean_row_cross_entropy': float(np.mean([s['binary_cross_entropy'] for s in scopes])) if scopes else None,
                'interpretation': 'only explicitly labeled evidence/exclusion tokens; not a certified explanation'},
            'evidence_prediction': evidence_prediction,
            'reference': 'semantic_reference_scores_not_clinical_measurements', 'independent_human_gold': False}
