"""Calibrate evidence acceptance on separate people; unknown scores never become zero."""

import math
from hashlib import sha256
from .schema import METRICS
from .labels import evidence_target

METHOD = 'validation_threshold_independent_people_exact_bonferroni'


def _person_pairs(rows, predictions, key):
    # PSEUDOCODE: choose one row per person using identity alone, so repeated texts do not inflate sample size.
    chosen = {}
    for row, prediction in zip(rows, predictions):
        if key not in row['scores']:
            continue
        target = evidence_target(row, key)
        if target is None:
            continue
        value = prediction['evidence'][key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid evidence probability.')
        order = sha256((key + '\0' + row['example_id']).encode()).hexdigest()
        person = row['participant_id']
        if person not in chosen or order < chosen[person][0]:
            chosen[person] = (order, value, bool(target))
    return [(value, known) for _, value, known in chosen.values()]


def calibrate_evidence(rows, predictions, *, selection_rows, selection_predictions, target_precision, alpha=.05):
    # PSEUDOCODE: select thresholds on validation -> freeze -> certify once on independent calibration people.
    from scipy.stats import beta
    from .guard import choose_empirical_threshold
    if (len(rows) != len(predictions) or len(selection_rows) != len(selection_predictions)
            or type(target_precision) not in (int, float) or not 0 < target_precision < 1
            or type(alpha) not in (int, float) or not 0 < alpha < 1):
        raise ValueError('Invalid evidence calibration inputs.')
    all_rows = [*rows, *selection_rows]
    if any(any(not isinstance(r.get(k), str) or not r[k].strip() for k in ('example_id', 'participant_id')) for r in all_rows):
        raise ValueError('Evidence calibration requires example and participant identities.')
    if len({r['example_id'] for r in all_rows}) != len(all_rows) or {r['participant_id'] for r in rows} & {r['participant_id'] for r in selection_rows}:
        raise ValueError('Threshold selection and certification must use independent people/examples.')
    per_metric_alpha = alpha / len(METRICS)
    thresholds = {}
    for key, *_ in METRICS:
        selection = _person_pairs(selection_rows, selection_predictions, key)
        pairs = _person_pairs(rows, predictions, key)
        positives = sum(known for _, known in pairs)
        candidate = None
        if any(known for _, known in selection) and any(not known for _, known in selection):
            candidate = choose_empirical_threshold([int(known) for _, known in selection],
                [probability for probability, _ in selection], target_precision)
        item = {'threshold': None, 'n': len(pairs), 'known': positives, 'accepted': 0,
                'precision': None, 'precision_lower': 0., 'candidate_threshold': candidate,
                'status': 'insufficient_evidence_calibration'}
        if candidate is not None:
            accepted = [known for value, known in pairs if value >= candidate]
            n, correct = len(accepted), sum(accepted)
            lower = float(beta.ppf(per_metric_alpha, correct, n - correct + 1)) if correct else 0.
            item.update(accepted=n, correct=correct, precision=correct / n if n else None, precision_lower=lower)
            if lower >= target_precision:
                item.update(threshold=candidate, status='calibrated')
        thresholds[key] = item
    return {'source': 'separate_calibration_people', 'method': METHOD, 'target_precision': target_precision,
            'alpha': alpha, 'per_metric_alpha': per_metric_alpha,
            'minimum_accepted_if_all_correct': math.ceil(math.log(per_metric_alpha) / math.log(target_precision)),
            'thresholds': thresholds, 'sampling': 'one_identity_selected_text_per_person_per_metric',
            'selection_objective': 'maximum_supported_recall_then_minimum_false_acceptances_at_target_precision',
            'interpretation': 'requires_independent_representative_people_and_frozen_selection_not_a_clinical_guarantee'}


def qualified_scores(scores, evidence, calibration):
    # PSEUDOCODE: apply frozen per-metric thresholds -> retain a reason for every rejected estimate.
    values, reasons = {}, {}
    for key, score in scores.items():
        if not math.isfinite(score) or not 0 <= score <= 100 or not math.isfinite(evidence[key]) or not 0 <= evidence[key] <= 1:
            raise ValueError('Invalid intensity or evidence estimate.')
        policy = (calibration or {}).get('thresholds', {}).get(key, {})
        threshold = policy.get('threshold')
        if threshold is not None and (calibration or {}).get('method') != METHOD:
            raise ValueError('Evidence thresholds require independent statistical certification.')
        if threshold is not None and (type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1 or policy.get('status') != 'calibrated'):
            raise ValueError('Invalid frozen evidence threshold.')
        accepted = threshold is not None and evidence[key] >= threshold
        values[key] = score if accepted else None
        reasons[key] = None if accepted else ('insufficient_text_evidence' if threshold is not None else 'uncalibrated_text_evidence')
    return values, reasons
