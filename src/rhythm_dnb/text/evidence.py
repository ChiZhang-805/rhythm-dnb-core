"""Calibrate evidence acceptance on separate people; unknown scores never become zero."""

import math
from .schema import METRICS


def calibrate_evidence(rows, predictions, *, target_precision, minimum):
    # PSEUDOCODE: use calibration labels only -> find maximal coverage meeting precision -> retain absent thresholds.
    if len(rows) != len(predictions) or type(minimum) is not int or minimum < 1 or not 0 < target_precision <= 1:
        raise ValueError('Invalid evidence calibration inputs.')
    thresholds = {}
    for key, *_ in METRICS:
        pairs = [(p['evidence'][key], r['scores'][key] is not None) for r, p in zip(rows, predictions) if key in r['scores']]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v, _ in pairs):
            raise ValueError('Invalid evidence probability.')
        positives = sum(known for _, known in pairs)
        item = {'threshold': None, 'n': len(pairs), 'known': positives, 'accepted': 0,
                'precision': None, 'status': 'insufficient_evidence_calibration'}
        if positives >= minimum and len(pairs) - positives >= minimum:
            for threshold in sorted({v for v, _ in pairs}):
                selected = [known for v, known in pairs if v >= threshold]
                if len(selected) >= minimum and sum(selected) / len(selected) >= target_precision:
                    item.update(threshold=threshold, accepted=len(selected), precision=sum(selected) / len(selected), status='calibrated')
                    break
        thresholds[key] = item
    return {'source': 'separate_calibration_people', 'target_precision': target_precision,
            'minimum_per_class': minimum, 'thresholds': thresholds,
            'interpretation': 'empirical_calibration_target_not_a_population_guarantee'}


def qualified_scores(scores, evidence, calibration):
    # PSEUDOCODE: apply frozen per-metric thresholds -> retain a reason for every rejected estimate.
    values, reasons = {}, {}
    for key, score in scores.items():
        if not math.isfinite(score) or not 0 <= score <= 100 or not math.isfinite(evidence[key]) or not 0 <= evidence[key] <= 1:
            raise ValueError('Invalid intensity or evidence estimate.')
        policy = (calibration or {}).get('thresholds', {}).get(key, {})
        threshold = policy.get('threshold')
        if threshold is not None and (type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1 or policy.get('status') != 'calibrated'):
            raise ValueError('Invalid frozen evidence threshold.')
        accepted = threshold is not None and evidence[key] >= threshold
        values[key] = score if accepted else None
        reasons[key] = None if accepted else ('insufficient_text_evidence' if threshold is not None else 'uncalibrated_text_evidence')
    return values, reasons
