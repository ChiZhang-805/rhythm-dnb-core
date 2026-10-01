"""Blind annotation comparison; disagreement is reviewed rather than auto-averaged."""

import numpy as np
from .schema import CATEGORIES
from ..provenance import fingerprint


def compare_annotations(first, second, *, tolerance=10):
    # PSEUDOCODE: verify same item/different raters -> compare every head -> request adjudication for disagreement.
    if type(tolerance) not in (int, float) or not np.isfinite(tolerance) or not 0 <= tolerance <= 100 or not first.get('rater_id') or not second.get('rater_id'):
        raise ValueError('Invalid adjudication tolerance or rater identity.')
    if first['example_id'] != second['example_id'] or first['category'] != second['category'] or first['rater_id'] == second['rater_id']:
        raise ValueError('Two independent raters of the same example are required.')
    keys = CATEGORIES[first['category']][1]
    if set(first['scores']) != set(keys) or set(second['scores']) != set(keys):
        raise ValueError('Incomplete annotation heads.')
    raw = [r['scores'][k] for r in (first, second) for k in keys]
    if any(v is not None and (type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 100) for v in raw):
        raise ValueError('Annotation score outside 0-100.')
    differences = {k: abs(first['scores'][k] - second['scores'][k]) if first['scores'][k] is not None and second['scores'][k] is not None else None for k in keys}
    evidence_disagreement = [k for k in keys if (first['scores'][k] is None) != (second['scores'][k] is None)]
    comparable = [v for v in differences.values() if v is not None]
    return {'example_id': first['example_id'], 'evidence_id': fingerprint([first, second]),
            'mean_absolute_disagreement': float(np.mean(comparable)) if comparable else None,
            'adjudication_required': bool(evidence_disagreement or any(v > tolerance for v in comparable)),
            'evidence_disagreement': evidence_disagreement, 'per_head': differences}
