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
    vectors = np.asarray([[r['scores'][k] for k in keys] for r in (first, second)], dtype=float)
    if not np.isfinite(vectors).all() or np.any((vectors < 0) | (vectors > 100)):
        raise ValueError('Annotation score outside 0-100.')
    differences = np.abs(vectors[0] - vectors[1])
    return {'example_id': first['example_id'], 'evidence_id': fingerprint([first, second]),
            'mean_absolute_disagreement': float(differences.mean()),
            'adjudication_required': bool(np.any(differences > tolerance)),
            'per_head': dict(zip(keys, differences.tolist()))}
