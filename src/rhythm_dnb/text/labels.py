"""Separate supported scores, absent symptoms, missing evidence and unfinished annotation."""

import math
from .schema import CATEGORIES, METRICS

STATES = ('supported', 'supported_unscored', 'explicit_absence', 'insufficient_evidence', 'unreviewed', 'disputed')


def validate_labels(row):
    # PSEUDOCODE: validate explicit annotation states without guessing meaning from a missing score.
    states = row.get('label_states')
    if states is None:
        return
    keys = CATEGORIES[row['category']][1]
    if not isinstance(states, dict) or set(states) != set(keys):
        raise ValueError('Label states must cover every category metric.')
    for key, state in states.items():
        value = row['scores'][key]
        if state not in STATES:
            raise ValueError('Unknown label state: ' + key)
        if state in ('supported', 'explicit_absence'):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError('Supported labels need finite 0-100 scores.')
        elif value is not None:
            raise ValueError('Unreviewed, disputed or unsupported labels must retain null, not a guessed number.')
        if state == 'explicit_absence' and (value != 0 or next(m[3] for m in METRICS if m[0] == key) != 'absent_to_extreme'):
            raise ValueError('Explicit absence requires the zero endpoint of a symptom intensity scale.')


def evidence_target(row, key):
    # PSEUDOCODE: supervise confirmed support/absence only; legacy nulls and unresolved labels remain masked.
    state = row.get('label_states', {}).get(key)
    if state in ('supported', 'supported_unscored', 'explicit_absence'):
        return 1.
    if state == 'insufficient_evidence':
        return 0.
    if state in ('unreviewed', 'disputed'):
        return None
    # A legacy numerical reference supplies a positive, but a legacy null is not a negative annotation.
    return 1. if row['scores'].get(key) is not None else None


def has_supervision(row):
    # PSEUDOCODE: allow evidence-only or span-only rows without inventing numerical references.
    return (any(v is not None for v in row['scores'].values())
            or any(evidence_target(row, k) is not None for k in row['scores']) or bool(row.get('scope_targets')))
