"""Independent sleep/eating/activity endpoint rules. DNB scores never enter these functions."""

from datetime import timedelta
import numpy as np
from ..contracts import OutcomeAssessment
from ..provenance import fingerprint
from ..timebase import instant

DOMAINS = ('S1', 'E1', 'A1')


def personal_anchor(values_by_day, baseline_start, *, baseline_days=14, window=7, minimum=6):
    """Median of baseline rolling-window measures, days 7 through 14 inclusive."""
    # PSEUDOCODE: select preregistered baseline window endings -> require six complete measurements.
    if any(type(v) is not int for v in (baseline_days, window, minimum)) or not 2 <= minimum <= window or baseline_days - window + 1 < minimum:
        raise ValueError('Incoherent endpoint baseline windows.')
    selected = []
    for offset in range(window - 1, baseline_days):
        values = values_by_day.get(baseline_start + timedelta(days=offset), {})
        if all(values.get(k) is not None and np.isfinite(values[k]) for k in DOMAINS):
            selected.append([values[k] for k in DOMAINS])
    if len(selected) < minimum:
        return None
    return dict(zip(DOMAINS, np.median(selected, axis=0).tolist()))


def weighted_quantile(values, weights, quantile):
    # PSEUDOCODE: sort observations -> accumulate person-balanced mass -> invert empirical CDF.
    x, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if x.ndim != 1 or x.shape != w.shape or len(x) == 0 or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w <= 0) or not 0 < quantile < 1:
        raise ValueError('Invalid weighted quantile input.')
    order = np.argsort(x, kind='stable')
    return float(x[order][min(len(x) - 1, np.searchsorted(np.cumsum(w[order]), quantile * w.sum()))])


def fit_criteria(stable_people, cutoff, *, quantile=.95, minimum_people=60):
    """Each person supplies anchor, stable evidence and follow-up domain measurements.

    Schema: participant_id, anchor, stable=True, evidence_id, available_at, values
    (list of dicts with S1/E1/A1). Windows within a person get total weight one.
    """
    # PSEUDOCODE: require independently certified stable follow-up -> fit absolute and positive-change bounds.
    people = list(stable_people); ids = [p['participant_id'] for p in people]
    if len(set(ids)) != len(ids) or len(ids) < minimum_people:
        raise ValueError('Endpoint thresholds require independent stable people.')
    thresholds = {}
    for p in people:
        if p.get('stable') is not True or not p.get('evidence_id') or instant(p['available_at']) > instant(cutoff):
            raise ValueError('Stable outcome evidence is unavailable at fitting cutoff.')
        if not all(k in p['anchor'] and np.isfinite(p['anchor'][k]) for k in DOMAINS):
            raise ValueError('Missing stable-person baseline anchor.')
    for domain in DOMAINS:
        levels, changes, weights = [], [], []
        for p in people:
            values = [float(row[domain]) for row in p['values'] if row.get(domain) is not None and np.isfinite(row[domain])]
            if not values:
                raise ValueError('Stable follow-up lacks domain ' + domain)
            levels.extend(values)
            changes.extend(max(0., value - p['anchor'][domain]) for value in values)
            weights.extend([1 / len(values)] * len(values))
        thresholds[domain] = {'absolute': weighted_quantile(levels, weights, quantile),
                              'increase': weighted_quantile(changes, weights, quantile)}
    payload = {'definition': 'sleep-eating-activity', 'thresholds': thresholds, 'quantile': quantile,
               'people': sorted(ids), 'cutoff': instant(cutoff).isoformat()}
    return {**payload, 'id': fingerprint(payload)}


def assess_day(participant_id, day, values, anchor, criteria, available_at):
    # PSEUDOCODE: require all domains/anchor -> compare both independent bounds -> keep missingness explicit.
    if criteria.get('id') != fingerprint({k: v for k, v in criteria.items() if k != 'id'}):
        raise ValueError('Endpoint criteria checksum mismatch.')
    if instant(available_at) < instant(criteria['cutoff']) or participant_id in criteria['people']:
        raise ValueError('Endpoint criteria must precede use on independent people.')
    valid = anchor is not None and all(values.get(k) is not None and np.isfinite(values[k]) and
                                       k in anchor and np.isfinite(anchor[k]) for k in DOMAINS)
    abnormal = () if not valid else tuple(k for k in DOMAINS
        if values[k] > criteria['thresholds'][k]['absolute'] and
           values[k] - anchor[k] > criteria['thresholds'][k]['increase'])
    return OutcomeAssessment(participant_id, day, abnormal, bool(valid), instant(available_at))
