"""Calibrate the full max-module + persistence + cooldown strategy on separate people."""

import numpy as np
from ..provenance import fingerprint
from ..timebase import instant, local_boundary
from .evaluate import event_metrics


def calibrate(rows, config, reference, discovery, cutoff, *, method='single_sample', minimum_events=None,
              minimum_negative_days=None, events=None, monitoring=None):
    # PSEUDOCODE: validate fitting partition -> sweep score thresholds -> maximize event sensitivity under budget.
    minimum_events = config.calibration_min_events if minimum_events is None else minimum_events
    minimum_negative_days = config.calibration_min_negative_days if minimum_negative_days is None else minimum_negative_days
    if method not in ('single_sample', 'rolling') or any(type(v) is not int or v < 1 for v in (minimum_events, minimum_negative_days)):
        raise ValueError('Invalid calibration method or minimum outcome evidence.')
    rows = tuple(rows)
    registry = tuple(events) if events is not None else None
    monitoring = tuple(monitoring) if monitoring is not None else None
    people = sorted({r.participant_id for r in rows} | {e.participant_id for e in registry or ()} | {p.participant_id for p in monitoring or ()})
    if set(people) & (set(reference['people']) | set(discovery['people'])):
        raise ValueError('Calibration must use independent participants.')
    if any(instant(r.issued_at) >= instant(cutoff) for r in rows):
        raise ValueError('Calibration cutoff does not cover its observations.')
    if any(instant(local_boundary(p.last_issue_day, p.timezone, hour=12)) >= instant(cutoff) for p in monitoring or ()):
        raise ValueError('Calibration monitoring calendar contains future days.')
    if any(r.label is not None and (r.label_available_at is None or instant(r.label_available_at) > instant(cutoff)) for r in rows):
        raise ValueError('Calibration labels are not yet known at cutoff.')
    if events is None:
        raise ValueError('Primary real-data calibration requires an independent confirmed-event registry.')
    if monitoring is None:
        raise ValueError('Primary real-data calibration requires an independently registered monitoring calendar.')
    if registry is not None and any(instant(e.confirmed_at) > instant(cutoff) or e.participant_id not in people for e in registry):
        raise ValueError('Calibration event registry has unavailable or out-of-partition events.')
    event_keys = {(e.participant_id, e.onset) for e in registry} if registry is not None else {(r.participant_id, r.onset) for r in rows if r.label == 1}
    negatives = sum(r.label == 0 and r.score is not None for r in rows)
    event_metrics(rows, None, events=registry, monitoring=monitoring, horizon_days=config.horizon_days,
                  min_lead_hours=config.min_lead_hours, confirmation_days=config.persistence_days - 1)
    payload = {'reference_id': reference['id'], 'discovery_id': discovery['id'], 'people': people,
               'cutoff': instant(cutoff).isoformat(), 'method': method, 'threshold': None,
               'minimum_events': minimum_events, 'minimum_negative_days': minimum_negative_days,
               'monitoring_id': fingerprint(monitoring) if monitoring is not None else None,
               'status': 'insufficient_outcomes', 'selection': None, 'candidates': []}
    if len(event_keys) >= minimum_events and negatives >= minimum_negative_days and discovery['modules']:
        scores = np.asarray([r.score for r in rows if r.score is not None], dtype=float)
        if not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError('Invalid calibration scores.')
        candidates = np.unique(np.r_[0., scores])
        feasible = []
        for threshold in candidates:
            metrics = event_metrics(rows, float(threshold), consecutive=config.alarm_consecutive,
                                    cooldown_days=config.cooldown_days, events=registry, monitoring=monitoring,
                                    horizon_days=config.horizon_days, min_lead_hours=config.min_lead_hours,
                                    confirmation_days=config.persistence_days - 1)
            payload['candidates'].append({'threshold': float(threshold), **metrics})
            if metrics['false_alarms_per_30_days'] is not None and metrics['false_alarms_per_30_days'] <= config.max_false_alarms_per_30_days:
                feasible.append((metrics['event_sensitivity'], -metrics['false_alarms_per_30_days'], float(threshold), metrics))
        if feasible:
            best = max(feasible, key=lambda row: row[:3])
            if best[0] > 0:
                payload.update(threshold=best[2], status='research_calibrated', selection=best[3])
            else:
                payload['status'] = 'no_useful_threshold'
    return {**payload, 'id': fingerprint(payload)}
