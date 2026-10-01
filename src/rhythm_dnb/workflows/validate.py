"""Locked evaluation uses no fitting calls and never selects new modules or thresholds."""

from ..bundles import check_compatibility
from ..research.evaluate import event_metrics, cluster_intervals
from ..timebase import instant
from .develop import score_cases


def validate(bundle, cases, *, bootstrap_repetitions=None, evaluation_as_of=None, events=None, monitoring=None):
    # PSEUDOCODE: verify frozen policy/available labels -> infer unseen people -> report event metrics and intervals.
    config = check_compatibility(bundle)
    if bootstrap_repetitions is None:
        bootstrap_repetitions = config.bootstrap_repetitions
    if type(bootstrap_repetitions) is not int or bootstrap_repetitions < 0 or bootstrap_repetitions == 1:
        raise ValueError('Bootstrap repetitions must be zero (disabled) or an integer of at least two.')
    calibration = bundle['calibration']
    if events is None:
        raise ValueError('Primary real-data evaluation requires a separately locked confirmed-event registry.')
    if monitoring is None:
        raise ValueError('Primary real-data evaluation requires an independently registered monitoring calendar.')
    if calibration is None or calibration['threshold'] is None:
        raise ValueError('No useful calibrated policy is available for locked validation.')
    cases = tuple(cases)
    if not cases:
        raise ValueError('Evaluation cases are empty.')
    if evaluation_as_of is None:
        raise ValueError('An explicit evaluation as-of cutoff is required.')
    if any(instant(c.request.issued_at) > instant(evaluation_as_of) for c in cases):
        raise ValueError('Evaluation includes future forecast requests.')
    if any(c.label.value is not None and instant(c.label_available_at) > instant(evaluation_as_of) for c in cases):
        raise ValueError('Some evaluation labels have not yet become available.')
    if events is not None:
        events = tuple(events)
        if any(instant(e.confirmed_at) > instant(evaluation_as_of) for e in events):
            raise ValueError('The event registry contains unconfirmed future outcomes.')
        fitted = set(bundle['reference']['people'] + bundle['discovery']['people'] + calibration['people'] + bundle['text_fitted_people'])
        if any(e.participant_id in fitted for e in events):
            raise ValueError('Event registry includes fitted participants.')
    rows, responses = score_cases(bundle, cases, method=calibration['method'])
    policy = {'consecutive': config.alarm_consecutive, 'cooldown_days': config.cooldown_days,
              'horizon_days': config.horizon_days, 'min_lead_hours': config.min_lead_hours,
              'confirmation_days': config.persistence_days - 1}
    if monitoring is not None:
        monitoring = tuple(monitoring)
        from ..timebase import local_boundary
        fitted = set(bundle['reference']['people'] + bundle['discovery']['people'] + calibration['people'] + bundle['text_fitted_people'])
        if any(p.participant_id in fitted or instant(local_boundary(p.last_issue_day, p.timezone, hour=12)) > instant(evaluation_as_of) for p in monitoring):
            raise ValueError('Monitoring calendar overlaps fitting people or includes future days.')
    metrics = event_metrics(rows, calibration['threshold'], events=events, monitoring=monitoring, **policy)
    intervals = cluster_intervals(rows, calibration['threshold'], repetitions=bootstrap_repetitions,
                                  seed=config.seed, events=events, monitoring=monitoring, **policy) if bootstrap_repetitions else None
    return {'bundle_id': bundle['id'], 'domain': 'source_backed',
            'metrics': metrics, 'cluster_intervals': intervals, 'predictions': responses,
            'bootstrap_repetitions': bootstrap_repetitions,
            'fitting_on_test': False, 'clinical_validation': False}
