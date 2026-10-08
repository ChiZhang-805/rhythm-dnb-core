"""Compare frozen warning methods on identical calendars and tune thresholds only before testing."""

from dataclasses import asdict, replace
from ..contracts import EvaluationDay
from ..provenance import fingerprint
from ..timebase import instant
from .evaluate import event_metrics, include_monitoring_days


def _identity(row):
    # PSEUDOCODE: exclude the score from identity so methods must agree on every outcome and its availability.
    return fingerprint({k: v for k, v in asdict(row).items() if k != 'score'})


def compare_methods(methods, *, calibration_events, test_events, calibration_monitoring, test_monitoring,
                    calibration_cutoff, evaluation_as_of, config, fitted_people, bootstrap_repetitions=0):
    # PSEUDOCODE: enforce equal independent cohorts -> calibrate the same alarm budget -> evaluate every method without choosing on test.
    if not methods or not calibration_monitoring or not test_monitoring:
        raise ValueError('Comparison requires methods and independent registered calendars.')
    if type(bootstrap_repetitions) is not int or bootstrap_repetitions < 0 or bootstrap_repetitions == 1:
        raise ValueError('Bootstrap must be zero or at least two repetitions.')
    if instant(calibration_cutoff) >= instant(evaluation_as_of):
        raise ValueError('Calibration must precede the final evaluation cutoff.')
    cohorts = [{p.participant_id for p in periods} for periods in (calibration_monitoring, test_monitoring)]
    if cohorts[0] & cohorts[1] or (cohorts[0] | cohorts[1]) & set(fitted_people):
        raise ValueError('Fitting, calibration and test participants must be disjoint.')
    for registry, people, cutoff in ((calibration_events, cohorts[0], calibration_cutoff), (test_events, cohorts[1], evaluation_as_of)):
        if registry is None or any(e.participant_id not in people or instant(e.confirmed_at) > instant(cutoff) for e in registry):
            raise ValueError('Events must be independently registered and confirmed within their partition cutoff.')
    aligned, identities = {}, {}
    for name, partitions in methods.items():
        if set(partitions) != {'calibration', 'test'}:
            raise ValueError('Each method needs calibration and test scores, with no test-dependent fit.')
        aligned[name] = {}
        for role, calendar, cutoff in (('calibration', calibration_monitoring, calibration_cutoff), ('test', test_monitoring, evaluation_as_of)):
            rows = include_monitoring_days(partitions[role], calendar)
            if any(instant(r.issued_at) >= instant(cutoff) or r.label is not None and (r.label_available_at is None or instant(r.label_available_at) > instant(cutoff)) for r in rows):
                raise ValueError('Comparison contains observations or labels unavailable at the cutoff.')
            ids = {_identity(r) for r in rows}
            if role in identities and identities[role] != ids:
                raise ValueError('Methods must share identical participants, issue times, labels and follow-up.')
            identities[role] = ids
            aligned[name][role] = rows
    policy = {'consecutive': config.alarm_consecutive, 'cooldown_days': config.cooldown_days,
        'horizon_days': config.horizon_days, 'min_lead_hours': config.min_lead_hours, 'confirmation_days': config.persistence_days - 1}
    result = {}
    for name, partitions in aligned.items():
        candidates = []
        rows = partitions['calibration']
        reference = event_metrics(rows, None, events=calibration_events, monitoring=calibration_monitoring, **policy)
        enough = (len({e.participant_id for e in calibration_events}) >= config.calibration_min_events
                  and sum(r.label == 0 and r.score is not None for r in rows) >= config.calibration_min_negative_days)
        if enough:
            import math
            scores = [r.score for r in rows if r.score is not None]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in scores):
                raise ValueError('Comparison scores must be finite, nonnegative risks, not raw signed features.')
            for threshold in sorted({0., *scores}):
                metrics = event_metrics(rows, threshold, events=calibration_events, monitoring=calibration_monitoring, **policy)
                if metrics['false_alarms_per_30_days'] is not None and metrics['false_alarms_per_30_days'] <= config.max_false_alarms_per_30_days:
                    candidates.append((metrics['event_sensitivity'], -metrics['false_alarms_per_30_days'], threshold, metrics))
        selected = max(candidates, key=lambda x: x[:3]) if candidates else None
        threshold = selected[2] if selected and selected[0] > 0 else None
        test_rows = partitions['test']
        evaluation = event_metrics(test_rows, threshold, events=test_events, monitoring=test_monitoring, **policy)
        result[name] = {'threshold': threshold, 'calibration': selected[3] if threshold is not None else reference,
            'status': 'calibrated' if threshold is not None else 'no_useful_calibrated_policy', 'test': evaluation,
            'intervals': None}
    differences = None
    if bootstrap_repetitions:
        differences = _paired_intervals(aligned, result, test_events, test_monitoring, policy,
                                        repetitions=bootstrap_repetitions, seed=config.seed)
    return {'methods': result, 'test_used_for_selection': False, 'winner_selected_on_test': False,
        'paired_differences': differences,
        'cohort_id': fingerprint(identities_as_lists(identities)), 'false_alarm_budget': config.max_false_alarms_per_30_days,
        'interpretation': 'requires independently produced as-of scores and frozen methods; comparison alone is not clinical validation'}


def _paired_intervals(aligned, result, events, monitoring, policy, *, repetitions, seed):
    # PSEUDOCODE: draw the same whole people for every frozen method -> retain each method and paired difference on that replicate.
    from collections import defaultdict
    from itertools import combinations
    import numpy as np
    groups = {name: defaultdict(list) for name in aligned}
    for name, partitions in aligned.items():
        for row in partitions['test']:
            groups[name][row.participant_id].append(row)
    periods, event_groups = defaultdict(list), defaultdict(list)
    for period in monitoring:
        periods[period.participant_id].append(period)
    for event in events:
        event_groups[event.participant_id].append(event)
    ids = sorted(periods)
    if len(ids) < 2:
        raise ValueError('Paired cluster intervals need at least two independent test people.')
    keys = ('event_sensitivity', 'false_alarms_per_30_days', 'alarm_ppv', 'median_lead_days')
    samples = {name: {key: [] for key in keys} for name in aligned}
    pairs = {pair: {key: [] for key in (*keys, 'score_coverage')} for pair in combinations(sorted(aligned), 2)}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        chosen = rng.choice(ids, len(ids), replace=True)
        selected_events = [replace(event, participant_id=f'bootstrap_{i}') for i, person in enumerate(chosen)
                           for event in event_groups[person]]
        selected_periods = [replace(period, participant_id=f'bootstrap_{i}') for i, person in enumerate(chosen)
                            for period in periods[person]]
        metrics = {}
        for name, people in groups.items():
            selected = [replace(row, participant_id=f'bootstrap_{i}') for i, person in enumerate(chosen)
                        for row in people[person]]
            metrics[name] = event_metrics(selected, result[name]['threshold'], events=selected_events,
                monitoring=selected_periods, **policy)
            for key in keys:
                if metrics[name][key] is not None:
                    samples[name][key].append(metrics[name][key])
        for (left, right), measures in pairs.items():
            for key, values in measures.items():
                if metrics[left][key] is not None and metrics[right][key] is not None:
                    values.append(metrics[left][key] - metrics[right][key])
    def interval(values):
        # PSEUDOCODE: expose undefined replicates rather than replacing missing outcomes by zero.
        return {'percentile_95': np.quantile(values, [.025, .975]).tolist() if values else None,
                'valid_replicates': len(values), 'requested_replicates': repetitions}
    for name, measures in samples.items():
        result[name]['intervals'] = {key: interval(values) for key, values in measures.items()}
    output = []
    for (left, right), measures in pairs.items():
        point = {key: result[left]['test'][key] - result[right]['test'][key]
                 if result[left]['test'][key] is not None and result[right]['test'][key] is not None else None
                 for key in measures}
        output.append({'left': left, 'right': right, 'direction': 'left_minus_right',
            'metrics': {key: {'estimate': point[key], **interval(values)} for key, values in measures.items()}})
    return {'comparisons': output, 'sampling': 'same_whole_people_in_each_method_per_replicate',
        'thresholds_refitted': False, 'confidence_intervals': 'pointwise_percentile_95_not_multiplicity_adjusted',
        'winner_selected': False,
        'interpretation': 'paired descriptive uncertainty; lead-time comparisons may involve different detected events'}


def identities_as_lists(identities):
    # PSEUDOCODE: serialize calendar identities in a stable order.
    return {k: sorted(v) for k, v in identities.items()}


def window_sensitivity(base, candidates):
    # PSEUDOCODE: enumerate explicitly chosen predictor windows while freezing the outcome definition and horizon.
    allowed = {'rolling_days', 'rolling_min_days', 'max_missing_run'}
    if not candidates:
        raise ValueError('Provide prespecified predictor-window alternatives.')
    output, seen = [], set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate or set(candidate) - allowed:
            raise ValueError('Predictor sensitivity cannot alter outcome windows, persistence, baseline or horizon.')
        config = replace(base, **candidate)
        identity = fingerprint(asdict(config))
        if identity in seen:
            raise ValueError('Duplicate sensitivity configuration.')
        seen.add(identity)
        output.append({'id': identity, 'study': asdict(config)})
    return {'candidates': output, 'outcome_definition_fixed': True, 'selection_partition': 'development_only',
        'test_selection_permitted': False, 'status': 'prespecified_candidates_not_results'}
