"""Event-level performance and participant-cluster uncertainty, including abstention."""

from collections import defaultdict
from dataclasses import replace
from datetime import timedelta
import numpy as np
from ..contracts import AlarmState, EvaluationDay
from ..timebase import instant, forecast_bounds, local_boundary
from ..warning.policy import advance


def replay(rows, threshold, *, consecutive=2, cooldown_days=7):
    # PSEUDOCODE: replay every chronological day per person through the production alarm policy.
    states, output = {}, []
    for row in sorted(rows, key=lambda r: (r.participant_id, instant(r.issued_at))):
        if row.label is not None and (type(row.label) is not int or row.label not in (0, 1)) or row.label == 1 and row.onset is None:
            raise ValueError('Positive evaluation labels require a confirmed onset.')
        day, _, _ = forecast_bounds(row.issued_at, row.timezone)
        warning, status, state = advance(states.get(row.participant_id, AlarmState()), row.participant_id,
                                         'evaluation', day, row.score, threshold,
                                         consecutive=consecutive, cooldown_days=cooldown_days, context=row.timezone)
        states[row.participant_id] = state
        output.append((row, warning, status))
    return output


def include_monitoring_days(rows, monitoring):
    # PSEUDOCODE: enumerate the independently registered calendar -> retain absent forecasts as missing scores/labels.
    expected = {}
    for period in monitoring:
        if period.last_issue_day < period.first_issue_day:
            raise ValueError('Invalid monitoring interval.')
        for offset in range((period.last_issue_day - period.first_issue_day).days + 1):
            issued = local_boundary(period.first_issue_day + timedelta(days=offset), period.timezone, hour=12)
            key = (period.participant_id, instant(issued))
            if key in expected:
                raise ValueError('Overlapping monitoring intervals.')
            expected[key] = EvaluationDay(period.participant_id, issued, None, None, timezone=period.timezone)
    seen = set()
    for row in rows:
        key = (row.participant_id, instant(row.issued_at))
        if key not in expected or key in seen or row.timezone != expected[key].timezone:
            raise ValueError('Forecast rows differ from the registered monitoring calendar.')
        expected[key] = row; seen.add(key)
    return tuple(expected.values())


def event_metrics(rows, threshold, *, consecutive=2, cooldown_days=7, events=None, horizon_days=7, min_lead_hours=24, confirmation_days=2, monitoring=None):
    # PSEUDOCODE: retain each person's first registered event -> match positive alarms -> count false and unevaluable alarms.
    monitoring = tuple(monitoring) if monitoring is not None else None
    rows = include_monitoring_days(rows, monitoring) if monitoring is not None else rows
    sequence = replay(rows, threshold, consecutive=consecutive, cooldown_days=cooldown_days)
    excluded_recurrences = 0
    if events is not None:
        events = tuple(events)
        keys = [(e.participant_id, instant(e.onset)) for e in events]
        if len(set(keys)) != len(keys) or any(instant(e.confirmed_at) < instant(e.onset) for e in events):
            raise ValueError('Invalid or duplicate registry events.')
        if monitoring is not None and {e.participant_id for e in events} - {p.participant_id for p in monitoring}:
            raise ValueError('Event registry contains people outside the registered monitoring cohort.')
        first_events = {}
        for event in sorted(events, key=lambda e: instant(e.onset)):
            first_events.setdefault(event.participant_id, event)
        excluded_recurrences = len(events) - len(first_events)
        events = tuple(first_events.values())
    for row, _, _ in sequence:
        if row.label_available_at is not None and row.label is not None:
            required = instant(row.onset) if row.label == 1 else instant(row.issued_at) + timedelta(days=horizon_days + confirmation_days)
            if instant(row.label_available_at) < required:
                raise ValueError('Outcome label is declared available before its evidence could be known.')
        if row.label == 1 and not timedelta(hours=min_lead_hours) <= instant(row.onset) - instant(row.issued_at) <= timedelta(days=horizon_days):
            raise ValueError('Positive outcome lies outside the registered prediction horizon.')
        if row.label is not None and events is not None:
            relevant = [e for e in events if e.participant_id == row.participant_id and instant(e.onset) <= instant(row.issued_at) + timedelta(days=horizon_days)]
            if row.label == 0 and relevant or any(instant(e.onset) < instant(row.issued_at) + timedelta(hours=min_lead_hours) for e in relevant):
                raise ValueError('Outcome label contradicts the first-event registry.')
            if row.label == 1 and row.label_available_at is not None and any(instant(e.confirmed_at) > instant(row.label_available_at) for e in relevant if instant(e.onset) == instant(row.onset)):
                raise ValueError('Positive label predates registry confirmation.')
    observable_events = {(r.participant_id, instant(r.onset)) for r, _, _ in sequence if r.label == 1}
    denominator = 'all_registered_confirmed_first_events' if events is not None else 'events_with_evaluable_forecast_labels'
    event_keys = {(e.participant_id, instant(e.onset)) for e in events} if events is not None else observable_events
    if not observable_events <= event_keys:
        raise ValueError('Evaluation labels do not match first events in the locked registry.')
    if len({person for person, _ in observable_events}) != len(observable_events):
        raise ValueError('First-event evaluation cannot contain recurrent positive outcomes for one person.')
    detected, leads = set(), []
    true_alarms = false_alarms = unknown_alarms = 0
    valid_risk_days = sum(r.score is not None and r.label is not None for r, _, _ in sequence)
    for row, warning, _ in sequence:
        if warning != 1:
            continue
        if row.label == 1:
            key = (row.participant_id, instant(row.onset))
            true_alarms += 1
            if key not in detected:
                leads.append((instant(row.onset) - instant(row.issued_at)).total_seconds() / 86400)
            detected.add(key)
        elif row.label == 0:
            false_alarms += 1
        else:
            unknown_alarms += 1
    result = {'events': len(event_keys), 'detected_events': len(detected), 'event_denominator': denominator,
              'excluded_recurrent_events': excluded_recurrences,
              'events_with_evaluable_labels': len(observable_events),
              'event_sensitivity': len(detected) / len(event_keys) if event_keys else None,
              'false_alarms': false_alarms, 'true_alarms': true_alarms, 'unknown_alarms': unknown_alarms,
              'valid_risk_days': valid_risk_days,
              'false_alarms_per_30_days': false_alarms * 30 / valid_risk_days if valid_risk_days else None,
              'alarm_ppv': true_alarms / (true_alarms + false_alarms) if true_alarms + false_alarms else None,
              'coverage_denominator': 'registered_monitoring_days' if monitoring is not None else 'submitted_forecast_rows; absent scheduled days are not measured',
              'forecast_days': len(sequence),
              'score_coverage': sum(r.score is not None for r, _, _ in sequence) / len(sequence) if sequence else 0.,
              'label_coverage': sum(r.label is not None for r, _, _ in sequence) / len(sequence) if sequence else 0.,
              'lead_days': leads, 'median_lead_days': float(np.median(leads)) if leads else None}
    eligible = [r for r, _, _ in sequence if r.score is not None and r.label is not None]
    if len({r.label for r in eligible}) == 2:
        from sklearn.metrics import roc_auc_score, average_precision_score
        result['roc_auc'] = float(roc_auc_score([r.label for r in eligible], [r.score for r in eligible]))
        result['average_precision'] = float(average_precision_score([r.label for r in eligible], [r.score for r in eligible]))
    else:
        result.update(roc_auc=None, average_precision=None)
    # A risk classification is different from a notification suppressed by cooldown.
    classified = [(r.label, int(r.score > threshold)) for r in eligible] if threshold is not None else []
    tn = sum(y == 0 and p == 0 for y, p in classified)
    fp = sum(y == 0 and p == 1 for y, p in classified)
    fn = sum(y == 1 and p == 0 for y, p in classified)
    tp = sum(y == 1 and p == 1 for y, p in classified)
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    result.update(risk_confusion={'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
                  risk_accuracy=(tn + tp) / len(classified) if classified else None,
                  risk_balanced_accuracy=(sensitivity + specificity) / 2
                  if sensitivity is not None and specificity is not None else None,
                  risk_sensitivity=sensitivity, risk_specificity=specificity,
                  classified_days=len(classified),
                  classification_coverage=len(classified) / len(sequence) if sequence else 0.,
                  classification_definition='score > frozen threshold; before notification persistence/cooldown')
    return result


def cluster_intervals(rows, threshold, *, repetitions=10000, seed=20261001, events=None, monitoring=None, **policy):
    # PSEUDOCODE: resample whole participants with replacement -> re-identify copies -> recompute metrics.
    groups = defaultdict(list)
    for row in rows:
        groups[row.participant_id].append(row)
    event_groups = defaultdict(list)
    for event in events or ():
        event_groups[event.participant_id].append(event)
        groups.setdefault(event.participant_id, [])
    periods = defaultdict(list)
    for period in monitoring or ():
        periods[period.participant_id].append(period)
        groups.setdefault(period.participant_id, [])
    ids = sorted(groups)
    if len(ids) < 2 or type(repetitions) is not int or repetitions < 2:
        raise ValueError('Cluster intervals need at least two people and two replicates.')
    rng = np.random.default_rng(seed)
    keys = ('event_sensitivity', 'false_alarms_per_30_days', 'alarm_ppv', 'median_lead_days',
            'risk_accuracy', 'risk_balanced_accuracy', 'risk_sensitivity', 'risk_specificity')
    samples = {key: [] for key in keys}
    for _ in range(repetitions):
        chosen = rng.choice(ids, len(ids), replace=True)
        selected = [replace(row, participant_id=f'bootstrap_{i}')
                    for i, person in enumerate(chosen) for row in groups[person]]
        selected_events = [replace(e, participant_id=f'bootstrap_{i}') for i, person in enumerate(chosen)
                           for e in event_groups[person]] if events is not None else None
        selected_periods = [replace(p, participant_id=f'bootstrap_{i}') for i, person in enumerate(chosen)
                            for p in periods[person]] if monitoring is not None else None
        result = event_metrics(selected, threshold, events=selected_events, monitoring=selected_periods, **policy)
        for key in keys:
            if result[key] is not None:
                samples[key].append(result[key])
    return {key: {'percentile_95': np.quantile(values, [.025, .975]).tolist() if values else None,
                  'valid_replicates': len(values), 'requested_replicates': repetitions}
            for key, values in samples.items()}
