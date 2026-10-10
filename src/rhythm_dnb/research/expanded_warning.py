"""Causal warning comparison on the sealed, authored 84-day follow-up sequences."""

from collections import defaultdict
from datetime import timedelta

import numpy as np

from ..contracts import EvaluationDay, MonitoringPeriod, OutcomeEvent
from ..timebase import instant
from .evaluate import event_metrics
from .reference_stability import network_changes, personal_components
from .simulated_followup import FEATURES


def sequence_features(rows, config):
    # PSEUDOCODE: freeze a person's first baseline days; all later predictors use only that baseline and past inputs.
    ordered = sorted(rows, key=lambda r: instant(r['observed_at']))
    if len({r['participant_id'] for r in ordered}) != 1:
        raise ValueError('One person is required.')
    days = [r['day_index'] for r in ordered]
    times = [instant(r['observed_at']) for r in ordered]
    if days != list(range(len(rows))) or any(b-a != timedelta(days=1) for a, b in zip(times, times[1:])):
        raise ValueError('Expected complete ordered daily follow-up.')
    if any(instant(r['observed_at']) > instant(r['issued_at']) for r in ordered):
        raise ValueError('Future input used in a forecast.')
    names = sorted(FEATURES)
    x = np.asarray([[r['features'][k] for k in names] for r in ordered], float)
    count = config['baseline_days']; epsilon = config['epsilon']
    if len(x) <= count or not np.isfinite(x).all():
        raise ValueError('Incomplete follow-up baseline or inputs.')
    center = x[:count].mean(0)
    for name in ('sleep_midpoint_h', 'exercise_hour'):
        j = names.index(name)
        resultant = np.exp(1j * x[:count, j] * 2*np.pi/24).mean()
        if abs(resultant) <= epsilon:
            raise ValueError('Undefined baseline clock phase.')
        center[j] = np.angle(resultant) * 24/(2*np.pi) % 24
    z = x - center
    for name in ('sleep_midpoint_h', 'exercise_hour'):
        j = names.index(name); z[:, j] = (z[:, j] + 12) % 24 - 12
    scale = z[:count].std(0, ddof=1)
    if np.any(scale <= epsilon):
        raise ValueError('Constant baseline; do not silently invent its variance.')
    z /= scale
    deviation, perturbation = network_changes(z[:count], z[count:])
    output = {}
    for i in range(count, len(ordered)):
        part = personal_components(deviation[i-count], perturbation[i-count], epsilon)
        personal = part['score'] if part else np.nan
        ratio = part['ratio'] if part else np.nan
        control = [*z[i], *np.abs(z[i]), *(z[i]-z[i-1])]
        for window in config['history_windows']:
            values = z[max(0, i-window+1):i+1]
            t = np.arange(len(values), dtype=float); t -= t.mean()
            slope = t @ values / (t @ t)
            control.extend([*values.mean(0), *values.std(0, ddof=1), *slope])
        rolling, network = {}, [np.log1p(personal), np.log1p(ratio)]
        for window in config['rolling_windows']:
            values = z[i-window+1:i+1]
            sd = values.std(0, ddof=1)
            if len(values) != window or np.any(sd <= epsilon):
                parts = None
            else:
                corr = np.corrcoef(values, rowvar=False); np.fill_diagonal(corr, 0.)
                parts = personal_components(sd, corr, epsilon)
            rolling[window] = parts['score'] if parts else np.nan
            network.extend([np.log1p(rolling[window]), np.log1p(parts['ratio']) if parts else np.nan])
        output[ordered[i]['record_id']] = {'control': control, 'network': network,
            'personal_dnb': personal, 'mean_deviation': float(deviation[i-count].mean()),
            'rolling': rolling, 'module': [names[j] for j in part['module']] if part else [],
            'baseline_record_ids': [r['record_id'] for r in ordered[:count]],
            'latest_input_at': ordered[i]['observed_at']}
    return output, {'features': names, 'center': center.tolist(), 'scale': scale.tolist(),
                    'reference_kind': 'within_person_stable_days_not_independent_reference_people'}


def forecast_objects(records, answers, sequences, scores, config):
    # PSEUDOCODE: reconstruct virtual-day endpoints and an outcome-blind monitoring calendar; verify every saved label.
    groups = defaultdict(list)
    for row in records:
        groups[row['participant_id']].append(row)
    days, events, periods = [], [], []
    for person, rows in sorted(groups.items()):
        seq = sequences[person]
        rows = sorted(rows, key=lambda r: r['day_index'])
        origin = instant(rows[0]['issued_at'])
        onset = instant(seq['first_onset_date'] + 'T12:00:00+00:00') if seq['first_onset_date'] else None
        confirmed = instant(seq['first_confirmed_date'] + 'T12:00:00+00:00') if onset else None
        end = instant(rows[-1]['issued_at'])
        if end.date().isoformat() != seq['followup_end_date'] or bool(onset) != bool(seq['event_in_followup']):
            raise ValueError('Follow-up registry contradicts the record calendar.')
        if onset:
            events.append(OutcomeEvent(person, onset, confirmed, 'authored_sustained_multi_domain_regime'))
        first = origin + timedelta(days=config['baseline_days'])
        periods.append(MonitoringPeriod(person, first.date(), end.date(), 'UTC'))
        for row in rows:
            if row['day_index'] < config['baseline_days']:
                continue
            issued = instant(row['issued_at'])
            distance = (onset - issued).days if onset else None
            expected = (None if onset and issued >= onset else 1 if distance is not None and 1 <= distance <= 7
                        else None if issued + timedelta(days=7 + config['confirmation_days']) > end else 0)
            label = answers[row['record_id']]['future_event_7d']
            if label != expected:
                raise ValueError('Frozen label/time mismatch: ' + row['record_id'])
            value = scores.get(row['record_id'])
            if value is not None and not np.isfinite(value):
                value = None
            days.append(EvaluationDay(person, issued, float(value) if value is not None else None,
                                      label, onset if label == 1 else None, end))
    return days, events, periods


def policy(config):
    # PSEUDOCODE: pass the same frozen alert policy to calibration, evaluation and replay.
    return {k: config[k] for k in ('consecutive', 'cooldown_days', 'horizon_days', 'min_lead_hours', 'confirmation_days')}


def threshold_table(days, config):
    # PSEUDOCODE: replay every calibration threshold in parallel with the exact production persistence/cooldown state.
    valid = [r for r in days if r.label is not None and r.score is not None]
    if {r.label for r in valid} != {0, 1}:
        raise ValueError('Calibration requires both outcomes and finite scores.')
    thresholds = np.unique([0., *[r.score for r in valid]])
    shape = thresholds.shape
    tp, tn, fp, fn = [np.zeros(shape, int) for _ in range(4)]
    false, detected, unknown = [np.zeros(shape, int) for _ in range(3)]
    groups = defaultdict(list)
    for row in days:
        groups[row.participant_id].append(row)
    for records in groups.values():
        run = np.zeros(shape, int); last_alarm = np.full(shape, -100000, int)
        hit = np.zeros(shape, bool); previous = None
        for row in sorted(records, key=lambda r: instant(r.issued_at)):
            day = instant(row.issued_at).date().toordinal()
            if previous is not None and day != previous + 1:
                run[:] = 0
            risk = row.score is not None and row.score > thresholds
            run = np.where(risk, run + 1, 0)
            alarm = (run >= config['consecutive']) & (day - last_alarm >= config['cooldown_days'])
            last_alarm[alarm] = day; run[alarm] = 0
            if row.label is None:
                unknown += alarm
            elif row.score is not None:
                if row.label == 1:
                    tp += risk; fn += ~risk; hit |= alarm
                else:
                    fp += risk; tn += ~risk; false += alarm
            previous = day
        detected += hit
    balanced = .5 * (tp / (tp+fn) + tn/(tn+fp))
    rate = false * 30 / len(valid)
    feasible = np.flatnonzero(rate <= config['max_false_alarms_per_30_days'])
    if not len(feasible):
        raise ValueError('No threshold satisfies the frozen alarm budget.')
    best = max(feasible, key=lambda j: (balanced[j], detected[j], -rate[j], thresholds[j]))
    return {'threshold': float(thresholds[best]), 'selection': config['threshold_selection'],
            'calibration_people': sorted(groups), 'selected_index': int(best),
            'candidates': [{'threshold': float(thresholds[i]), 'balanced_accuracy': float(balanced[i]),
                            'detected_events': int(detected[i]), 'false_alarms': int(false[i]),
                            'unknown_alarms': int(unknown[i]), 'false_alarms_per_30_days': float(rate[i]),
                            'risk_confusion': dict(tp=int(tp[i]), tn=int(tn[i]), fp=int(fp[i]), fn=int(fn[i]))}
                           for i in range(len(thresholds))]}


def paired_uncertainty(evaluations, config):
    # PSEUDOCODE: resample entire held-out people with shared draws; aggregate independently replayed person results.
    aggregates, intervals = {}, {}
    people = sorted({r.participant_id for r in next(iter(evaluations.values()))['days']})
    draws = np.random.default_rng(config['seed']).integers(len(people), size=(config['cluster_bootstrap_repetitions'], len(people)))
    for name, item in evaluations.items():
        per_person = []
        for person in people:
            days = [r for r in item['days'] if r.participant_id == person]
            events = [r for r in item['events'] if r.participant_id == person]
            periods = [r for r in item['periods'] if r.participant_id == person]
            per_person.append(event_metrics(days, item['threshold'], events=events, monitoring=periods, **policy(config)))
        counts = np.array([[m['risk_confusion'][k] for k in ('tp', 'tn', 'fp', 'fn')] for m in per_person])
        samples = counts[draws].sum(1); tp, tn, fp, fn = samples.T
        with np.errstate(divide='ignore', invalid='ignore'):
            ba = .5 * (tp/(tp+fn) + tn/(tn+fp)); accuracy = (tp+tn)/samples.sum(1)
        auxiliary = np.array([[m['detected_events'], m['events'], m['false_alarms'], m['valid_risk_days']] for m in per_person])
        total = auxiliary[draws].sum(1)
        with np.errstate(divide='ignore', invalid='ignore'):
            sensitivity = total[:, 0]/total[:, 1]; false = total[:, 2]*30/total[:, 3]
        aggregates[name] = ba
        arrays = dict(risk_balanced_accuracy=ba, risk_accuracy=accuracy, event_sensitivity=sensitivity,
                      false_alarms_per_30_days=false)
        intervals[name] = {key: {'percentile_95': np.quantile(v[np.isfinite(v)], [.025, .975]).tolist(),
                                'valid_replicates': int(np.isfinite(v).sum())} for key, v in arrays.items()}
    pairs = {}
    for a, b in (('personal_dnb', 'mean_deviation'), ('rolling_dnb', 'mean_deviation'), ('history_dnb', 'history_control')):
        delta = aggregates[a] - aggregates[b]; delta = delta[np.isfinite(delta)]
        pairs[a + '_minus_' + b] = {'balanced_accuracy_difference_95': np.quantile(delta, [.025, .975]).tolist(),
                                   'valid_replicates': len(delta)}
    return {'methods': intervals, 'paired_differences': pairs, 'resampling_unit': 'held_out_parent_person',
            'scope': 'Conditional on the frozen fitted models; does not include refitting or clinical transport uncertainty.'}
