"""Reference labels, exploratory rule annotations and saved warnings stay distinct."""

from collections import defaultdict
from datetime import datetime, timedelta
import math

import numpy as np

from ..provenance import fingerprint


DOMAINS = ('sleep_timing_spread', 'meal_interval_irregularity', 'exercise_timing_spread')
POLICY = {
    'purpose': 'retrospective_candidate_rhythm_worsening_annotation',
    'baseline_days': 28, 'baseline_min_days': 23,
    'window_days': 7, 'window_min_days': 6, 'baseline_min_windows': 6,
    'quantile': .95, 'quantile_method': 'higher', 'persistence_days': 3,
    'minimum_abnormal_domains': 2,
    'clinical_gold': False, 'formal_endpoint': False, 'independent_test_truth': False,
    'uses_dnb_or_model_predictions': False,
    'scope': 'Relative worsening of the stored experimental sleep/meal/exercise patterns; '
             'not a diagnosis, sleep SRI, full actigraphy rhythm or the formal SEA endpoint.',
    'parameter_basis': 'Reuse the existing study 28/23-day coverage, 7/6-day window, '
                       '95th percentile and three-day persistence as exploratory settings. '
                       'Require two of three domains; no fitting against warning accuracy.',
    'onset_precision': 'day_only_first_day_of_confirmed_run',
    'availability': 'retrospective_annotation_created_at; not historical acquisition time',
    'references': ['https://www.nature.com/articles/s41598-017-03171-4',
                   'https://www.vldb.org/pvldb/vol11/p269-ratner.pdf'],
}


def _number(value, low=0, high=None):
    # PSEUDOCODE: accept finite recorded values; missing measurements never become zero.
    return (type(value) in (int, float) and math.isfinite(value) and value >= low
            and (high is None or value < high))


def record_day(row):
    # PSEUDOCODE: use the date written in the source timestamp, retaining its documented time basis.
    stamp = datetime.fromisoformat(row['observed_at'].replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('Record needs an explicit timestamp offset.')
    return stamp.date()


def daily_values(row):
    # PSEUDOCODE: use stored sleep and active exercise clocks; never infer meals or all-day activity.
    start, end = row.get('sleep_start_hour'), row.get('sleep_end_hour')
    sleep = None
    if _number(start, high=24) and _number(end, high=24) and (end - start) % 24 > 0:
        sleep = (start + ((end - start) % 24) / 2) % 24
    exercise = row.get('exercise_hour')
    if not (_number(exercise, high=24) and _number(row.get('exercise_minutes'))
            and row['exercise_minutes'] > 0):
        exercise = None
    meal = row.get('meal_interval_cv')
    return (sleep, meal if _number(meal) else None, exercise)


def _spread(values):
    # PSEUDOCODE: circular SD in hours keeps 23:55 and 00:05 close together.
    radians = np.asarray(values) * (2 * np.pi / 24)
    resultant = float(abs(np.exp(1j * radians).mean()))
    if resultant < 1e-12:
        return None
    return float(np.sqrt(max(0., -2 * np.log(min(1., resultant)))) * 24 / (2 * np.pi))


def _window(daily, last_day, policy):
    # PSEUDOCODE: require a common set of complete calendar days across the three domains.
    selected = [(last_day - timedelta(days=i), daily.get(last_day - timedelta(days=i)))
                for i in range(policy['window_days'] - 1, -1, -1)]
    selected = [(d, item) for d, item in selected if item is not None and all(v is not None for v in item[1])]
    if len(selected) < policy['window_min_days']:
        return None, [item[0] for _, item in selected]
    matrix = np.asarray([item[1] for _, item in selected], dtype=float)
    values = (_spread(matrix[:, 0]), float(np.median(matrix[:, 1])), _spread(matrix[:, 2]))
    return (dict(zip(DOMAINS, values)) if all(v is not None for v in values) else None,
            [item[0] for _, item in selected])


def annotate_sequence(rows, policy=None):
    """Annotate one source/person stream without consulting truth or warning fields.

    The event list is retrospective. Confirmation may establish the start of a
    three-day run, but an unevaluable day or absent date always breaks that run.
    A zero means no qualifying relative worsening on that evaluable day only.
    """
    # PSEUDOCODE: freeze the first baseline -> assess nonoverlapping follow-up -> confirm persistent multi-domain changes.
    policy = dict(POLICY if policy is None else policy)
    ordered = sorted(rows, key=record_day)
    if not ordered or len({(r['source_dataset'], r['participant_id']) for r in ordered}) != 1:
        raise ValueError('Need one source/person sequence.')
    days = [record_day(r) for r in ordered]
    if len(set(days)) != len(days):
        raise ValueError('Duplicate source/person/day requires upstream review, not averaging.')
    start = days[0]
    baseline_end = start + timedelta(days=policy['baseline_days'])
    first_followup = baseline_end + timedelta(days=policy['window_days'] - 1)
    daily = {record_day(r): (r['record_id'], daily_values(r)) for r in ordered}
    complete = [item for d, item in daily.items() if start <= d < baseline_end and all(v is not None for v in item[1])]
    baseline = []
    for offset in range(policy['window_days'] - 1, policy['baseline_days']):
        values, _ = _window(daily, start + timedelta(days=offset), policy)
        if values is not None:
            baseline.append(values)
    usable = (len(complete) >= policy['baseline_min_days'] and
              len(baseline) >= policy['baseline_min_windows'] and days[-1] >= baseline_end)
    thresholds = ({key: float(np.quantile([v[key] for v in baseline], policy['quantile'],
                                          method=policy['quantile_method'])) for key in DOMAINS} if usable else None)
    evidence = {'baseline_start': start.isoformat(), 'baseline_end_exclusive': baseline_end.isoformat(),
                'baseline_complete_days': len(complete), 'baseline_windows': len(baseline),
                'baseline_record_ids': [item[0] for item in complete], 'thresholds': thresholds,
                'first_followup_window_end': first_followup.isoformat(), 'policy_id': fingerprint(policy)}
    results = []
    for row, day in zip(ordered, days):
        result = {'record_id': row['record_id'], 'candidate_state': None, 'abnormal_domains': [],
                  'candidate_onset_date': None, 'candidate_confirmed_date': None,
                  'event_id': None, 'measures': None, 'window_record_ids': [], 'record_day': day.isoformat()}
        if day < baseline_end:
            result['status'] = 'baseline_only'
        elif not usable:
            result['status'] = 'insufficient_baseline'
        elif day < first_followup:
            result['status'] = 'followup_window_incomplete'
        else:
            measures, parents = _window(daily, day, policy)
            result.update(measures=measures, window_record_ids=parents)
            if measures is None or any(v is None for v in daily[day][1]):
                result['status'] = 'insufficient_window'
            else:
                result['abnormal_domains'] = [k for k in DOMAINS if measures[k] > thresholds[k] + 1e-10]
                result['status'] = 'above_threshold' if len(result['abnormal_domains']) >= policy['minimum_abnormal_domains'] else 'no_candidate_worsening'
                result['candidate_state'] = None if result['status'] == 'above_threshold' else 0
        results.append(result)
    events, run = [], []

    def finish_run():
        # PSEUDOCODE: confirm only sustained runs, preserve interval precision and explicit incomplete confirmations.
        if not run:
            return
        confirmed = len(run) >= policy['persistence_days']
        if confirmed:
            onset, confirmation = run[0]['record_day'], run[policy['persistence_days'] - 1]['record_day']
            event = {'onset_date': onset, 'confirmed_date': confirmation,
                     'last_qualifying_date': run[-1]['record_day'],
                     'support_record_ids': [r['record_id'] for r in run]}
            event['event_id'] = fingerprint([ordered[0]['source_dataset'], ordered[0]['participant_id'], event, evidence['policy_id']])
            events.append(event)
        for item in run:
            item['status'] = 'candidate_confirmed' if confirmed else 'persistence_not_confirmed'
            if confirmed:
                item.update(candidate_state=1, candidate_onset_date=onset, candidate_confirmed_date=confirmation, event_id=event['event_id'])

    last = None
    for item, day in zip(results, days):
        if item['status'] != 'above_threshold' or last is not None and day != last + timedelta(days=1):
            finish_run(); run = []
        if item['status'] == 'above_threshold':
            run.append(item)
        last = day
    finish_run()
    return {'rows': results, 'events': events, 'baseline': evidence}


def annotate_records(rows):
    # PSEUDOCODE: preserve the scenario labels; annotate the remaining streams as weak candidates only.
    groups = defaultdict(list)
    for row in rows:
        groups[(row['source_dataset'], row['participant_id'])].append(row)
    annotations, streams = [], []
    for (source, person), records in sorted(groups.items()):
        simulated = source == 'SIMULATED-RHYTHM'
        if simulated:
            onsets = {r['event_onset_at'] for r in records if r.get('event_onset_at')}
            if len(onsets) > 1:
                raise ValueError('Conflicting scenario onset times.')
            first = next(iter(onsets), None)
            for row in records:
                annotations.append({'record_id': row['record_id'], 'source_dataset': source, 'participant_id': person,
                    'is_simulated_cohort': 1, 'reference_scope': 'scenario',
                    'reference_event_in_followup': int(first is not None) if row['split'] != 'reference' else None,
                    'reference_onset_at': first, 'candidate_state': None, 'candidate_onset_date': None,
                    'candidate_confirmed_date': None, 'status': 'existing_scenario_preserved',
                    'evidence': {'original_state': row.get('rhythm_state_gt'), 'label_source': row.get('label_source'),
                                 'followup_end_at': row.get('followup_end_at'), 'reference_only': row['split'] == 'reference'}})
            continue
        sequence = annotate_sequence(records)
        streams.append({'source_dataset': source, 'participant_id': person, 'baseline': sequence['baseline'], 'events': sequence['events']})
        by_id = {r['record_id']: r for r in records}
        for item in sequence['rows']:
            original = by_id[item['record_id']]
            known = original.get('rhythm_state_gt') is not None
            annotations.append({'record_id': item['record_id'], 'source_dataset': source, 'participant_id': person,
                'is_simulated_cohort': 0, 'reference_scope': 'prevalent_sleep_phase_diagnosis' if source == 'DelSoM-baseline' and known else 'not_provided',
                'reference_event_in_followup': None, 'reference_onset_at': None,
                'candidate_state': item['candidate_state'], 'candidate_onset_date': item['candidate_onset_date'],
                'candidate_confirmed_date': item['candidate_confirmed_date'], 'status': item['status'],
                'evidence': {'assessment': item, 'baseline_id': fingerprint(sequence['baseline']),
                             'time_basis': 'relative_source_day' if source == 'REST-relative-day' else 'recorded_source_date',
                             'independent_gold': False}})
    return sorted(annotations, key=lambda r: r['record_id']), streams
