"""Explicit simulated follow-up for short histories, without rewriting their past."""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import math

import numpy as np

from ..provenance import fingerprint
from .dataset_annotations import POLICY, annotate_sequence, daily_values


SCENARIOS = ('stable', 'stable_late_phase', 'transient_change',
             'gradual_irregularity', 'abrupt_irregularity', 'coupled_fluctuation')
SCENARIO_NAMES = dict(zip(SCENARIOS, ('稳定作息', '稳定晚作息', '短暂变化',
                                   '逐渐不规律', '突然不规律', '协同波动')))
FEATURES = ('sleep_midpoint_h', 'sleep_duration_h', 'meal_interval_cv',
            'exercise_hour', 'exercise_minutes', 'resting_hr_bpm', 'screen_time_min')
SETTINGS = {
    'seed': 20261010, 'days': 84, 'anchor_max_records': 7,
    'baseline_days': 28, 'change_day_min': 42, 'change_day_max': 56,
    'train_fraction': .6, 'calibration_fraction': .2,
    'scenarios': list(SCENARIOS), 'outcome_policy': POLICY,
    'domain': 'authored_simulation_not_clinical_validation',
    'time_basis': 'simulated_day_zero_2000_01_01_not_actual_followup_dates',
    'parameter_basis': '84 days = existing 28-day baseline + eight weeks of complete simulated '
                       'follow-up. Six balanced stress cases, one per eligible source identity; '
                       'no selection by DNB accuracy. Noise and changes below are authoring '
                       'settings, not estimated patient dynamics or optimal research thresholds.',
    'baseline_noise': {'sleep_clock_h': .22, 'meal_cv': .025, 'exercise_clock_h': .3,
                       'sleep_duration_h': .3, 'exercise_log_sd': .2,
                       'resting_hr_bpm': 2., 'screen_log_sd': .15},
    'change_noise': {'sleep_clock_h': 1.6, 'meal_cv': .22, 'exercise_clock_h': 2.2},
    'stable_late_shift_h': 3., 'transient_days': 2,
    'gradual_ramp_days': 21, 'coupling_ramp_days': 14,
    'noise_ar': .25, 'coupled_ar': .8,
    'authoring_bounds': {'sleep_duration_h': [3, 11], 'meal_interval_cv': [.001, 1.5],
                         'exercise_minutes': [1, 240], 'resting_hr_bpm': [35, 120],
                         'screen_time_min': [0, 960]},
    'forecast_horizon_days': 7, 'minimum_lead_days': 1,
    'truth_definition': 'First day of the authored sustained multi-domain irregular regime. '
                        'Abrupt starts at change_day; gradual starts after its 21-day ramp; '
                        'coupled starts after its 14-day ramp. Stable, stable late and two-day '
                        'transient scenarios have no sustained regime event. These are simulation '
                        'answers, not conclusions about the original person or clinical thresholds.',
    'truth_confirmation_days': 3,
    'candidate_rule_is_reference_truth': False,
    'uses_warning_outputs': False, 'uses_model_scores': False,
    'text_inference_performed': False, 'clinical_gold': False,
}


def select_backgrounds(source, annotations):
    """Choose only identities lacking any evaluable original candidate window."""
    # PSEUDOCODE: retain stable reference and labelled cohorts; preserve every source identity and anchor lineage.
    annotated = {a['record_id']: a for a in annotations}
    if len(annotated) != len(annotations) or set(annotated) != {r['record_id'] for r in source}:
        raise ValueError('Need exactly one saved annotation for each source record.')
    groups = defaultdict(list)
    for row in source:
        groups[row['participant_id']].append(row)
    backgrounds = []
    for person, rows in sorted(groups.items()):
        if any(r['source_dataset'] == 'SIMULATED-RHYTHM' for r in rows):
            continue
        if any(annotated[r['record_id']]['candidate_state'] is not None for r in rows):
            continue
        ordered = sorted(rows, key=lambda r: (r['observed_at'], r['record_id']))
        first_source = ordered[0]['source_dataset']
        anchors = [r for r in ordered if r['source_dataset'] == first_source][:SETTINGS['anchor_max_records']]
        clocks = [daily_values(r) for r in anchors]
        if any(a is None or b is None or c is None for a, b, c in clocks):
            raise ValueError('Incomplete anchor requires review: ' + person)
        values = {
            'sleep_midpoint_h': circular_center([v[0] for v in clocks]),
            'exercise_hour': circular_center([v[2] for v in clocks]),
        }
        for key in FEATURES:
            if key not in values:
                observed = [r.get(key) for r in anchors]
                if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in observed):
                    raise ValueError('Missing anchor measurement: ' + person + '/' + key)
                values[key] = float(np.median(observed))
        backgrounds.append({'parent_participant_id': person, 'anchor_source': first_source,
                            'anchor_record_ids': [r['record_id'] for r in anchors], 'anchor_features': values,
                            'original_records': len(rows), 'original_followup_outcome': None})
    ordered = sorted(backgrounds, key=lambda r: fingerprint([SETTINGS['seed'], r['parent_participant_id']]))
    n_train = math.floor(len(ordered) * SETTINGS['train_fraction'])
    n_cal = math.floor(len(ordered) * SETTINGS['calibration_fraction'])
    for i, row in enumerate(ordered):
        row['split'] = 'train' if i < n_train else 'calibration' if i < n_train + n_cal else 'test'
        row['scenario'] = SCENARIOS[i % len(SCENARIOS)]
        row['sequence_id'] = 'FOLLOWUP-' + fingerprint([SETTINGS['seed'], row['parent_participant_id']])[:20]
    return sorted(ordered, key=lambda r: r['sequence_id'])


def circular_center(values):
    # PSEUDOCODE: summarize clock anchors around midnight without inventing a noon average.
    z = np.exp(1j * np.asarray(values, float) * 2 * np.pi / 24).mean()
    if abs(z) < 1e-8:
        raise ValueError('Undefined circular anchor.')
    return float(np.angle(z) % (2 * np.pi) * 24 / (2 * np.pi))


def generate_sequence(background, settings=None):
    """Create a virtual trajectory with authored regime truth and separate rule outputs."""
    # PSEUDOCODE: fix regime changes without predictions; generate measurements; retain candidate-rule errors instead of relabelling truth.
    cfg = SETTINGS if settings is None else settings
    if cfg['days'] < 56 or cfg['baseline_days'] != cfg['outcome_policy']['baseline_days']:
        raise ValueError('Invalid simulated follow-up coverage.')
    case = background['scenario']
    if case not in SCENARIOS:
        raise ValueError('Unknown authored scenario.')
    sequence_id = background['sequence_id']
    rng = np.random.default_rng(int(fingerprint([cfg['seed'], sequence_id])[:16], 16))
    change_day = int(rng.integers(cfg['change_day_min'], cfg['change_day_max'] + 1))
    anchor = background['anchor_features']
    base_noise, added = cfg['baseline_noise'], cfg['change_noise']
    start = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
    onset_day = (change_day if case == 'abrupt_irregularity' else
                 change_day + cfg['gradual_ramp_days'] if case == 'gradual_irregularity' else
                 change_day + cfg['coupling_ramp_days'] if case == 'coupled_fluctuation' else None)
    if onset_day is not None and onset_day + cfg['truth_confirmation_days'] > cfg['days']:
        raise ValueError('Authored event lacks full confirmation follow-up.')
    onset = (start + timedelta(days=onset_day)).date().isoformat() if onset_day is not None else None
    confirmed = ((start + timedelta(days=onset_day + cfg['truth_confirmation_days'] - 1)).date().isoformat()
                 if onset_day is not None else None)
    autoregression = np.zeros(3)
    common = 0.
    rows = []
    for day in range(cfg['days']):
        amplitude = 0.
        if case == 'transient_change':
            amplitude = float(change_day <= day < change_day + cfg['transient_days'])
        elif case == 'gradual_irregularity':
            amplitude = float(np.clip((day - change_day) / cfg['gradual_ramp_days'], 0, 1))
        elif case == 'abrupt_irregularity':
            amplitude = float(day >= change_day)
        elif case == 'coupled_fluctuation':
            amplitude = float(np.clip((day - change_day) / cfg['coupling_ramp_days'], 0, 1))
        autoregression = cfg['noise_ar'] * autoregression + np.sqrt(1-cfg['noise_ar']**2) * rng.normal(size=3)
        common = cfg['coupled_ar'] * common + np.sqrt(1-cfg['coupled_ar']**2) * rng.normal()
        disturbance = rng.normal(size=3)
        if case == 'coupled_fluctuation':
            disturbance = np.array([common, -common, common])
        clock = (anchor['sleep_midpoint_h'] + (cfg['stable_late_shift_h'] if case == 'stable_late_phase' else 0)
                 + base_noise['sleep_clock_h'] * autoregression[0] + added['sleep_clock_h'] * amplitude * disturbance[0]) % 24
        bounds = cfg['authoring_bounds']
        duration = float(np.clip(anchor['sleep_duration_h'] + base_noise['sleep_duration_h'] * rng.normal(), *bounds['sleep_duration_h']))
        # These bounds are explicit scenario-authoring ranges, not clinical cutoffs or missing-value imputations.
        meal = float(np.clip(anchor['meal_interval_cv'] + base_noise['meal_cv'] * autoregression[1]
                             + added['meal_cv'] * amplitude * (1 + abs(disturbance[1])), *bounds['meal_interval_cv']))
        exercise_hour = float((anchor['exercise_hour'] + base_noise['exercise_clock_h'] * autoregression[2]
                               + added['exercise_clock_h'] * amplitude * disturbance[2]) % 24)
        features = dict(zip(FEATURES, (float(clock), duration, meal, exercise_hour,
            float(np.clip(anchor['exercise_minutes'] * np.exp(base_noise['exercise_log_sd'] * rng.normal()), *bounds['exercise_minutes'])),
            float(np.clip(anchor['resting_hr_bpm'] + base_noise['resting_hr_bpm'] * rng.normal(), *bounds['resting_hr_bpm'])),
            float(np.clip(anchor['screen_time_min'] * np.exp(base_noise['screen_log_sd'] * rng.normal()), *bounds['screen_time_min'])))))
        stamp = (start + timedelta(days=day)).isoformat()
        row = {'record_id': f'{sequence_id}-{day:03d}', 'participant_id': sequence_id,
               'parent_participant_id': background['parent_participant_id'], 'split': background['split'],
               'source_dataset': 'SIMULATED-FOLLOWUP', 'day_index': day, 'simulated_at': stamp,
               'features': features, 'regime_state': int(onset_day is not None and day >= onset_day)}
        rows.append(row)
    rule_rows = [{**r, **r['features'], 'observed_at': r['simulated_at'],
                  'sleep_start_hour': (r['features']['sleep_midpoint_h'] - r['features']['sleep_duration_h']/2) % 24,
                  'sleep_end_hour': (r['features']['sleep_midpoint_h'] + r['features']['sleep_duration_h']/2) % 24}
                 for r in rows]
    labels = annotate_sequence(rule_rows, cfg['outcome_policy'])
    for row, label in zip(rows, labels['rows']):
        if row['record_id'] != label['record_id']:
            raise ValueError('Annotation order mismatch.')
        row.update(candidate_state=label['candidate_state'], candidate_status=label['status'],
                   candidate_onset_date=label['candidate_onset_date'],
                   candidate_confirmed_date=label['candidate_confirmed_date'])
    # Only forecasting before the first confirmed event is eligible; a truncated horizon stays unknown.
    for row in rows:
        day = row['day_index']
        current = datetime.fromisoformat(row['simulated_at']).date()
        difference = (datetime.fromisoformat(onset).date() - current).days if onset else None
        if day < cfg['baseline_days']:
            value, reason = None, 'baseline_only'
        elif onset is not None and current >= datetime.fromisoformat(onset).date():
            value, reason = None, 'first_event_already_started'
        elif difference is not None and cfg['minimum_lead_days'] <= difference <= cfg['forecast_horizon_days']:
            value, reason = 1, 'complete_simulated_followup'
        elif day + cfg['forecast_horizon_days'] + cfg['truth_confirmation_days'] - 1 >= cfg['days']:
            value, reason = None, 'insufficient_future_confirmation_coverage'
        else:
            value, reason = 0, 'complete_simulated_followup'
        row.update(future_event_7d=value, future_label_status=reason)
    evidence = {**background, 'simulated_days': cfg['days'], 'change_day': change_day,
                'event_in_followup': int(onset is not None), 'first_onset_date': onset,
                'first_confirmed_date': confirmed, 'followup_end_date': rows[-1]['simulated_at'][:10],
                'label_source': 'authored_sustained_multi_domain_regime', 'candidate_events': labels['events'],
                'baseline': labels['baseline'], 'measurement_sha256': fingerprint([
                    [r['record_id'], r['features']] for r in rows])}
    return rows, evidence
