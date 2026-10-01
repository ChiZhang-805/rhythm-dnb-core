"""Causal observation-to-panel preparation and independent rolling endpoint measurements."""

from datetime import timedelta
from zoneinfo import ZoneInfo
import numpy as np
from ..contracts import DailyPanel, FeatureValue
from ..timebase import available_as_of, local_boundary, instant
from ..provenance import build_lineage
from ..io.validation import validate_observations
from ..measures.panel import get_panel, UNITS, ALL_MEASURES
from ..text.schema import CATEGORIES
from ..measures.sleep import daily_sleep, sleep_regularity
from ..measures.eating import daily_eating, eating_regularity
from ..measures.activity import daily_activity, activity_regularity
from ..measures.physiology import daily_physiology


def prepare_day(observations, participant_id, day, zone, issued_at, *, panel_id='joint12',
                sleep_complete=False, eating_complete=False, text_predictor=None):
    # PSEUDOCODE: quantify every supported domain -> select the prespecified fixed DNB universe.
    names = get_panel(panel_id)
    all_values = quantify_day(observations, participant_id, day, zone, issued_at,
        sleep_complete=sleep_complete, eating_complete=eating_complete, text_predictor=text_predictor)
    features = {f.name: f for f in all_values.features}
    return DailyPanel(participant_id, day, zone, tuple(features[name] for name in names))


def quantify_day(observations, participant_id, day, zone, issued_at, *,
                 sleep_complete=False, eating_complete=False, text_predictor=None):
    """Known acquisition variables: sleep_episode, caloric_event, activity_hour,
    wear_minutes_hour, resting_hr_bpm, text:emotion/stress/diet/sleep/social.

    Sleep duration is time asleep inside the half-open research day. The main
    sleep midpoint belongs to its completion day and may precede that day's start.
    Whole-day scalar metrics cannot be repeated or silently averaged.
    """
    # PSEUDOCODE: validate observations -> select same-person causal completed events -> derive each domain.
    observations = validate_observations(observations)
    begin, cutoff = instant(local_boundary(day, zone)), instant(local_boundary(day + timedelta(days=1), zone))
    if cutoff > instant(issued_at):
        raise ValueError('Cannot prepare an unfinished research day.')
    if type(sleep_complete) is not bool or type(eating_complete) is not bool:
        raise ValueError('Coverage declarations must be explicit booleans.')
    rows = [r for r in observations if r.participant_id == participant_id and
            r.timezone == zone and ((begin <= instant(r.start) < cutoff) if instant(r.start) == instant(r.end)
                                   else (begin < instant(r.end) <= cutoff))
            and available_as_of(r, cutoff, issued_at)]
    groups = {}
    for row in rows:
        groups.setdefault(row.variable, []).append(row)
    values, parents, model_ids, reasons = {}, {}, {}, {}
    sleeps = groups.get('sleep_episode', [])
    # A completed interval may cross 04:00: its known overlap still belongs to the previous day.
    duration_sources = [r for r in observations if r.participant_id == participant_id and r.timezone == zone
                        and r.variable == 'sleep_episode' and instant(r.start) < cutoff and instant(r.end) > begin
                        and instant(r.end) <= instant(issued_at) and instant(r.available_at) <= instant(issued_at)]
    main = groups.get('main_sleep_period', [])
    if len(main) > 1 or any(r.unit != 'interval' for r in main):
        raise ValueError('At most one explicitly identified main sleep period is allowed.')
    if main:
        # Assign the whole completed main sleep to its awakening day, including earlier segments.
        within_main = [r for r in observations if r.participant_id == participant_id and r.timezone == zone
                       and r.variable == 'sleep_episode' and instant(main[0].start) <= instant(r.start)
                       and instant(r.end) <= instant(main[0].end) and available_as_of(r, cutoff, issued_at)]
        sleeps = list({r.observation_id: r for r in sleeps + within_main}.values())
    if any(r.unit != 'state' or type(r.value) not in (int, float) or r.value != 1 for r in sleeps + duration_sources):
        raise ValueError('Sleep episodes must be explicit asleep-state intervals.')
    sleep = daily_sleep([(r.start, r.end) for r in sleeps], zone,
                        main_period=(main[0].start, main[0].end) if main else None) if sleep_complete else {'sleep_midpoint_h': None, 'sleep_duration_h': None}
    # Completed main-period metadata must never add yesterday's sleep to today's duration.
    if sleep_complete:
        clipped = [(max(begin, instant(r.start)), min(cutoff, instant(r.end))) for r in duration_sources]
        sleep['sleep_duration_h'] = daily_sleep(clipped, zone)['sleep_duration_h']
    values.update(sleep)
    for key in sleep:
        parents[key] = sleeps + main
    parents['sleep_duration_h'] = duration_sources
    meals = groups.get('caloric_event', [])
    if any(r.unit != 'kcal' or r.value is None for r in meals):
        raise ValueError('Caloric events require measured energy in kcal.')
    eating = daily_eating([(r.start, r.value) for r in meals], zone, complete=eating_complete)
    values.update(eating)
    for key in eating:
        parents[key] = meals
    activity = groups.get('activity_hour', []); wear = groups.get('wear_minutes_hour', [])
    hourly, wearing = np.full(24, np.nan), np.full(24, np.nan)
    seen = set()
    for variable, items, output in [('activity_hour', activity, hourly), ('wear_minutes_hour', wear, wearing)]:
        for row in items:
            expected_unit = 'count' if variable == 'activity_hour' else 'minute'
            if row.unit != expected_unit:
                raise ValueError('Hourly acquisition unit mismatch.')
            start = instant(row.start).astimezone(ZoneInfo(zone))
            key = (variable, start.hour)
            if key in seen or start.minute or start.second or start.microsecond or (instant(row.end) - instant(row.start)).total_seconds() != 3600:
                raise ValueError('Repeated/DST/unaligned hour requires explicit upstream resampling.')
            seen.add(key); output[start.hour] = row.value
    # Missing wear evidence is not silently promoted to a fully observed actigraphy day.
    act = daily_activity(hourly, wearing) if np.isfinite(wearing).all() else {k: None for k in ('activity_m10_start_h', 'activity_ra', 'activity_total')}
    values.update(act)
    for key in act:
        parents[key] = activity + wear
    resting = groups.get('resting_hr_bpm', [])
    if any(r.unit != 'bpm' for r in resting):
        raise ValueError('Resting heart rate requires bpm.')
    if len(resting) > 1:
        raise ValueError('Multiple device resting summaries for one day.')
    values.update(daily_physiology([], device_resting_hr=resting[0].value if resting else None))
    parents['resting_hr_bpm'] = resting
    # PSEUDOCODE: score only actual supplied descriptions; preserve their lineage and model identity.
    if text_predictor is not None:
        for category in CATEGORIES:
            descriptions = groups.get('text:' + category, [])
            if len(descriptions) > 1:
                raise ValueError('Multiple prompt responses need a prespecified aggregation protocol.')
            if descriptions:
                if descriptions[0].unit != 'text' or not isinstance(descriptions[0].value, str) or not descriptions[0].value.strip():
                    raise ValueError('Prompt response requires nonempty text and unit text.')
                result = text_predictor.predict(category, descriptions[0].value)
                for key, value in result['scores'].items():
                    name = 'text_' + key; values[name] = value; parents[name] = descriptions
                    model_ids[name] = result['model_identity']
                    reasons[name] = result.get('reasons', {}).get(key)
    features = []
    for name in ALL_MEASURES:
        sources = parents.get(name, []); value = values.get(name)
        provenance = build_lineage([r.provenance for r in sources], 'prepare_day:' + name)
        features.append(FeatureValue(name, value, UNITS[name],
            max((instant(r.available_at) for r in sources), default=instant(issued_at)), provenance,
            reason=(reasons.get(name) or 'missing_or_incomplete_source') if value is None else None,
            coverage=(float(wearing.sum() / 1440) if name.startswith('activity_') and value is not None else 1. if value is not None else 0.),
            measured_until=min(cutoff, max((instant(r.end) for r in sources), default=cutoff)),
            model_id=model_ids.get(name)))
    return DailyPanel(participant_id, day, zone, tuple(features))


def rolling_outcome_measures(daily, last_day, *, window=7, minimum=6):
    """daily maps dates to source-backed midpoint/meal/hourly values, independently of DNB."""
    # PSEUDOCODE: align a fixed calendar window -> require paired domain coverage -> calculate S1/E1/A1.
    if type(window) is not int or type(minimum) is not int or not 2 <= minimum <= window:
        raise ValueError('Outcome windows need at least two complete days.')
    rows = [daily.get(last_day - timedelta(days=i), {}) for i in range(window - 1, -1, -1)]
    sleep = [r.get('sleep_midpoint_h', np.nan) for r in rows]
    first = [r.get('first_caloric_h', np.nan) for r in rows]
    last = [r.get('last_caloric_h', np.nan) for r in rows]
    activity = [r.get('activity_hours', [np.nan] * 24) for r in rows]
    return {'S1': sleep_regularity(sleep, minimum), 'E1': eating_regularity(first, last, minimum),
            'A1': activity_regularity(activity, minimum)}
