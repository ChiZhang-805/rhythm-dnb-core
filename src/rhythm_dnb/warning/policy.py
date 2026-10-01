"""Daily alarm state machine shared by online inference and offline calibration."""

from datetime import timedelta
import math
from ..contracts import AlarmState


def advance(state, participant_id, bundle_id, day, score, threshold, *, consecutive=2, cooldown_days=7, context=''):
    """Return warning/status/state. A missing score resets persistence and yields None."""
    # PSEUDOCODE: bind state -> enforce monotone dates -> count adjacent exceedances -> apply cooldown.
    if (state.last_day is not None or state.last_alarm_day is not None or state.consecutive) and not all(
            isinstance(value, str) and value.strip() for value in (state.participant_id, state.bundle_id)):
        raise ValueError('Persisted alarm history requires person and bundle identities.')
    if state.participant_id and (state.participant_id != participant_id or state.bundle_id != bundle_id):
        raise ValueError('Alarm state belongs to another person or bundle.')
    if state.participant_id and state.context != context:
        raise ValueError('Alarm state belongs to another method, timezone or policy.')
    if type(state.consecutive) is not int or state.consecutive < 0 or (state.last_alarm_day is not None and (state.last_day is None or state.last_alarm_day > state.last_day)):
        raise ValueError('Invalid persisted alarm state.')
    if state.last_day is not None and day <= state.last_day:
        raise ValueError('Replay/out-of-order forecast: reuse its saved response instead.')
    if type(consecutive) is not int or consecutive < 1 or type(cooldown_days) is not int or cooldown_days < 0:
        raise ValueError('Invalid alarm policy.')
    if threshold is not None and (type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0):
        raise ValueError('Invalid threshold.')
    if score is not None and (type(score) not in (int, float) or not math.isfinite(score) or score < 0):
        raise ValueError('Invalid score.')
    last_alarm = state.last_alarm_day
    run = state.consecutive if state.last_day == day - timedelta(days=1) else 0
    run = run + 1 if score is not None and threshold is not None and score > threshold else 0
    if score is None:
        warning, status = None, 'insufficient_data'
    elif threshold is None:
        warning, status = None, 'uncalibrated'
    elif last_alarm is not None and (day - last_alarm).days < cooldown_days:
        warning, status = 0, 'cooldown'
    elif run >= consecutive:
        warning, status, last_alarm, run = 1, 'warning', day, 0
    else:
        warning, status = 0, 'monitoring'
    return warning, status, AlarmState(participant_id, bundle_id, day, last_alarm, run, context)
