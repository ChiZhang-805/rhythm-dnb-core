"""Confirm sustained multidomain episodes; onset is backdated only offline."""

from datetime import timedelta
from ..contracts import OutcomeEvent
from ..timebase import local_boundary, instant


def detect_events(assessments, zone, *, persistence=3, first_only=True):
    # PSEUDOCODE: sort daily assessments -> break on gaps/unknown -> confirm first sustained run.
    if type(persistence) is not int or persistence < 1:
        raise ValueError('Persistence must be a positive integer.')
    rows = sorted(assessments, key=lambda r: r.day)
    if len({r.participant_id for r in rows}) > 1 or len({r.day for r in rows}) != len(rows):
        raise ValueError('Events require unique days for one person.')
    for row in rows:
        if type(row.evaluable) is not bool or len(set(row.abnormal_domains)) != len(row.abnormal_domains) or not set(row.abnormal_domains) <= {'S1', 'E1', 'A1'}:
            raise ValueError('Invalid or repeated outcome domains.')
        if instant(row.available_at) < instant(local_boundary(row.day + timedelta(days=1), zone)):
            raise ValueError('Assessment cannot be available before its research day ends.')
        if not row.evaluable and row.abnormal_domains:
            raise ValueError('Unevaluable assessments cannot declare abnormal domains.')
    events, run, previous, active = [], [], None, False
    for row in rows:
        gap = previous is not None and row.day != previous + timedelta(days=1)
        if gap or not row.evaluable or len(row.abnormal_domains) < 2:
            run = []
            # Missingness does not prove recovery, so it cannot re-arm recurrence detection.
            if row.evaluable and len(row.abnormal_domains) < 2:
                active = False
        if row.evaluable and len(row.abnormal_domains) >= 2:
            run.append(row)
            if len(run) == persistence and not active:
                onset = instant(local_boundary(run[0].day + timedelta(days=1), zone))
                confirmed = max(instant(r.available_at) for r in run)
                if confirmed < onset:
                    raise ValueError('Event confirmation precedes onset.')
                events.append(OutcomeEvent(row.participant_id, onset, confirmed))
                active = True
                if first_only:
                    break
        previous = row.day
    return tuple(events)
