"""Prospective target labels, generated offline with censoring and confirmation."""

from datetime import timedelta
from zoneinfo import ZoneInfo
from ..contracts import OutcomeLabel
from ..timebase import instant, local_boundary


def future_label(issued_at, events, *, followup_end, observed_days, horizon_days=7,
                 min_lead_hours=24, confirmation_days=2, label_as_of=None, timezone='UTC'):
    """observed_days are local research-day dates of evaluable assessments (v2).

    Negative labels need full follow-up through horizon + confirmation days;
    unknown coverage, unconfirmed outcomes and prevalent events never become zero.
    """
    # PSEUDOCODE: exclude prevalent/too-soon events -> confirm future positive -> otherwise check censoring.
    if any(type(n) is not int for n in (horizon_days, min_lead_hours, confirmation_days)) or horizon_days < 1 or not 0 < min_lead_hours <= horizon_days * 24 or confirmation_days < 0:
        raise ValueError('Invalid forecast horizon or confirmation window.')
    t = instant(issued_at); end = t + timedelta(days=horizon_days)
    events = sorted(events, key=lambda e: instant(e.onset))
    if len({e.participant_id for e in events}) > 1 or any(instant(e.confirmed_at) < instant(e.onset) for e in events):
        raise ValueError('Outcome events must belong to one person and have valid confirmation times.')
    if any(instant(e.onset) <= t for e in events):
        return OutcomeLabel(None, 'not_at_risk_after_first_event')
    future = next((e for e in events if instant(e.onset) <= end), None)
    if future and instant(future.onset) < t + timedelta(hours=min_lead_hours):
        return OutcomeLabel(None, 'event_inside_minimum_lead')
    needed_end = instant(future.confirmed_at) if future else end + timedelta(days=confirmation_days)
    known_until = instant(label_as_of) if label_as_of is not None else instant(followup_end)
    if instant(followup_end) < needed_end or known_until < needed_end:
        return OutcomeLabel(None, 'right_censored_or_unconfirmed')
    first = t.astimezone(ZoneInfo(timezone)).date() - timedelta(days=1)
    last = needed_end.astimezone(ZoneInfo(timezone)).date()
    required = {first + timedelta(days=i) for i in range((last - first).days + 1)
                if t < instant(local_boundary(first + timedelta(days=i + 1), timezone)) <= needed_end}
    if not required <= set(observed_days):
        return OutcomeLabel(None, 'outcome_coverage_gap')
    return OutcomeLabel(1, 'confirmed_future_event', future.onset) if future else OutcomeLabel(0, 'complete_event_free_followup')
