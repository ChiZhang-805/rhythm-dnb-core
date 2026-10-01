"""Local research days and UTC comparisons, including daylight-saving changes."""

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


def instant(value: datetime | str) -> datetime:
    """Return a UTC instant; reject naive or nonexistent local times."""
    # PSEUDOCODE: parse -> require an offset -> check round-trip -> normalize UTC.
    value = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('An offset-aware timestamp is required.')
    result = value.astimezone(timezone.utc)
    if result.astimezone(value.tzinfo).replace(tzinfo=None) != value.replace(tzinfo=None):
        raise ValueError('Nonexistent local timestamp.')
    return result


def local_boundary(day: date, zone: str, hour: int = 4) -> datetime:
    # PSEUDOCODE: construct a wall-clock boundary and validate its UTC round-trip.
    result = datetime.combine(day, time(hour), ZoneInfo(zone))
    instant(result)
    return result


def research_day(value: datetime, zone: str, boundary_hour: int = 4) -> date:
    # PSEUDOCODE: convert to the declared local zone; assign pre-boundary events to yesterday.
    local = instant(value).astimezone(ZoneInfo(zone))
    return local.date() - timedelta(days=int(local.hour < boundary_hour))


def forecast_bounds(issued_at: datetime, zone: str) -> tuple[date, datetime, datetime]:
    """Issue at local noon for the completed day ending at local 04:00."""
    # PSEUDOCODE: enforce noon schedule -> derive completed day and strict data cutoff.
    local = instant(issued_at).astimezone(ZoneInfo(zone))
    if local.time().replace(tzinfo=None) != time(12):
        raise ValueError('Study v1 forecasts must be issued at local 12:00:00.')
    day = local.date() - timedelta(days=1)
    return day, instant(local_boundary(local.date(), zone)), instant(issued_at)


def available_as_of(observation, cutoff: datetime, issued_at: datetime) -> bool:
    # PSEUDOCODE: require the full event and its arrival to precede the allowed boundaries.
    return instant(observation.start) < instant(cutoff) and instant(observation.end) <= instant(cutoff) and instant(observation.available_at) <= instant(issued_at)


def clock_delta(value: float, center: float, period: float = 24.0) -> float:
    # PSEUDOCODE: unwrap a periodic value to the signed half-period around a fixed center.
    if period <= 0:
        raise ValueError('period must be positive.')
    return (value - center + period / 2) % period - period / 2
