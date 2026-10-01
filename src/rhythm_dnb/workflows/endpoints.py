"""Offline endpoint construction with frozen criteria and explicit availability.

The caller supplies source-backed daily endpoint inputs, including an exposure-
normalized 24-hour activity profile. No DNB score or forecast enters this module.
"""

from dataclasses import dataclass
from ..definitions import MEASUREMENT_ID
from datetime import date, datetime, timedelta
from ..contracts import Provenance, OutcomeAssessment, OutcomeEvent, OutcomeLabel
from ..provenance import eligible_measurement, fingerprint
from ..timebase import instant, local_boundary
from ..outcomes.criteria import personal_anchor, assess_day, fit_criteria
from ..outcomes.events import detect_events
from ..outcomes.labels import future_label
from .prepare import rolling_outcome_measures


@dataclass(frozen=True)
class EndpointDay:
    participant_id: str
    day: date
    timezone: str
    values: dict
    available_at: datetime
    provenance: Provenance


def fit_endpoint_criteria(stable_people, config, cutoff):
    # PSEUDOCODE: pass registered endpoint quantile and independent-person minimum into threshold fitting.
    return fit_criteria(stable_people, cutoff, quantile=config.threshold_quantile, minimum_people=config.endpoint_min_people)


def _endpoint_protocol(criteria_id, config):
    # PSEUDOCODE: bind the endpoint rules and forecast target to the same immutable protocol identity.
    return {'criteria_id': criteria_id, 'baseline_days': config.baseline_days,
            'window': config.outcome_window, 'minimum': config.outcome_min_days,
            'persistence': config.persistence_days, 'measurement_id': MEASUREMENT_ID,
            'threshold_quantile': config.threshold_quantile,
            'horizon_days': config.horizon_days, 'min_lead_hours': config.min_lead_hours}


def build_endpoint_timeline(rows, baseline_start, criteria, config, *, as_of):
    # PSEUDOCODE: validate independent causal endpoint evidence -> freeze baseline -> assess post-baseline days.
    rows = sorted(rows, key=lambda r: r.day)
    if not rows or len({(r.participant_id, r.timezone) for r in rows}) != 1 or len({r.day for r in rows}) != len(rows):
        raise ValueError('Endpoint timeline requires unique days for one person and timezone.')
    person, zone = rows[0].participant_id, rows[0].timezone
    if not isinstance(person, str) or not person.strip() or type(baseline_start) is not date:
        raise ValueError('Endpoint timelines require a person identity and baseline start date.')
    if criteria.get('id') != fingerprint({k: v for k, v in criteria.items() if k != 'id'}):
        raise ValueError('Endpoint criteria checksum mismatch.')
    if criteria['quantile'] != config.threshold_quantile:
        raise ValueError('Endpoint quantile differs from the registered study.')
    mapping = {}
    for row in rows:
        if instant(row.available_at) < instant(local_boundary(row.day + timedelta(days=1), zone)):
            raise ValueError('Endpoint evidence precedes the end of measurement.')
        if instant(row.available_at) <= instant(as_of) and eligible_measurement(row.provenance):
            mapping[row.day] = row
    end_baseline = baseline_start + timedelta(days=config.baseline_days)
    if instant(local_boundary(end_baseline, zone)) > instant(as_of):
        raise ValueError('Personal baseline has not ended.')
    daily = {d: row.values for d, row in mapping.items()}
    endings = [baseline_start + timedelta(days=i) for i in range(config.outcome_window - 1, config.baseline_days)]
    baseline = {d: rolling_outcome_measures(daily, d, window=config.outcome_window, minimum=config.outcome_min_days) for d in endings}
    anchor = personal_anchor(baseline, baseline_start, baseline_days=config.baseline_days,
                             window=config.outcome_window, minimum=config.outcome_min_days)
    baseline_available = max([instant(local_boundary(end_baseline, zone))] +
                             [instant(row.available_at) for d, row in mapping.items() if baseline_start <= d < end_baseline])
    assessments = []
    last_day = max(mapping, default=baseline_start)
    for offset in range(max(0, (last_day - end_baseline).days + 1)):
        day = end_baseline + timedelta(days=offset)
        measures = rolling_outcome_measures(daily, day, window=config.outcome_window, minimum=config.outcome_min_days)
        inputs = [r for d, r in mapping.items() if day - timedelta(days=config.outcome_window - 1) <= d <= day]
        known = max([baseline_available, instant(local_boundary(day + timedelta(days=1), zone))] + [instant(r.available_at) for r in inputs])
        if known <= instant(as_of):
            assessments.append(assess_day(person, day, measures, anchor, criteria, known))
    events = detect_events(assessments, zone, persistence=config.persistence_days)
    protocol = _endpoint_protocol(criteria['id'], config)
    return {'participant_id': person, 'timezone': zone, 'anchor': anchor, 'baseline_available_at': baseline_available,
            'baseline_start': baseline_start, 'as_of': instant(as_of), 'protocol': protocol,
            'assessments': tuple(assessments), 'events': events, 'protocol_id': fingerprint(protocol)}


def parse_timeline(payload):
    # PSEUDOCODE: restore an endpoint artifact without guessing missing protocol or time metadata.
    required = {'participant_id', 'timezone', 'anchor', 'baseline_available_at', 'baseline_start',
                'as_of', 'protocol', 'assessments', 'events', 'protocol_id'}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError('Endpoint timeline fields differ; rebuild legacy artifacts from source.')
    result = dict(payload)
    result['baseline_start'] = date.fromisoformat(result['baseline_start'])
    for key in ('baseline_available_at', 'as_of'):
        result[key] = instant(result[key])
    assessments = []
    for row in result['assessments']:
        item = dict(row)
        item.update(day=date.fromisoformat(item['day']), available_at=instant(item['available_at']),
                    abnormal_domains=tuple(item['abnormal_domains']))
        assessments.append(OutcomeAssessment(**item))
    result['assessments'] = tuple(assessments)
    result['events'] = tuple(OutcomeEvent(**{**row, 'onset': instant(row['onset']),
                                            'confirmed_at': instant(row['confirmed_at'])}) for row in result['events'])
    return result


def label_timeline(issued_at, timeline, config, *, followup_end, as_of):
    # PSEUDOCODE: verify the frozen protocol, person and knowledge cutoff -> label without refitting outcomes.
    protocol = timeline.get('protocol')
    if (not isinstance(protocol, dict) or not isinstance(protocol.get('criteria_id'), str) or not protocol['criteria_id']
            or protocol != _endpoint_protocol(protocol['criteria_id'], config)
            or timeline.get('protocol_id') != fingerprint(protocol)):
        raise ValueError('Endpoint protocol differs from the study; rebuild the timeline from source.')
    person, zone = timeline['participant_id'], timeline['timezone']
    if not isinstance(person, str) or not person.strip():
        raise ValueError('Endpoint timeline lacks a participant identity.')
    baseline_end = timeline['baseline_start'] + timedelta(days=config.baseline_days)
    if (instant(as_of) < instant(timeline['as_of']) or instant(issued_at) > instant(as_of)
            or not instant(local_boundary(baseline_end, zone)) <= instant(timeline['baseline_available_at']) <= instant(timeline['as_of'])):
        raise ValueError('Timeline or personal baseline was not available at the requested label cutoff.')
    assessments = timeline['assessments']
    if any(r.participant_id != person or r.day < baseline_end or
           not instant(timeline['baseline_available_at']) <= instant(r.available_at) <= instant(timeline['as_of']) for r in assessments):
        raise ValueError('Endpoint assessments have inconsistent participants or availability.')
    expected = detect_events(assessments, zone, persistence=config.persistence_days)
    if tuple(timeline['events']) != expected:
        raise ValueError('Endpoint events differ from the confirmed daily assessments.')
    if timeline['anchor'] is None or instant(issued_at) < instant(timeline['baseline_available_at']):
        return OutcomeLabel(None, 'personal_endpoint_baseline_unavailable')
    return future_label(issued_at, timeline['events'], followup_end=followup_end,
        observed_days={r.day for r in timeline['assessments'] if r.evaluable and instant(r.available_at) <= instant(as_of)},
        horizon_days=config.horizon_days, min_lead_hours=config.min_lead_hours,
        confirmation_days=config.persistence_days - 1, label_as_of=as_of, timezone=timeline['timezone'])
