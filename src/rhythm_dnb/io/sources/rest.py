"""Legacy raw-source parser; explicit path, retrospective semantics retained.

Returned legacy records are NOT automatically prospective Observation objects.
Use the hospital/normalized adapter after supplying verified time and lineage.
"""
"""Import REST athlete study days with explicitly synthetic calendar anchors."""

import csv
from collections import defaultdict
from datetime import date, timedelta
from hashlib import sha256
import math
from pathlib import Path

from .legacy_schema import validate_record


FILE = 'daily_responses.csv'
PUBLISHER_URL = 'https://zenodo.org/records/16937033'
ARCHIVE_URL = PUBLISHER_URL + '/files/REST.zip?download=1'
ARCHIVE_MD5 = 'c32dbd9ec817ddd3ac111ecf67d222f9'
PUBLISHED_FILE_SHA256 = 'ed98033c88648e3ddeb72ae284a56c310c5e293ec75a13a51a4361a5350f8b28'
WEEKDAYS = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
NOMINAL_MONDAY = date(2000, 1, 3)




def _group_days(path):
    # PSEUDOCODE: preserve source order -> verify consecutive weekdays -> group responses by participant-relative day.
    by_person = defaultdict(list)
    with path.open(encoding='utf-8-sig', newline='') as source:
        for line, row in enumerate(csv.DictReader(source), 2):
            participant = row['id']
            weekday = row['weekday']
            if not participant or weekday not in WEEKDAYS:
                raise ValueError(f'Invalid REST identity/weekday at line {line}.')
            by_person[participant].append((line, row))
    for participant, rows in by_person.items():
        origin_weekday = WEEKDAYS.index(rows[0][1]['weekday'])
        day_index = 0
        grouped = defaultdict(list)
        prior = origin_weekday
        for line, row in rows:
            current = WEEKDAYS.index(row['weekday'])
            if current != prior:
                if current != (prior + 1) % 7:
                    raise ValueError(f'Nonconsecutive REST weekdays for {participant} at line {line}.')
                day_index += 1
            grouped[day_index].append((line, row))
            prior = current
        for index, entries in sorted(grouped.items()):
            yield participant, origin_weekday, index, entries


def _agreed(entries, key):
    # PSEUDOCODE: retain the sole agreed nonempty value -> flag conflicting same-day responses.
    values = {row[key].strip() for _, row in entries if row[key] and row[key].strip()}
    return next(iter(values)) if len(values) == 1 else None, len(values) > 1


def _number(value, minimum=0, maximum=None):
    # PSEUDOCODE: parse a finite source number -> reject values outside documented bounds.
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < minimum or maximum is not None and result > maximum:
        return None
    return result


def _clock(value):
    # PSEUDOCODE: parse HH:MM:SS -> verify clock bounds -> return fractional local hour.
    if not value:
        return None
    parts = value.split(':')
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = map(int, parts)
    except ValueError:
        return None
    if not (0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 60):
        return None
    return hours + minutes / 60 + seconds / 3600


def build_records(root, verify=True):
    """One row per relative day; conflicting duplicate responses stay missing."""
    # PSEUDOCODE: verify the source digest -> retain relative dates explicitly -> convert only agreed measured fields.
    path = Path(root) / FILE
    digest = sha256(path.read_bytes()).hexdigest()
    if verify and digest != PUBLISHED_FILE_SHA256:
        raise ValueError('REST daily source SHA-256 differs from the published selection.')
    records = []
    for participant, origin_weekday, day_index, entries in _group_days(path):
        nominal = NOMINAL_MONDAY + timedelta(days=origin_weekday + day_index)
        warnings = ['Publisher removed actual dates; observed_at is a relative-day bookkeeping anchor, not real UTC time.',
                    'Do not align calendar date or season with other cohorts.']
        provenance = {'source': {'publisher': PUBLISHER_URL, 'source_file': FILE,
                                 'source_sha256': digest, 'source_lines': [line for line, _ in entries],
                                 'relative_day_index': day_index,
                                 'source_weekday': entries[0][1]['weekday'],
                                 'calendar_date_removed_by_publisher': True},
                      'observed_at': {'meaning': 'Synthetic date aligned only to weekday and within-person day order'}}
        record = {'record_id': f'REST-{participant}-d{day_index:02d}',
                  'participant_id': f'REST-{participant}',
                  'observed_at': nominal.isoformat() + 'T12:00:00+00:00',
                  'source_dataset': 'REST-relative-day', 'split': 'train',
                  'warnings': warnings, 'provenance': provenance}

        def source_value(key):
            # PSEUDOCODE: resolve duplicate responses -> record a warning when conflicting values must remain missing.
            value, conflict = _agreed(entries, key)
            if conflict:
                warnings.append(f'Duplicate same-day source values conflict for {key}; field omitted.')
            return value

        sensor_minutes = _number(source_value('total_sleep_time'), 0, 1440)
        reported_hours = _number(source_value('sleepdura_sr'), 0, 24)
        if sensor_minutes is not None:
            record['sleep_duration_h'] = sensor_minutes / 60
            provenance['sleep_duration_h'] = {'source_key': 'total_sleep_time',
                                               'conversion': 'sensor minutes/60'}
            if sensor_minutes < 120:
                warnings.append('Very short actigraphy sleep estimate; check nonwear before analysis.')
        elif reported_hours is not None:
            record['sleep_duration_h'] = reported_hours
            provenance['sleep_duration_h'] = {'source_key': 'sleepdura_sr',
                                               'meaning': 'self-reported hours; sensor missing'}
        for target, key in (('sleep_start_hour', 'sleep_onset'),
                            ('sleep_end_hour', 'sleep_offset')):
            value = _clock(source_value(key))
            if value is not None:
                record[target] = value
                provenance[target] = {'source_key': key, 'meaning': 'local clock without actual date/timezone'}
        caffeine = _number(source_value('caffeine_mg'), 0, 2000)
        if caffeine is not None:
            record['caffeine_mg'] = caffeine
            provenance['caffeine_mg'] = {'source_key': 'caffeine_mg',
                'meaning': 'publisher-derived estimate from reported beverage counts'}
        fatigue = _number(source_value('fatigue'), 1, 5)
        if fatigue is not None and fatigue.is_integer():
            record['fatigue_score'] = math.floor((5 - fatigue) * 2.5 + 0.5)
            provenance['fatigue_score'] = {'source_key': 'fatigue',
                'source_value': int(fatigue),
                'conversion': 'round_half_up((5 - source_1_to_5) * 2.5)',
                'scale': '1=very fatigued, 5=not fatigued (publisher questionnaire)'}
        records.append(validate_record(record))
    return records
