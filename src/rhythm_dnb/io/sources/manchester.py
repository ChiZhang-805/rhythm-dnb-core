"""Legacy raw-source parser; explicit path, retrospective semantics retained.

Returned legacy records are NOT automatically prospective Observation objects.
Use the hospital/normalized adapter after supplying verified time and lineage.
"""
"""Import daily Fitbit sleep from the Manchester light/sleep study."""

from datetime import datetime, time, timedelta
from hashlib import sha256
import json
import math
from pathlib import Path
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import pyreadr

from .legacy_schema import validate_record


ARTICLE = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC13354558/'
REPOSITORY = 'https://github.com/altugdidikoglu/light-sleep-inreallife'
COMMIT = 'ef570332cd6bd04cafea40a66f5a70e6f2206505'
FILES = {
    'sleep_fitbit.RData': 'd6ca44f158ef162e13394bcbc16246523bb190037fa4a262f256e4209a8bd095',
    'sleep_diary.RData': '449085c3117b5b1ed9cc56b7f2dc02d8191dd898fa598952e93424fb46194176',
    'baseline_data.RData': 'cf485c46e9db456693f7fb3ddccd1ac54315d886009ea318aae10395f800ef5d',
}
TIMEZONE = ZoneInfo('Europe/London')




def _read_frame(path, expected_name):
    # PSEUDOCODE: validate inputs -> read frame -> return the fixed semantic contract.
    objects = pyreadr.read_r(str(path))
    if set(objects) != {expected_name}:
        raise ValueError('Unexpected Manchester RData objects: ' + str(path))
    return objects[expected_name]


def _local_datetime(value):
    # PSEUDOCODE: validate inputs -> local datetime -> return the fixed semantic contract.
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError('Manchester sleep timestamp missing or invalid') from error
    if pd.isna(value) or not isinstance(value, datetime):
        raise ValueError('Manchester sleep timestamp missing or invalid')
    return value.astimezone(TIMEZONE) if value.tzinfo else value.replace(tzinfo=TIMEZONE)


def _number(value, maximum):
    # PSEUDOCODE: validate inputs -> number -> return the fixed semantic contract.
    if isinstance(value, bool) or pd.isna(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and 0 <= result <= maximum else None


def _hour(value):
    # PSEUDOCODE: validate inputs -> hour -> return the fixed semantic contract.
    return value.hour + value.minute / 60 + value.second / 3600


def assemble_records(fitbit, diary, baseline):
    """Keep labels empty; source eligibility is not a clinical negative diagnosis."""
    # PSEUDOCODE: validate inputs -> assemble records -> return the fixed semantic contract.
    participants = {str(row['Id']) for _, row in baseline.iterrows()}
    diary_by_night = {}
    for position, (_, row) in enumerate(diary.iterrows(), 2):
        person = str(row['Id'])
        wake = _local_datetime(row['datetimeWake'])
        key = (person, wake.date())
        if key in diary_by_night:
            raise ValueError('Duplicate Manchester diary wake date: ' + str(key))
        diary_by_night[key] = (position, row)
    records = []
    seen = set()
    matched = set()
    for position, (_, row) in enumerate(fitbit.iterrows(), 2):
        person = str(row['Id'])
        if person not in participants:
            raise ValueError('Manchester Fitbit participant missing from baseline: ' + person)
        start = _local_datetime(row['startTime'])
        end = _local_datetime(row['endTime'])
        key = (person, end.date())
        if key in seen:
            raise ValueError('Duplicate Manchester Fitbit wake date: ' + str(key))
        seen.add(key)
        duration = _number(row['sleepDuration'], 1440)
        if end <= start or duration is None or duration > (end - start).total_seconds() / 60 + 1e-6:
            raise ValueError('Invalid Manchester Fitbit sleep episode: ' + str(key))
        wake_date = end.date()
        anchor = datetime.combine(wake_date + timedelta(days=1), time(12), TIMEZONE)
        provenance = {
            'source': {'article': ARTICLE, 'repository': REPOSITORY, 'commit': COMMIT,
                       'files_sha256': FILES, 'fitbit_row': position,
                       'source_wake_date': wake_date.isoformat()},
            'observed_at': {'meaning': 'Nominal next-day local-noon summary anchor; diary submission time unavailable'},
        }
        warnings = ['No independently assessed circadian-disorder outcome; rhythm_state_gt is unknown.',
                    'Source Fitbit times are local Europe/London; summary anchor is nominal, not a measured time.']
        raw = {'record_id': f'Manchester-{person}-{wake_date.isoformat()}',
               'participant_id': f'Manchester-{person}',
               'observed_at': anchor.isoformat(),
               'source_dataset': 'Manchester-Light-Sleep', 'split': 'train',
               'sleep_start_hour': _hour(start), 'sleep_end_hour': _hour(end),
               'sleep_duration_h': duration / 60,
               'provenance': provenance, 'warnings': warnings}
        for target, source_key, value, conversion in (
            ('sleep_start_hour', 'startTime', raw['sleep_start_hour'], 'local decimal hour'),
            ('sleep_end_hour', 'endTime', raw['sleep_end_hour'], 'local decimal hour'),
            ('sleep_duration_h', 'sleepDuration', raw['sleep_duration_h'], 'minutes/60'),
        ):
            provenance[target] = {'file': 'sleep_fitbit.RData', 'row': position,
                                  'source_key': source_key, 'conversion': conversion}
        if key in diary_by_night:
            matched.add(key)
            diary_position, diary_row = diary_by_night[key]
            diary_duration = _number(diary_row['sleepDuration'], 1440)
            efficiency = _number(diary_row['sleepEfficiency'], 100)
            in_bed = diary_duration * 100 / efficiency if diary_duration is not None and efficiency else None
            if in_bed is not None and duration <= in_bed + 1e-6 and 0 <= in_bed <= 1440:
                raw['time_in_bed_h'] = in_bed / 60
                provenance['time_in_bed_h'] = {'file': 'sleep_diary.RData', 'row': diary_position,
                                              'source_keys': ['sleepDuration', 'sleepEfficiency'],
                                              'conversion': 'reported minutes / (reported percent / 100) / 60'}
                warnings.append('Diary in-bed estimate may have been submitted after the nominal summary anchor.')
            else:
                warnings.append('Diary in-bed estimate missing or shorter than Fitbit sleep; left blank.')
        records.append(validate_record(raw))
    if matched != set(diary_by_night):
        raise ValueError('Manchester diary night without matching Fitbit sleep episode')
    return sorted(records, key=lambda row: row['record_id'])


def build_records(root, verify=True):
    # PSEUDOCODE: validate inputs -> build records -> return the fixed semantic contract.
    root = Path(root)
    for name, expected in FILES.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError('Manchester source missing: ' + str(path))
        if verify and sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Manchester source SHA-256 differs: ' + name)
    fitbit = _read_frame(root / 'sleep_fitbit.RData', 'sleep_fitbit_final')
    diary = _read_frame(root / 'sleep_diary.RData', 'sleep_diary_final')
    baseline = _read_frame(root / 'baseline_data.RData', 'baseline_data_final')
    records = assemble_records(fitbit, diary, baseline)
    if verify and (len(records), len({r['participant_id'] for r in records})) != (547, 89):
        raise ValueError('Manchester verified source population differs from published files')
    return records
