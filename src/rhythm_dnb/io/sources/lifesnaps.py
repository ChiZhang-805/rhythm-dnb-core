"""Legacy raw-source parser; explicit path, retrospective semantics retained.

Returned legacy records are NOT automatically prospective Observation objects.
Use the hospital/normalized adapter after supplying verified time and lineage.
"""
"""Import comparable measured LifeSnaps daily features without outcome labels."""

import csv
from collections import defaultdict
from datetime import date
from hashlib import sha256
import math
from pathlib import Path
from statistics import mean, stdev

from .legacy_schema import validate_record


ARCHIVE_URL = 'https://zenodo.org/records/7229547/files/rais_anonymized.zip?download=1'
PUBLISHER_URL = 'https://zenodo.org/records/7229547'
ARCHIVE_MD5 = '726afe263ab4b900a721eac19b2ca13a'
FILES = {
    'daily_fitbit_sema_df_unprocessed.csv': '82c84ee495be4b0ff636a8a44271ffb124e2358d9dd52012b529b897f3d54963',
    'hourly_fitbit_sema_df_unprocessed.csv': '99ecc8e2e0a5d7cfd766de835e97aef62b08d1ff767a73185f2d14fed0aed8fd',
}




def _number(value, minimum=0, maximum=None):
    # PSEUDOCODE: parse a finite source number -> retain only values within documented input bounds.
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < minimum or maximum is not None and result > maximum:
        return None
    return result


def _hourly_sd(path):
    # PSEUDOCODE: group valid hourly heart rates by person/date -> require eight hours before sample SD.
    by_hour = defaultdict(list)
    with path.open(encoding='utf-8-sig', newline='') as source:
        for row in csv.DictReader(source):
            bpm = _number(row['bpm'], 1, 400)
            hour = _number(row['hour'], 0, 23)
            if bpm is not None and hour is not None:
                by_hour[(row['id'], row['date'], hour)].append(bpm)
    by_day = defaultdict(list)
    for (participant, day, _), values in by_hour.items():
        by_day[(participant, day)].append(mean(values))
    return {key: (stdev(values), len(values)) for key, values in by_day.items()
            if len(values) >= 8}


def build_records(root, verify=True):
    """Keep only source-compatible fields; date-only rows use a nominal anchor."""
    # PSEUDOCODE: verify selected source files -> extract comparable daily measures -> retain missing outcomes.
    root = Path(root)
    paths = {name: root / name for name in FILES}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError('LifeSnaps selected file missing: ' + str(path))
        if verify and sha256(path.read_bytes()).hexdigest() != FILES[name]:
            raise ValueError('LifeSnaps source SHA-256 differs: ' + name)
    hourly = _hourly_sd(paths['hourly_fitbit_sema_df_unprocessed.csv'])
    records = []
    with paths['daily_fitbit_sema_df_unprocessed.csv'].open(encoding='utf-8-sig', newline='') as source:
        for line, item in enumerate(csv.DictReader(source), 2):
            participant, day = item['id'], item['date']
            if not participant or date.fromisoformat(day).isoformat() != day:
                raise ValueError(f'Invalid LifeSnaps identity at line {line}.')
            provenance = {'source': {'publisher': PUBLISHER_URL,
                                     'daily_sha256': FILES['daily_fitbit_sema_df_unprocessed.csv'] if verify else None,
                                     'hourly_sha256': FILES['hourly_fitbit_sema_df_unprocessed.csv'] if verify else None,
                                     'daily_line': line, 'source_local_date': day},
                          'observed_at': {'meaning': 'Nominal UTC noon storage anchor for a local date with unknown timezone; not a measurement time'}}
            record = {'record_id': f'LifeSnaps-{participant}-{day}',
                      'participant_id': f'LifeSnaps-{participant}',
                      'observed_at': day + 'T12:00:00+00:00',
                      'source_dataset': 'LifeSnaps', 'split': 'train',
                      'warnings': ['Local date only; storage noon is not a real measurement timestamp.',
                                   'No independently adjudicated rhythm-disorder outcome.'],
                      'provenance': provenance}
            for key, source_key, minimum, maximum in (
                ('sleep_duration_h', 'minutesAsleep', 0, 1440),
                ('time_in_bed_h', 'sleep_duration', 0, 86400000),
                ('resting_hr_bpm', 'resting_hr', 1, 400),
                ('heart_rate_bpm', 'bpm', 1, 400),
                ('spo2_pct', 'spo2', 0, 100),
            ):
                value = _number(item[source_key], minimum, maximum)
                if value is None:
                    continue
                conversion = 'minutes/60' if key == 'sleep_duration_h' else (
                    'milliseconds/3600000' if key == 'time_in_bed_h' else 'identity')
                record[key] = (value / 60 if key == 'sleep_duration_h' else
                               value / 3600000 if key == 'time_in_bed_h' else value)
                provenance[key] = {'source_file': 'daily_fitbit_sema_df_unprocessed.csv',
                                   'source_line': line, 'source_key': source_key,
                                   'conversion': conversion}
            if record.get('time_in_bed_h') is not None and record.get('sleep_duration_h') is not None:
                if record['time_in_bed_h'] + 1e-6 < record['sleep_duration_h']:
                    record.pop('time_in_bed_h')
                    provenance.pop('time_in_bed_h')
                    record['warnings'].append('Sleep duration exceeded in-bed duration; in-bed value omitted.')
            hourly_value = hourly.get((participant, day))
            if hourly_value:
                sd, n_hours = hourly_value
                record['hr_sd_bpm'] = sd
                provenance['hr_sd_bpm'] = {'source_file': 'hourly_fitbit_sema_df_unprocessed.csv',
                    'source_key': 'bpm', 'calculation': 'sample SD of unique hourly mean BPM',
                    'valid_hours': n_hours, 'minimum_valid_hours': 8}
            if any(key in record for key in ('sleep_duration_h', 'time_in_bed_h',
                                              'resting_hr_bpm', 'heart_rate_bpm',
                                              'spo2_pct', 'hr_sd_bpm')):
                records.append(validate_record(record))
    return records
