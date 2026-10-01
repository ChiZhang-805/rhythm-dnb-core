"""Legacy raw-source parser; explicit path, retrospective semantics retained.

Returned legacy records are NOT automatically prospective Observation objects.
Use the hospital/normalized adapter after supplying verified time and lineage.
"""
"""Import the latest Figshare Sleepmeter diary into the quantitative database."""

import csv
from collections import defaultdict
from datetime import datetime, timedelta
from hashlib import md5, sha256
from pathlib import Path
from urllib.request import urlopen

from .legacy_schema import validate_record


SOURCE_URL = 'https://figshare.com/articles/dataset/sleep-diary-sleepmeter-v2/20362908'
PUBLISHED_MD5 = '3ce68fe506a91d2a2c21cd4f3dfaba04'
DOWNLOAD_URL = 'https://ndownloader.figshare.com/files/36561603'
PARTICIPANT = 'SleepDiary-author'
HEADERS = ('wake', 'sleep', 'bedtime', 'holes', 'type', 'dreams', 'aid',
           'hindrances', 'tags', 'quality', 'notes')




def _read_sessions(path):
    # PSEUDOCODE: validate inputs -> read sessions -> return the fixed semantic contract.
    with Path(path).open(encoding='utf-8-sig', newline='') as source:
        reader = csv.reader(source)
        for line, fields in enumerate(reader, 1):
            if tuple(fields) == HEADERS:
                break
        else:
            raise ValueError('Sleepmeter session table not found.')
        for line, fields in enumerate(reader, line + 1):
            if not fields:
                continue
            if len(fields) != len(HEADERS):
                raise ValueError(f'Malformed Sleepmeter session at line {line}.')
            row = dict(zip(HEADERS, fields))
            row['_line'] = line
            for key in ('wake', 'sleep', 'bedtime'):
                row[key] = datetime.strptime(row[key], '%Y-%m-%d %H:%M%z')
            if not row['bedtime'] <= row['sleep'] <= row['wake']:
                raise ValueError(f'Sleepmeter times out of order at line {line}.')
            yield row


def _hole_minutes(holes, session_minutes):
    # PSEUDOCODE: validate inputs -> hole minutes -> return the fixed semantic contract.
    intervals = []
    for item in filter(None, holes.split('|')):
        start, end = map(int, item.split('-'))
        if not 0 <= start < end <= session_minutes + 1e-6:
            raise ValueError('Sleepmeter interruption outside session.')
        intervals.append((start, end))
    total = 0
    previous_end = 0
    for start, end in sorted(intervals):
        total += end - max(start, previous_end) if end > previous_end else 0
        previous_end = max(end, previous_end)
    return total


def _clock(instant):
    # PSEUDOCODE: validate inputs -> clock -> return the fixed semantic contract.
    return (instant.hour * 3600 + instant.minute * 60 + instant.second) / 3600


def build_records(path, expected_md5=PUBLISHED_MD5):
    """Create one retrospective row per wake date, without clinical labels.

    The longest NIGHT_SLEEP is the daily anchor. Naps must have ended before
    that anchor's sleep onset, so the row never borrows a later event.
    """
    # PSEUDOCODE: validate inputs -> build records -> return the fixed semantic contract.
    path = Path(path)
    payload = path.read_bytes()
    if expected_md5 and md5(payload).hexdigest() != expected_md5:
        raise ValueError('Sleep diary does not match the published Figshare MD5.')
    digest = sha256(payload).hexdigest()
    sessions = list(_read_sessions(path))
    night_by_date = defaultdict(list)
    naps = []
    for row in sessions:
        if row['type'] == 'NIGHT_SLEEP':
            night_by_date[row['wake'].date().isoformat()].append(row)
        elif row['type'] == 'NAP':
            naps.append(row)
    records = []
    for day, candidates in sorted(night_by_date.items()):
        chosen = max(candidates, key=lambda row: (
            (row['wake'] - row['sleep']).total_seconds(), row['wake'], -row['_line']))
        awake = (chosen['wake'] - chosen['sleep']).total_seconds() / 60
        in_bed = (chosen['wake'] - chosen['bedtime']).total_seconds() / 60
        holes = _hole_minutes(chosen['holes'], awake)
        previous_naps = [row for row in naps
                         if chosen['wake'] - timedelta(days=1) <= row['sleep']
                         and row['wake'] <= chosen['sleep']]
        nap_minutes = sum((row['wake'] - row['sleep']).total_seconds() / 60
                          - _hole_minutes(row['holes'],
                                          (row['wake'] - row['sleep']).total_seconds() / 60)
                          for row in previous_naps)
        warnings = ['Retrospective diary row; later annotations may have been edited.']
        if len(candidates) > 1:
            warnings.append('Multiple NIGHT_SLEEP entries on this wake date; longest selected.')
        if awake == 0:
            warnings.append('Diary recorded a zero-duration NIGHT_SLEEP; sleep onset is unknown.')
        if in_bed > 1440 or awake > 1440:
            warnings.append('Sleep interval exceeds 24 hours; duration fields left missing.')
        source = {'source_url': SOURCE_URL, 'source_file': path.name,
                  'source_sha256': digest, 'source_line': chosen['_line'],
                  'wake_date': day, 'candidate_night_lines': [r['_line'] for r in candidates],
                  'source_quality_0_to_9': int(chosen['quality']),
                  'source_holes': chosen['holes'],
                  'note_present': bool(chosen['notes'].strip()),
                  'phenotype': 'author self-reports non-24 circadian rhythm disorder; not independently verified'}
        raw = {'record_id': f'SleepDiary-{day}', 'participant_id': PARTICIPANT,
               'observed_at': chosen['wake'].isoformat(), 'source_dataset': 'Sleepmeter-Figshare',
               'split': 'train', 'warnings': warnings, 'provenance': {'source': source}}
        if 0 <= awake <= 1440:
            raw['sleep_duration_h'] = round((awake - holes) / 60, 6)
            raw['sleep_end_hour'] = _clock(chosen['wake'])
            raw['provenance']['sleep_duration_h'] = {
                'source_line': chosen['_line'], 'calculation': '(wake - sleep - union(holes)) / 60'}
            raw['provenance']['sleep_end_hour'] = {'source_line': chosen['_line'], 'source_key': 'wake'}
            if awake > 0:
                raw['sleep_start_hour'] = _clock(chosen['sleep'])
                raw['provenance']['sleep_start_hour'] = {'source_line': chosen['_line'], 'source_key': 'sleep'}
        if 0 <= in_bed <= 1440:
            raw['time_in_bed_h'] = round(in_bed / 60, 6)
            raw['provenance']['time_in_bed_h'] = {
                'source_line': chosen['_line'], 'calculation': '(wake - bedtime) / 60'}
        raw['nap_count'] = len(previous_naps)
        raw['nap_duration_min'] = round(nap_minutes, 6)
        for key in ('nap_count', 'nap_duration_min'):
            raw['provenance'][key] = {'source_lines': [r['_line'] for r in previous_naps],
                                      'window': '24h before anchor wake, ending before anchor sleep'}
        records.append(validate_record(raw))
    return records
