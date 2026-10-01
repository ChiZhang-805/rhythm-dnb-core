"""Legacy raw-source parser; explicit path, retrospective semantics retained.

Returned legacy records are NOT automatically prospective Observation objects.
Use the hospital/normalized adapter after supplying verified time and lineage.
"""
"""Import baseline and separately unlabeled treatment nights from DelSoM."""

from collections import defaultdict
from datetime import date, datetime, timedelta
from hashlib import sha256
import math
from pathlib import Path

from openpyxl import load_workbook

from .legacy_schema import validate_record


ARTICLE = 'https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002587'
FILES = {
    'pmed.1002587.s002.xlsx': '55d4e2ba68766d86ee32295677f53c0f11cac449db76d12ee08b557c6be887ee',
    'pmed.1002587.s005.xlsx': 'd338d4b746c94553a19814ee166f01cc30bda0a80ea543a0e3347d65b8feec38',
}
LABEL_SOURCE = 'DelSoM sleep physician ICSD-2 DSWPD diagnosis before baseline; randomized delayed-DLMO cohort (PLOS Medicine 2018)'




def _table(sheet):
    # PSEUDOCODE: pair worksheet headers with nonempty rows while retaining source line numbers.
    rows = sheet.values
    header = next(rows)
    for line, values in enumerate(rows, 2):
        if values[0] is not None:
            yield line, dict(zip(header, values))


def _date(value):
    # PSEUDOCODE: read a supplied calendar date -> return missing for invalid date fields.
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    return None


def _number(value, maximum):
    # PSEUDOCODE: reject boolean/nonfinite values -> retain numbers within the source field bounds.
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and 0 <= result <= maximum else None


def _pick_value(row, primary, fallback, maximum):
    # PSEUDOCODE: prefer the valid objective measurement -> otherwise use the documented diary fallback.
    first = _number(row.get(primary), maximum)
    second = _number(row.get(fallback), maximum)
    return (first, primary) if first is not None else (second, fallback) if second is not None else (None, None)


def _build_period(root, verify, period):
    """Only baseline has an independently established current-state diagnosis."""
    # PSEUDOCODE: verify source hashes -> join eligible nights -> separate baseline diagnosis from treatment observations.
    root = Path(root)
    for name, expected in FILES.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError('DelSoM supplement missing: ' + str(path))
        if verify and sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('DelSoM source SHA-256 differs: ' + name)
    demographics = load_workbook(root / 'pmed.1002587.s002.xlsx', read_only=True, data_only=True)
    nights = load_workbook(root / 'pmed.1002587.s005.xlsx', read_only=True, data_only=True)
    try:
        randomized = {row['ParticipantID'] for _, row in _table(demographics['ITT_demographics'])}
        selected = [(line, row) for line, row in _table(nights['Combined'])
                    if row['StudyPeriod'] == period and row['ParticipantID'] in randomized
                    and (period != 'Baseline' or row['Delayed/Not Delayed'] == 1)]
    finally:
        demographics.close()
        nights.close()
    if verify and period == 'Baseline' and {row['ParticipantID'] for _, row in selected} != randomized:
        raise ValueError('DelSoM randomized identifiers do not match baseline nights.')
    if verify and period == 'Treatment' and len({row['ParticipantID'] for _, row in selected}) != 106:
        raise ValueError('DelSoM treatment participants differ from published supplement.')
    by_date = defaultdict(list)
    missing_date = 0
    for line, row in selected:
        day = _date(row.get('Date_Onset_ACT')) or _date(row.get('Date_Onset_SD'))
        if day is None:
            missing_date += 1
            continue
        by_date[(row['ParticipantID'], day)].append((line, row))
    records = []
    for (participant, day), candidates in sorted(by_date.items()):
        line, row = max(candidates, key=lambda pair: (
            sum(pair[1].get(key) is not None for key in
                ('TST_ACT', 'TIB_ACT', 'ST_ACT', 'Risetime_ACT', 'TST_SD', 'TIB_SD')),
            -int(pair[1]['SleepEpisodeNo'])))
        warnings = (['Baseline physician-confirmed DSWPD; no pre-onset observation or healthy control.']
                    if period == 'Baseline' else
                    ['Treatment-period observation; current disorder status and remission are not independently assessed.',
                     'Randomized treatment and behavioral scheduling may change sleep; do not treat as untreated reference.'])
        warnings.append('Source dates lack timezone; next-day noon UTC is a nominal storage anchor, not a measured timestamp.')
        if len(candidates) > 1:
            warnings.append('Multiple baseline episodes share source date; most complete selected; all lines retained in provenance.')
        if missing_date:
            warnings.append(f'Some source {period.lower()} rows lacked both date fields and were excluded from this import.')
        provenance = {'source': {'article': ARTICLE, 'supplements_sha256': FILES,
                                 'sheet': 'Combined', 'selected_row': line,
                                 'same_date_rows': [n for n, _ in candidates],
                                 'source_night_date': day.isoformat(),
                                 'sleep_episode_no': row['SleepEpisodeNo'],
                                 'randomized_itt_id_crosscheck': True},
                      'observed_at': {'meaning': 'nominal next-day noon; source night date lacks timezone'},
                      'rhythm_state_gt': {'meaning': 'clinician-confirmed DSWPD at enrollment; not incident onset'}
                                         if period == 'Baseline' else
                                         {'meaning': 'unknown during randomized treatment; baseline diagnosis not carried forward'}}
        raw = {'record_id': (f'DelSoM-{participant}-{day.isoformat()}' if period == 'Baseline' else
                             f'DelSoM-Treatment-{participant}-{day.isoformat()}'),
               'participant_id': f'DelSoM-{participant}',
               'observed_at': (day + timedelta(days=1)).isoformat() + 'T12:00:00+00:00',
               'source_dataset': 'DelSoM-baseline' if period == 'Baseline' else 'DelSoM-treatment',
               'split': 'train',
               'warnings': warnings, 'provenance': provenance}
        if period == 'Baseline':
            raw['rhythm_state_gt'] = 1
            raw['label_source'] = LABEL_SOURCE
        for target, objective, diary in (
            ('sleep_duration_h', 'TST_ACT', 'TST_SD'),
            ('time_in_bed_h', 'TIB_ACT', 'TIB_SD'),
        ):
            value, key = _pick_value(row, objective, diary, 1440)
            if value is not None:
                raw[target] = value / 60
                provenance[target] = {'source_sheet': 'Combined', 'source_row': line,
                                      'source_key': key, 'conversion': 'minutes/60'}
        for target, objective, diary in (
            ('sleep_start_hour', 'ST_ACT', 'ST_SD'),
            ('sleep_end_hour', 'Risetime_ACT', 'WT_SD'),
        ):
            value, key = _pick_value(row, objective, diary, 48)
            if value is not None:
                raw[target] = value % 24
                provenance[target] = {'source_sheet': 'Combined', 'source_row': line,
                                      'source_key': key, 'conversion': 'source local decimal hour modulo 24'}
        if (raw.get('sleep_duration_h') is not None and raw.get('time_in_bed_h') is not None
                and raw['sleep_duration_h'] > raw['time_in_bed_h'] + 1e-6):
            raw.pop('time_in_bed_h')
            provenance.pop('time_in_bed_h')
            warnings.append('Source sleep duration exceeds in-bed duration; in-bed value omitted.')
        records.append(validate_record(raw))
    return records


def build_records(root, verify=True):
    """Import diagnosis-confirmed baseline nights only."""
    # PSEUDOCODE: import baseline nights with independently established enrollment diagnoses.
    return _build_period(root, verify, 'Baseline')


def build_treatment_records(root, verify=True):
    """Import treatment nights without carrying baseline diagnosis forward."""
    # PSEUDOCODE: import treatment nights while leaving current outcome status unknown.
    return _build_period(root, verify, 'Treatment')
