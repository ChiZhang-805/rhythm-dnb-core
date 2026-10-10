"""Read the hospital monitoring XLS by field names; preserve unresolved source cells."""

import calendar
import math
import re
from collections import Counter, defaultdict

from ...provenance import file_hash


META = ('date', 'number', 'age', 'gender')
CLOCKS = ('sleep_start_time', 'sleep_end_time', 'breakfast_time', 'lunch_time',
          'dinner_time', 'caffeine_last_time', 'exercise_time_of_day')
FIELDS = META + ('sleep_duration_h', 'time_in_bed', 'nap_county', 'nap_duration_min',
    'late_night_eating_flag', 'breakfast_kcal', 'lunch_kcal', 'dinner_kcal', 'water_ml',
    'stool_count', 'smoking_state', 'caffeine_mg', 'exercise_minutes', 'exercise_type',
    'heart_rate_bpm', 'resting_hr_bpm', '_max_bpm', 'skin_temp_c', 'spO2_pct',
    'spO2_min_pct', 'press_score', 'general_mood', 'energy_score', 'self_rate_state',
    'fatigue_score', 'social_interaction_min', 'social_contact_count', 'ambient_temp_c',
    'noise_db', 'screen_time_min') + CLOCKS
OPTIONAL = ('meal_interval_CV', '_sd_bpm')
CONDITIONAL = {'breakfast_time': 'breakfast_kcal', 'lunch_time': 'lunch_kcal',
               'dinner_time': 'dinner_kcal', 'caffeine_last_time': 'caffeine_mg',
               'exercise_time_of_day': 'exercise_minutes', 'exercise_type': 'exercise_minutes'}
INTEGER = {'nap_county', 'stool_count', 'social_contact_count', 'late_night_eating_flag', 'smoking_state'}
BOUNDS = {**{f: (0, 10) for f in ('press_score', 'general_mood', 'energy_score', 'fatigue_score')},
          'self_rate_state': (1, 5), 'smoking_state': (0, 5), 'late_night_eating_flag': (0, 1),
          'spO2_pct': (0, 100), 'spO2_min_pct': (0, 100),
          'sleep_duration_h': (0, 24), 'time_in_bed': (0, 24),
          **{f: (0, 1440) for f in ('nap_duration_min', 'exercise_minutes', 'screen_time_min', 'social_interaction_min')}}


def absent(value):
    # PSEUDOCODE: distinguish explicit absence markers from numeric zero.
    return value is None or (isinstance(value, str) and value.strip() in ('', '—', '–', '-'))


def clock_minutes(value):
    """Return minutes since midnight, rejecting Excel day components instead of wrapping."""
    # PSEUDOCODE: parse a valid fractional Excel day or clock string without inventing a date.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) * 1440 if math.isfinite(value) and 0 <= value < 1 else None
    match = re.fullmatch(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', str(value).strip())
    if match:
        h, m, s = (int(v or 0) for v in match.groups())
        if h < 24 and m < 60 and s < 60:
            return h * 60 + m + s / 60
    return None


def partial_date(value, cell_type, datemode=0):
    """Keep month/day ambiguity; yearless dates never become full observed_at timestamps."""
    # PSEUDOCODE: keep real Excel dates, flag clocks in date cells, enumerate lost-trailing-zero alternatives.
    import xlrd
    if cell_type == xlrd.XL_CELL_DATE:
        if not isinstance(value, (int, float)) or value < 1:
            minutes = clock_minutes(value)
            if minutes is not None:
                hour, minute = divmod(round(minutes), 60)
                if 1 <= hour <= 12 and 1 <= minute <= calendar.monthrange(2000, hour)[1]:
                    return {'calendar_date': None, 'month_day': None,
                            'candidates': [f'{hour:02d}-{minute:02d}'], 'status': 'clock_in_date_cell'}
            return {'calendar_date': None, 'month_day': None, 'candidates': [], 'status': 'invalid_date'}
        try:
            date = xlrd.xldate_as_datetime(value, datemode)
        except (ValueError, OverflowError):
            return {'calendar_date': None, 'month_day': None, 'candidates': [], 'status': 'invalid_date'}
        return {'calendar_date': date.date().isoformat(), 'month_day': date.strftime('%m-%d'),
                'candidates': [], 'status': 'calendar_date_only'}
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    match = re.fullmatch(r'(\d{1,2})[./-](\d{1,2})', str(value).strip())
    candidates = []
    if match:
        month, raw_day = int(match[1]), match[2]
        days = [int(raw_day)]
        if numeric and len(raw_day) == 1:
            days.append(int(raw_day) * 10)
        if 1 <= month <= 12:
            candidates = sorted({f'{month:02d}-{d:02d}' for d in days
                                 if 1 <= d <= calendar.monthrange(2000, month)[1]})
    status = 'year_missing' if len(candidates) == 1 else 'ambiguous_day' if candidates else 'invalid_date'
    return {'calendar_date': None, 'month_day': candidates[0] if len(candidates) == 1 else None,
            'candidates': candidates if len(candidates) > 1 else [], 'status': status}


def normalized_number(field, value):
    # PSEUDOCODE: enforce declared scale and physical duration bounds, not fitted outlier limits.
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return None
    low, high = BOUNDS.get(field, (-math.inf if field in ('skin_temp_c', 'ambient_temp_c', 'noise_db') else 0, math.inf))
    if not low <= value <= high or (field in INTEGER and value != int(value)):
        return None
    return float(value)


def _address(row, column):
    # PSEUDOCODE: convert zero-based indices to Excel cell addresses for source review.
    letters, column = '', column + 1
    while column:
        column, rem = divmod(column - 1, 26)
        letters = chr(65 + rem) + letters
    return f'{letters}{row + 1}'


def read_monitoring_workbook(path, review_cells=()):
    """Return auditable numeric inputs, raw cells and issues, never clinical outcomes."""
    # PSEUDOCODE: map each sheet independently -> validate each cell -> retain raw lineage -> count usable inputs.
    import xlrd
    workbook = xlrd.open_workbook(str(path), formatting_info=True)
    rows, issues, definitions, counts = [], [], {}, Counter()
    review = {(r['sheet'], r['cell']): r for r in review_cells}
    if len(review) != len(review_cells):
        raise ValueError('Duplicate review-cell instructions.')
    seen_review = set()
    person_sheets = {}

    def issue(row, field, code):
        # PSEUDOCODE: attach each review item to its original cell without rewriting the source.
        cell = row['raw'][field]
        issues.append({'record_id': row['record_id'], 'participant_id': row['participant_id'],
                       'sheet': row['sheet'], 'cell': cell['cell'], 'field': field,
                       'code': code, 'raw_value': cell['value']})

    try:
        for sheet in workbook.sheets():
            header = [str(sheet.cell_value(0, c)).strip() for c in range(sheet.ncols)]
            populated = [h for h in header if h]
            if len(populated) != len(set(populated)) or not set(FIELDS) <= set(header):
                raise ValueError(f'{sheet.name}: missing or duplicate field names.')
            if set(populated) - set(FIELDS) - set(OPTIONAL):
                raise ValueError(f'{sheet.name}: unsupported fields; review schema before import.')
            mapping = {h: c for c, h in enumerate(header) if h}
            for field, c in mapping.items():
                label, unit = str(sheet.cell_value(1, c)).strip(), str(sheet.cell_value(2, c)).strip()
                spec = {'label': label, 'source_unit': unit,
                        'normalized_unit': 'minutes_since_midnight' if field in CLOCKS else unit,
                        'role': 'metadata' if field in META else 'clock' if field in CLOCKS else 'category' if field == 'exercise_type' else 'numeric'}
                if field in definitions and definitions[field]['source_unit'] != unit:
                    equivalent = field == 'self_rate_state' and {unit, definitions[field]['source_unit']} <= {'1-5星', '1为差，5为好'}
                    if not equivalent:
                        raise ValueError(f'{sheet.name}: inconsistent units for {field}.')
                definitions.setdefault(field, spec)
                definitions[field].setdefault('source_unit_variants', {}).setdefault(unit, []).append(sheet.name)
            for r in range(3, sheet.nrows):
                if all(absent(sheet.cell_value(r, c)) for c in mapping.values()):
                    continue
                identity = sheet.cell_value(r, mapping['number'])
                if absent(identity) or isinstance(identity, bool):
                    raise ValueError(f'{sheet.name}:{r + 1}: missing participant identity.')
                identity = str(int(identity)) if isinstance(identity, (int, float)) and identity == int(identity) else str(identity).strip()
                pid = 'hospital-' + identity
                if pid in person_sheets and person_sheets[pid] != sheet.name:
                    raise ValueError('Participant appears on multiple sheets; recording order needs review.')
                person_sheets[pid] = sheet.name
                counts[pid] += 1
                row = {'record_id': f'{pid}-{counts[pid]:02d}', 'participant_id': pid,
                       'occasion': counts[pid], 'sheet': sheet.name, 'observed_at': None,
                       'values': {}, 'states': {}, 'raw': {}}
                for field, c in mapping.items():
                    cell = sheet.cell(r, c)
                    row['raw'][field] = {'cell': _address(r, c), 'value': cell.value, 'type': cell.ctype}
                date_cell = sheet.cell(r, mapping['date'])
                row['date'] = partial_date(date_cell.value, date_cell.ctype, workbook.datemode)
                if row['date']['status'] not in ('year_missing', 'calendar_date_only'):
                    issue(row, 'date', row['date']['status'])
                for field in mapping:
                    if field in ('date', 'number'):
                        continue
                    value = row['raw'][field]['value']
                    state = 'missing' if absent(value) else 'observed'
                    if state == 'missing':
                        normalized = None
                    elif field in ('gender', 'exercise_type'):
                        normalized = str(value).strip()
                    elif field in CLOCKS:
                        normalized = clock_minutes(value) if row['raw'][field]['type'] in (xlrd.XL_CELL_TEXT, xlrd.XL_CELL_NUMBER, xlrd.XL_CELL_DATE) else None
                    else:
                        normalized = normalized_number(field, value) if row['raw'][field]['type'] == xlrd.XL_CELL_NUMBER else None
                    if normalized is None and state == 'observed':
                        state = 'invalid_value'
                    key = (sheet.name, row['raw'][field]['cell'])
                    if key in review:
                        if review[key]['field'] != field:
                            raise ValueError('Review-cell field mapping differs from workbook.')
                        seen_review.add(key)
                        state, normalized = 'requires_confirmation', None
                    row['values'][field], row['states'][field] = normalized, state
                    if state in ('invalid_value', 'requires_confirmation'):
                        issue(row, field, state)
                for field, parent in CONDITIONAL.items():
                    if row['states'][field] == 'missing' and row['values'][parent] == 0:
                        row['states'][field] = 'not_applicable'
                for field in mapping:
                    if field not in META and field not in OPTIONAL and row['states'][field] == 'missing':
                        issue(row, field, 'missing_not_zero')
                rows.append(row)
    finally:
        workbook.release_resources()
    if seen_review != set(review):
        raise ValueError('A declared review cell was not encountered; do not silently reuse a stale protocol.')
    if not rows:
        raise ValueError('No observations in workbook.')
    by_person = defaultdict(list)
    for row in rows:
        by_person[row['participant_id']].append(row)
    for person_rows in by_person.values():
        for field in ('age', 'gender'):
            if len({row['values'][field] for row in person_rows}) != 1 or person_rows[0]['values'][field] is None:
                raise ValueError(f'Inconsistent participant metadata: {person_rows[0]["participant_id"]} / {field}.')
    inventory = {field: dict(Counter(row['states'].get(field, 'not_collected') for row in rows))
                 for field in definitions if field not in ('date', 'number')}
    return {'source_sha256': file_hash(path), 'rows': rows, 'issues': issues,
            'definitions': definitions, 'inventory': inventory,
            'participant_count': len(counts), 'record_count': len(rows),
            'records_per_person': dict(Counter(counts.values())),
            'date_status_counts': dict(Counter(row['date']['status'] for row in rows))}
