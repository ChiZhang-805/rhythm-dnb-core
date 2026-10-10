"""Measure the full public NetHealth expansion capacity without creating outcome labels."""

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rhythm_dnb.provenance import file_hash
from rhythm_dnb.research.experiment_data import save_json


KEYS = ['egoid', 'datadate']


def windows(dates, history_days, future_days):
    # PSEUDOCODE: break at missing or ambiguous days -> count complete past/current/future windows.
    if any(type(n) is not int or n < 0 for n in (history_days, future_days)):
        raise ValueError('Window lengths must be nonnegative integers.')
    counts = Counter(dates)
    days = sorted(day for day, count in counts.items() if count == 1)
    lengths, streak, previous = [], 0, None
    for day in days:
        if previous is not None and (day - previous).days != 1:
            lengths.append(streak)
            streak = 0
        streak += 1
        previous = day
    if streak:
        lengths.append(streak)
    return sum(max(0, length - history_days - future_days) for length in lengths)


def pair_days(activity, sleep):
    # PSEUDOCODE: retain originals -> collapse exact copies -> exclude conflicting activity and tied longest sleeps.
    activity_unique, sleep_unique = activity.drop_duplicates(), sleep.drop_duplicates()
    conflicts = activity_unique.duplicated(KEYS, keep=False)
    longest = sleep_unique[sleep_unique.bedtimedur.eq(
        sleep_unique.groupby(KEYS).bedtimedur.transform('max'))]
    ties = longest.duplicated(KEYS, keep=False)
    paired = activity_unique[~conflicts].merge(longest[~ties], on=KEYS, validate='one_to_one')
    receipt = {
        'activity_exact_duplicate_rows': len(activity) - len(activity_unique),
        'sleep_exact_duplicate_rows': len(sleep) - len(sleep_unique),
        'conflicting_activity_days_excluded': len(activity_unique[conflicts][KEYS].drop_duplicates()),
        'tied_longest_sleep_days_excluded': len(longest[ties][KEYS].drop_duplicates()),
        'unambiguous_paired_days': len(paired),
        'unambiguous_paired_people': int(paired.egoid.nunique()),
        'sleep_selection': 'Longest in-bed episode on each wake date; planning only, not a frozen main-sleep definition.',
    }
    return paired, receipt


def basic_checks(frame):
    # PSEUDOCODE: screen finite values, physical bounds and source equations; do not invent replacements.
    numeric = ['complypercent', 'meanrate', 'sdrate', 'steps', 'fairlyactiveminutes',
               'veryactiveminutes', 'bedtimedur', 'minstofallasleep', 'minsafterwakeup',
               'minsasleep', 'minsawake', 'Efficiency']
    finite = np.isfinite(frame[numeric]).all(axis=1)
    clocks = {key: pd.to_timedelta(frame[key], errors='coerce').dt.total_seconds()
              for key in ('timetobed', 'timeoutofbed')}
    clock_ok = np.logical_and.reduce([v.ge(0) & v.lt(86400) for v in clocks.values()])
    duration = (clocks['timeoutofbed'] - clocks['timetobed']) % 86400 / 60
    derived_sleep = frame.bedtimedur - frame.minstofallasleep - frame.minsafterwakeup - frame.minsawake
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = frame.minsasleep / (frame.minsasleep + frame.minsawake)
    minute_fields = ['minstofallasleep', 'minsafterwakeup', 'minsasleep', 'minsawake',
                     'fairlyactiveminutes', 'veryactiveminutes']
    bounds = (frame.complypercent.between(0, 100) & frame.meanrate.gt(0) & frame.sdrate.ge(0)
              & frame.steps.ge(0) & frame.bedtimedur.gt(0) & frame.bedtimedur.le(1440)
              & frame[minute_fields].ge(0).all(axis=1) & frame[minute_fields].le(1440).all(axis=1)
              & frame.Efficiency.between(0, 1))
    checks = {'finite_required_fields': finite, 'valid_clock_strings': clock_ok,
              'nonnegative_and_unit_bounds': bounds,
              'clock_duration_agrees_within_one_minute': (duration - frame.bedtimedur).abs().le(1),
              'sleep_duration_equation_within_one_minute': (derived_sleep - frame.minsasleep).abs().le(1),
              'efficiency_equation_agrees': np.isclose(frame.Efficiency, efficiency, atol=1e-6, rtol=0)}
    good = np.logical_and.reduce(list(checks.values()))
    return good, {name: int((~check).sum()) for name, check in checks.items()}


def run(input_dir, output):
    # PSEUDOCODE: verify downloaded bytes -> census all days -> measure a declared coverage grid -> save no accuracy.
    input_dir, output = Path(input_dir), Path(output)
    if output.exists():
        raise FileExistsError('Keep previous audit; choose a fresh output directory.')
    download = json.loads((input_dir / 'downloads.json').read_text(encoding='utf-8'))
    for entry in download['files']:
        if Path(entry['file']).name != entry['file'] or file_hash(input_dir / entry['file']) != entry['sha256']:
            raise ValueError('Downloaded source identity mismatch.')
    activity = pd.read_csv(input_dir / 'activity.csv', low_memory=False)
    sleep = pd.read_csv(input_dir / 'sleep.csv', low_memory=False).rename(columns={'dataDate': 'datadate'})
    for table in (activity, sleep):
        if table[KEYS].isna().any().any():
            raise ValueError('Missing person or date; cannot count longitudinal windows.')
        table['datadate'] = table.datadate.map(date.fromisoformat)
    matched = activity[KEYS].drop_duplicates().merge(sleep[KEYS].drop_duplicates(), on=KEYS)
    frame, pairing = pair_days(activity, sleep)
    good, failures = basic_checks(frame)
    grid, participants = [], []
    # A sensitivity grid for capacity, never a choice based on predictive accuracy.
    for wear in (0, 50, 80, 90, 95):
        selected = frame[good & frame.complypercent.ge(wear)]
        for history in (0, 14, 28, 56):
            person_counts = []
            for person, group in selected.groupby('egoid'):
                count = windows(group.datadate.tolist(), history, 9)
                person_counts.append(count)
                participants.append({'participant_id': str(person), 'wear_percent_minimum': wear,
                    'history_days': history, 'future_days': 9, 'valid_daily_rows': len(group),
                    'complete_windows_without_outcome_labels': count})
            grid.append({'wear_percent_minimum': wear, 'history_days': history,
                'future_days': 9, 'valid_daily_rows': len(selected), 'people': int(selected.egoid.nunique()),
                'people_with_complete_windows': sum(n > 0 for n in person_counts),
                'complete_windows_without_outcome_labels': sum(person_counts)})
    result = {'source': download['official_source'], 'downloaded_files': download['files'],
        'audit_source_sha256': file_hash(__file__),
        'activity_rows': len(activity), 'activity_people': int(activity.egoid.nunique()),
        'sleep_rows': len(sleep), 'sleep_people': int(sleep.egoid.nunique()),
        'matched_person_days_before_quality_checks': len(matched),
        'matched_people': int(matched.egoid.nunique()), 'pairing': pairing,
        'date_range': [str(frame.datadate.min()), str(frame.datadate.max())],
        'basic_check_failures_may_overlap': failures, 'basic_check_passed_days': int(good.sum()),
        'capacity_grid': grid, 'rhythm_event_labels_created': False, 'warning_accuracy': None,
        'qualification': 'Counts describe source records and potentially evaluable date windows, not independent '
            'trials or confirmed clinical events. All rows remain in the saved originals. There are no Chinese '
            'diaries, meal-time panel or adjudicated rhythm-onset labels in these two tables. Main sleep '
            'selection, wear coverage, timing availability, outcome definition and people splits require a '
            'new frozen protocol before model evaluation. Capacity sensitivity must not choose a policy '
            'based on warning accuracy. Minute tolerances reflect rounded source fields; days with clock '
            'disagreements are retained for review, not automatically corrected.'}
    output.mkdir(parents=True)
    pd.DataFrame(grid).to_csv(output / 'capacity.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(participants).to_csv(output / 'participant-capacity.csv', index=False, encoding='utf-8-sig')
    save_json(output / 'result.json', result)
    save_json(output / 'manifest.json', {'files': {p.name: file_hash(p) for p in output.iterdir()}})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), ensure_ascii=False, indent=2))
