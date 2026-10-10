"""Inventory every existing record for expansion without inventing outcomes or altering frozen experiments."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
import csv
import json
from pathlib import Path
import sqlite3

from rhythm_dnb.io.sources.lineage import legacy_field_kind
from rhythm_dnb.provenance import file_hash
from rhythm_dnb.research.experiment_data import save_json


def calendar_capacity(dates, future_days=9):
    # PSEUDOCODE: exclude ambiguous duplicate days -> count consecutive date windows with complete future coverage.
    counts = Counter(dates)
    ordered = sorted(day for day, count in counts.items() if count == 1)
    lengths, streak, previous = [], 0, None
    for day in ordered:
        if previous is not None and (day - previous).days != 1:
            lengths.append(streak); streak = 0
        streak += 1; previous = day
    if streak:
        lengths.append(streak)
    return {'unique_days': len(counts), 'duplicate_days': sum(n > 1 for n in counts.values()),
            'longest_streak': max(lengths, default=0),
            'calendar_window_upper_bound': sum(max(0, n - future_days) for n in lengths)}


def audit(database, text_development, output):
    # PSEUDOCODE: audit a read-only snapshot -> preserve all row dispositions -> report size separately from label readiness.
    database, output = Path(database).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous inventory; choose a new output directory.')
    before = file_hash(database)
    text_hash = file_hash(text_development)
    text = json.loads(Path(text_development).read_text(encoding='utf-8'))
    exposed = {r['participant_id'][7:] for r in text['rows']
               if (r.get('participant_id') or '').startswith('rhythm:')}
    exposure_sources = {r['source_record_id'][7:] for r in text['rows']
                        if (r.get('source_record_id') or '').startswith('rhythm:')}
    by_source, by_person, persons, records, fields, texts = {}, defaultdict(list), set(), [], [], set()
    objective = ['sleep_start_hour', 'sleep_end_hour', 'sleep_duration_h', 'meal_interval_cv',
                 'exercise_minutes', 'resting_hr_bpm', 'screen_time_min']
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        exposed.update(r['participant_id'] for r in db.execute(
            'SELECT record_id, participant_id FROM observations') if r['record_id'] in exposure_sources)
        numeric = [r['name'] for r in db.execute('PRAGMA table_info(observations)') if r['type'] == 'REAL']
        query = ('SELECT o.*, p.payload AS provenance_payload, r.payload AS raw_payload FROM observations o '
                 'JOIN observation_provenance p USING(record_id) JOIN raw_inputs r USING(record_id) ORDER BY o.record_id')
        for saved in db.execute(query):
            row = dict(saved); source = row['source_dataset']; person = row['participant_id']
            if source not in by_source:
                by_source[source] = {'counts': Counter(), 'people': set(), 'traceable': Counter(), 'exposed': set()}
            item = by_source[source]; c = item['counts']; c['records'] += 1
            item['people'].add(person); persons.add(person)
            by_person[source, person].append(datetime.fromisoformat(row['observed_at']).date())
            provenance = json.loads(row['provenance_payload']).get('provenance', {})
            raw = json.loads(row['raw_payload'])
            supported = {k for k in numeric if row[k] is not None and
                         legacy_field_kind(provenance, k) == 'source_traceable_retrospective'}
            item['traceable'].update(supported)
            c['rows_with_source_traced_numeric'] += bool(supported)
            c['full_original_objective_panel_source_traced'] += all(k in supported for k in objective)
            c['source_traced_sleep_clock_duration'] += all(k in supported for k in
                ('sleep_start_hour', 'sleep_end_hour', 'sleep_duration_h'))
            c['source_traced_sleep_heart_panel'] += all(k in supported for k in
                ('sleep_duration_h', 'resting_hr_bpm', 'heart_rate_bpm', 'hr_sd_bpm'))
            c['current_state_labeled_rows'] += row['rhythm_state_gt'] is not None
            c['event_onset_rows'] += bool(row['event_onset_at'])
            c['followup_rows'] += bool(row['followup_end_at'])
            c['outcome_observed_at_rows'] += bool(row['label_observed_at'])
            if person in exposed or row['record_id'] in exposure_sources:
                item['exposed'].add(person)
            for category in ('emotion', 'diet', 'sleep', 'social'):
                value = (raw.get(category + '_description_text') or '').strip()
                if value:
                    c['text_segments_before_deduplication'] += 1; texts.add((category, value))
            simulation = source == 'SIMULATED-RHYTHM'
            records.append({'record_id': row['record_id'], 'participant_id': person, 'source': source,
                'observed_at': row['observed_at'], 'existing_split': row['split'],
                'source_traceable_numeric_fields': len(supported),
                'endpoint_status': 'simulated_outcomes_only' if simulation else 'no_usable_future_event_labels',
                'text_model_development_exposure': person in exposed or row['record_id'] in exposure_sources})
        if len(records) != db.execute('SELECT count(*) FROM observations').fetchone()[0]:
            raise ValueError('Missing or duplicate raw/provenance joins.')
    if file_hash(database) != before or file_hash(text_development) != text_hash:
        raise ValueError('Source changed during audit; no inventory published.')
    summaries, temporal = [], []
    for (source, person), dates in sorted(by_person.items()):
        temporal.append({'source': source, 'participant_id': person, **calendar_capacity(dates)})
    for source, item in sorted(by_source.items()):
        t = [r for r in temporal if r['source'] == source]
        summaries.append({'source': source, 'participants': len(item['people']), **dict(item['counts']),
            'people_with_14_consecutive_stored_dates': sum(r['longest_streak'] >= 14 for r in t),
            'people_with_30_consecutive_stored_dates': sum(r['longest_streak'] >= 30 for r in t),
            'calendar_window_upper_bound_NOT_labeled_tests': sum(r['calendar_window_upper_bound'] for r in t),
            'text_model_development_exposed_people': len(item['exposed']),
            'independent_observed_warning_test_ready': False})
        fields.extend({'source': source, 'field': k, 'source_traceable_rows': item['traceable'][k]}
                      for k in numeric)
    output.mkdir(parents=True)
    for name, data in [('sources.csv', summaries), ('participants.csv', temporal),
                       ('records.csv', records), ('field-coverage.csv', fields)]:
        with (output / name).open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    result = {'database_sha256': before, 'database_modified': False, 'records': len(records),
              'text_development_sha256': text_hash, 'audit_source_sha256': file_hash(__file__),
              'participant_ids': len(persons), 'source_groups': len(summaries),
              'distinct_category_text_pairs': len(texts), 'sources': summaries,
              'qualification': 'Stored source evidence and calendar dates only. Original files, arrival times, '
                  'clinical outcomes and independent identities are not established by these counts. '
                  'Calendar capacity is not a number of labeled predictions. Constructed fields remain excluded '
                  'from source-traceable coverage. All original splits and values are unchanged.'}
    save_json(output / 'result.json', result)
    save_json(output / 'manifest.json', {'files': {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('database', 'text-development', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.database, args.text_development, args.output), ensure_ascii=False, indent=2))
