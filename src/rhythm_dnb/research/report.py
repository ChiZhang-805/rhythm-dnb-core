"""Serializable evidence reports and read-only audits of the legacy data store."""

from collections import Counter
import json
from pathlib import Path
import sqlite3
from ..provenance import canonical_json, fingerprint
from ..measures.panel import JOINT12, OBJECTIVE8
from ..io.sources.lineage import legacy_field_kind
from ..io.sources.legacy_schema import FEATURE_KEYS, _feature_value


def audit_legacy_store(path):
    # PSEUDOCODE: open a stable read transaction -> classify cell evidence -> count prospective readiness without mutation.
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute('BEGIN')
        source = Counter(); origins = Counter(); supported = Counter(); methods = Counter()
        numeric_fields = {r[1] for r in db.execute('PRAGMA table_info(observations)') if r[2] == 'REAL'}
        panel_coverage = {name: {'present_columns': sorted(set(fields) & numeric_fields),
            'missing_columns': sorted(set(fields) - numeric_fields),
            'status': 'stored_columns_available' if set(fields) <= numeric_fields else 'requires_source_derivation'}
            for name, fields in (('objective8', OBJECTIVE8), ('joint12', JOINT12))}
        by_source = {}; source_traces = {}; distributions = {}
        people = set(); complete12 = complete8 = onset_count = negative_followup = 0
        digests, invalid_values = [], []
        missing_values = Counter()
        query = '''SELECT o.*, p.payload AS provenance_payload FROM observations o
                   LEFT JOIN observation_provenance p ON p.record_id=o.record_id ORDER BY o.record_id'''
        for row in db.execute(query):
            source[row['source_dataset']] += 1; people.add(row['participant_id'])
            provenance = json.loads(row['provenance_payload'] or '{}').get('provenance', {})
            source_name = row['source_dataset']
            by_source.setdefault(source_name, Counter()); source_traces.setdefault(source_name, Counter())
            eligible = {}
            for key, evidence in provenance.items():
                if key not in numeric_fields or not isinstance(evidence, dict):
                    continue
                method = evidence.get('method', '')
                methods[method or 'source_evidence_or_unknown'] += 1
                kind = legacy_field_kind(provenance, key)
                origins[kind] += 1
                present = key in row.keys() and row[key] is not None
                eligible[key] = present and kind == 'source_traceable_retrospective'
                if eligible[key]:
                    supported[key] += 1
                    source_traces[source_name][key] += 1
                    distributions.setdefault(key, {}).setdefault(source_name, []).append(row[key])
            for key in numeric_fields:
                if row[key] is not None:
                    by_source[source_name][key] += 1
                else:
                    missing_values[key] += 1
                if key in FEATURE_KEYS:
                    try:
                        _feature_value(row[key], key)
                    except ValueError as error:
                        invalid_values.append({'record_id': row['record_id'], 'field': key, 'error': str(error)})
            complete12 += all(eligible.get(k, False) for k in JOINT12)
            complete8 += all(eligible.get(k, False) for k in OBJECTIVE8)
            onset_count += bool(row['event_onset_at'])
            negative_followup += row['rhythm_state_gt'] == 0 and bool(row['followup_end_at'])
            digests.append((row['record_id'], row['row_sha256'], fingerprint(provenance)))
        result = {'rows': sum(source.values()), 'participants': len(people), 'sources': dict(source),
                  'cell_evidence': dict(origins), 'methods': dict(methods), 'source_traceable_fields': dict(supported),
                  'numeric_fields': sorted(numeric_fields),
                  'invalid_numeric_values': invalid_values, 'missing_numeric_values': dict(missing_values),
                  'present_by_source': {s: dict(v) for s, v in by_source.items()},
                  'traceable_by_source': {s: dict(v) for s, v in source_traces.items()},
                  'traceable_distributions': distributions,
                  'traceability_level': 'stored file/key evidence only; original bytes not reverified by this audit',
                  'stored_panel_coverage': panel_coverage,
                  'complete_source_traceable_joint12_rows': complete12 if not panel_coverage['joint12']['missing_columns'] else None,
                  'complete_source_traceable_objective8_rows': complete8 if not panel_coverage['objective8']['missing_columns'] else None,
                  'rows_with_onset_field': onset_count, 'negative_rows_with_followup_field': negative_followup,
                  'snapshot_content_id': fingerprint(digests), 'database_modified': False,
                  'prospective_validation_ready': False,
                  'qualification': 'Numeric cells only; metadata excluded. Panel row counts are null when required stored columns are absent: raw sources must first be transformed and validated. Matching column names alone do not establish compatible units or measurement definitions. Legacy date buckets lack verified arrival times and endpoint confirmation; retrospective traceability is not prospective eligibility.'}
        return result
    finally:
        db.close()


def save_report(report, path):
    # PSEUDOCODE: save strict JSON to a fresh artifact path, preserving previous research evidence.
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(canonical_json(report) + '\n')
    return path
