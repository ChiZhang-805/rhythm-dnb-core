"""Interpret heterogeneous historical field receipts without inventing real-time eligibility."""

CONSTRUCTED_METHODS = frozenset(('completed_input', 'manual_contextual_rewrite',
                                 'constructed_consistency_repair', 'dataset_coherence_repair'))


def legacy_field_kind(provenance, key):
    # PSEUDOCODE: exclude constructed methods first -> resolve field evidence through its source manifest.
    evidence = provenance.get(key, {})
    source = provenance.get('source', {})
    if not isinstance(evidence, dict) or not isinstance(source, dict):
        return 'unknown'
    method = evidence.get('method', '')
    if method == 'synthetic_scenario':
        return 'synthetic'
    if method in CONSTRUCTED_METHODS:
        return 'constructed'
    field_key = bool(evidence.get('source_key') or evidence.get('source_keys'))
    field_file = bool(evidence.get('source_file') or evidence.get('file'))
    if field_key and field_file:
        return 'source_traceable_retrospective'
    if source.get('source_file') and source.get('source_sha256') and (field_key or evidence.get('calculation') and evidence.get('source_line')):
        return 'source_traceable_retrospective'
    if evidence.get('source_sheet') and evidence.get('source_row') and field_key and source.get('supplements_sha256'):
        return 'source_traceable_retrospective'
    return 'unknown'
