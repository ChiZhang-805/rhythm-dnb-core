"""Explicit normalized/wide-table adapter; no guessed dates, zones, labels or units."""

from ..tabular import read_table
from ..validation import validate_observations
from ...contracts import Observation, Provenance
from ...timebase import instant
from ...provenance import file_hash


def read_observations(path, column_map, *, timezone, source_id):
    """Map source columns to observation_id, participant_id, variable, value, unit,
    start, end, available_at. Source timestamps must include their actual offsets.
    """
    # PSEUDOCODE: require a complete declared mapping -> preserve row values -> validate explicit time semantics.
    required = {'observation_id', 'participant_id', 'variable', 'value', 'unit', 'start', 'end', 'available_at'}
    if set(column_map) != required or len(set(column_map.values())) != len(required):
        raise ValueError('A complete unambiguous acquisition column map is required.')
    rows, observations = read_table(path), []
    digest = file_hash(path)
    for i, row in enumerate(rows):
        if not set(column_map.values()) <= set(row):
            raise ValueError(f'Missing source columns in row {i + 1}.')
        values = {key: row[column] for key, column in column_map.items()}
        if any(values[key] is None or isinstance(values[key], bool) or not str(values[key]).strip() for key in ('observation_id', 'participant_id', 'variable', 'unit')):
            raise ValueError('Missing or ambiguous source identity.')
        value = values['value']
        if isinstance(value, bool):
            raise ValueError('Boolean measurements need explicit upstream encoding.')
        if value == '' or value is None:
            value = None
        elif values['unit'] != 'text':
            value = float(value)
        observations.append(Observation(str(values['observation_id']), str(values['participant_id']),
            values['variable'], value, values['unit'], instant(values['start']), instant(values['end']),
            instant(values['available_at']), timezone, Provenance('observed', source_id + f':row:{i + 1}', digest)))
    return validate_observations(observations)
