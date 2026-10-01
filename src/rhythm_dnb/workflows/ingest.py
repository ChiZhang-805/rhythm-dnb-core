"""Explicit import orchestration; raw evidence remains immutable."""

from ..io.sources.hospital import read_observations
from ..io.repository import RhythmRepository


def ingest_table(source, destination, column_map, *, timezone, source_id):
    # PSEUDOCODE: validate a full acquisition batch before creating its separate destination store.
    observations = read_observations(source, column_map, timezone=timezone, source_id=source_id)
    repository = RhythmRepository.create(destination)
    repository.append_observations(observations)
    return {'observations': len(observations), 'store': str(repository.path)}
