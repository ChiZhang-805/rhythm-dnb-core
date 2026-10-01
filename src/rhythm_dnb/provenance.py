"""Content hashes and measurement lineage; plausible values do not prove origin."""

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path
from .contracts import Provenance


def json_default(value):
    # PSEUDOCODE: serialize supported scientific containers without accepting arbitrary objects.
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (date, datetime, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def canonical_json(value) -> str:
    # PSEUDOCODE: use sorted strict JSON so hashes do not depend on mapping insertion order.
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False, default=json_default)


def fingerprint(value) -> str:
    # PSEUDOCODE: hash the canonical UTF-8 representation.
    return sha256(canonical_json(value).encode('utf-8')).hexdigest()


def file_hash(path) -> str:
    # PSEUDOCODE: stream bytes so large checkpoints do not need a second in-memory copy.
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def eligible_measurement(provenance: Provenance, *, simulation: bool = False) -> bool:
    # PSEUDOCODE: require traceable real lineage or explicitly selected simulation lineage.
    if not provenance.source_id or not provenance.source_hash:
        return False
    if simulation:
        return provenance.kind == 'synthetic'
    return provenance.kind in ('observed', 'derived') and provenance.independent


def build_lineage(parents, method: str) -> Provenance:
    # PSEUDOCODE: propagate the least trustworthy parent class; hash ordered parent evidence.
    parents = tuple(parents)
    if not parents:
        return Provenance('unknown', '', method=method, independent=False)
    rank = {'observed': 0, 'derived': 1, 'synthetic': 2, 'constructed': 3, 'unknown': 4}
    kind = max((p.kind for p in parents), key=rank.__getitem__)
    if any(p.kind == 'synthetic' for p in parents) and not all(p.kind == 'synthetic' for p in parents):
        kind = 'unknown'
    if kind == 'observed':
        kind = 'derived'
    return Provenance(kind, method, fingerprint(parents), tuple(p.source_id for p in parents),
                      method, all(p.independent and bool(p.source_hash) and bool(p.source_id) for p in parents))
