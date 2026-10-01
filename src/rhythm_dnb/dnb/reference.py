"""Freeze one randomly selected eligible baseline day per independent person."""

from dataclasses import dataclass
from datetime import datetime
import numpy as np
from ..contracts import DailyPanel
from ..provenance import fingerprint
from ..timebase import instant
from ..measures.scaling import fit_scaler, transform


@dataclass(frozen=True)
class ReferenceCandidate:
    panel: DailyPanel
    stable: bool
    stability_evidence_id: str
    stability_available_at: datetime
    baseline: bool = True


def fit_reference(candidates, features, cutoff, *, minimum=60, seed=20261001):
    # PSEUDOCODE: validate stable evidence/as-of data -> group eligible days -> draw one per person.
    from ..warning.windows import panel_vector
    cutoff = instant(cutoff)
    grouped = {}
    seen = set()
    for candidate in candidates:
        p = candidate.panel
        identity = (p.participant_id, p.day)
        if identity in seen:
            raise ValueError('Duplicate reference person-day.')
        seen.add(identity)
        if (candidate.stable is not True or candidate.baseline is not True or
                not candidate.stability_evidence_id or instant(candidate.stability_available_at) > cutoff):
            continue
        vector, reasons = panel_vector(p, features, cutoff)
        if not reasons:
            grouped.setdefault(p.participant_id, []).append((p, vector, candidate.stability_evidence_id))
    if len(grouped) < minimum or minimum < 9:
        raise ValueError(f'Reference needs {minimum} eligible independent people; found {len(grouped)}.')
    rng = np.random.default_rng(seed)
    chosen = []
    for person in sorted(grouped):
        rows = sorted(grouped[person], key=lambda item: item[0].day)
        chosen.append(rows[int(rng.integers(len(rows)))])
    raw = np.asarray([item[1] for item in chosen])
    model_ids = {f.model_id for item in chosen for f in item[0].features if f.name in features and f.name.startswith('text_')}
    if model_ids and (None in model_ids or len(model_ids) != 1):
        raise ValueError('Reference text features require one pinned model identity.')
    scaler = fit_scaler(raw, features)
    reference = {'features': list(features), 'cutoff': cutoff.isoformat(), 'seed': seed,
                 'text_model_id': next(iter(model_ids), None),
                 'people': [item[0].participant_id for item in chosen],
                 'selected_days': [item[0].day.isoformat() for item in chosen],
                 'evidence_ids': [item[2] for item in chosen], 'scaler': scaler,
                 'raw_matrix': raw.tolist(), 'matrix': transform(raw, scaler).tolist()}
    return {**reference, 'id': fingerprint(reference)}
