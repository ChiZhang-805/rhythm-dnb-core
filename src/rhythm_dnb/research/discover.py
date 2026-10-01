"""Group DNB discovery, resampling independent people and requiring all three conditions."""

from dataclasses import asdict, dataclass
from datetime import datetime
import numpy as np
from ..dnb.modules import candidate_modules
from ..measures.scaling import transform
from ..provenance import fingerprint
from ..timebase import instant
from ..contracts import DailyPanel
from ..warning.windows import panel_vector
from ..timebase import local_boundary
from datetime import timedelta


@dataclass(frozen=True)
class DiscoveryPair:
    participant_id: str
    stable: tuple[float, ...]
    pre_event: tuple[float, ...]
    evidence_available_at: datetime
    evidence_id: str
    stable_panel: DailyPanel | None = None
    pre_event_panel: DailyPanel | None = None
    event_onset: datetime | None = None
    outcome_protocol_id: str | None = None


def _component_weights(indices, width):
    # PSEUDOCODE: precompute linear averaging masks once for all candidate modules.
    amplitude = np.zeros((len(indices), width))
    internal = np.zeros((len(indices), width, width)); external = internal.copy()
    for i, (inside, outside, pairs) in enumerate(indices):
        amplitude[i, inside] = 1 / len(inside)
        internal[i, inside[pairs[0]], inside[pairs[1]]] = 1 / len(pairs[0])
        external[i][np.ix_(inside, outside)] = 1 / (len(inside) * len(outside))
    return amplitude, internal.reshape(len(indices), -1), external.reshape(len(indices), -1)


def _components(matrix, weights):
    # PSEUDOCODE: calculate covariance once -> vectorize fixed module component summaries.
    sd = matrix.std(axis=0, ddof=1)
    if np.any(sd <= 1e-12) or not np.isfinite(sd).all():
        return np.full((len(weights[0]), 3), np.nan)
    correlation = np.abs(np.corrcoef(matrix, rowvar=False))
    return np.column_stack((weights[0] @ sd, weights[1] @ correlation.ravel(), weights[2] @ correlation.ravel()))


def _directional_statistic(base, pre, epsilon):
    # PSEUDOCODE: require all three directions -> compute log index ratio without flooring a denominator.
    valid = np.isfinite(base).all(axis=1) & np.isfinite(pre).all(axis=1) & (base > 0).all(axis=1) & (pre > 0).all(axis=1)
    valid &= (base[:, 2] > epsilon) & (pre[:, 2] > epsilon)
    direction = valid & (pre[:, 0] > base[:, 0]) & (pre[:, 1] > base[:, 1]) & (pre[:, 2] < base[:, 2])
    result = np.zeros(len(base))
    result[direction] = np.log(pre[direction, 0] / base[direction, 0]) + np.log(pre[direction, 1] / base[direction, 1]) - np.log(pre[direction, 2] / base[direction, 2])
    return result


def discover(pairs, config, reference, cutoff, *, minimum_people=9):
    """Each person contributes one protocol-selected stable and pre-event day.

    Selection of those days must be locked before this function; all outcome
    evidence must be known by cutoff. No test persons or samples may be supplied.
    """
    # PSEUDOCODE: validate independent development pairs -> enumerate bounded modules -> bootstrap paired people.
    pairs = sorted(pairs, key=lambda p: p.participant_id)
    ids = [p.participant_id for p in pairs]
    if len(set(ids)) != len(ids) or set(ids) & set(reference['people']):
        raise ValueError('Development pairs overlap or reuse reference participants.')
    if len(ids) < minimum_people or minimum_people < 9:
        raise ValueError('Insufficient independent development pairs.')
    if any(not p.evidence_id or instant(p.evidence_available_at) > instant(cutoff) for p in pairs):
        raise ValueError('Development outcome evidence is unavailable at cutoff.')
    if not config.simulation:
        for pair in pairs:
            if pair.stable_panel is None or pair.pre_event_panel is None or pair.event_onset is None or not pair.outcome_protocol_id:
                raise ValueError('Real discovery requires source-backed panels, onset and a frozen endpoint protocol.')
            a, b = pair.stable_panel, pair.pre_event_panel
            if a.participant_id != pair.participant_id or b.participant_id != pair.participant_id or a.timezone != b.timezone or a.day >= b.day:
                raise ValueError('Discovery pair identity or chronological order differs.')
            for panel, vector in ((a, pair.stable), (b, pair.pre_event)):
                checked, reasons = panel_vector(panel, reference['features'], cutoff)
                if reasons or not np.array_equal(checked, np.asarray(vector)):
                    raise ValueError('Discovery vector does not match eligible source-backed panel.')
                if any(f.name.startswith('text_') and f.name in reference['features'] and f.model_id != reference['text_model_id'] for f in panel.features):
                    raise ValueError('Discovery text model differs from the frozen reference.')
            issued = instant(local_boundary(b.day + timedelta(days=1), b.timezone, hour=12))
            lead = instant(pair.event_onset) - issued
            if not timedelta(hours=config.min_lead_hours) <= lead <= timedelta(days=config.horizon_days) or instant(pair.event_onset) > instant(pair.evidence_available_at):
                raise ValueError('Pre-event pair or confirmation lies outside the locked prediction protocol.')
        if len({p.outcome_protocol_id for p in pairs}) != 1:
            raise ValueError('Mixed endpoint definitions in development.')
    stable = transform([p.stable for p in pairs], reference['scaler'])
    transition = transform([p.pre_event for p in pairs], reference['scaler'])
    if stable.shape != transition.shape or not np.isfinite(stable).all() or not np.isfinite(transition).all():
        raise ValueError('Development pairs must have a complete fixed feature universe.')
    names = reference['features']
    candidates = candidate_modules(names, method='bounded_exhaustive', sizes=config.module_sizes)
    modules = list(candidates.values())
    indices = []
    for module in modules:
        inside = np.array([names.index(name) for name in module])
        outside = np.array([j for j, name in enumerate(names) if name not in module])
        indices.append((inside, outside, np.triu_indices(len(module), 1)))
    weights = _component_weights(indices, len(names))
    base, pre = _components(stable, weights), _components(transition, weights)
    if not np.isfinite(base).all() or not np.isfinite(pre).all():
        raise ValueError('Constant or undefined development features.')
    full_direction = ((pre[:, 0] > base[:, 0]) & (pre[:, 1] > base[:, 1]) &
                      (pre[:, 2] < base[:, 2]) & (pre[:, 2] > config.epsilon))
    hits = np.zeros(len(modules), dtype=int)
    valid = np.zeros(len(modules), dtype=int)
    rng = np.random.default_rng(config.seed)
    for _ in range(config.bootstrap_repetitions):
        selected = rng.integers(0, len(ids), len(ids))
        a, b = _components(stable[selected], weights), _components(transition[selected], weights)
        ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & (a[:, 2] > config.epsilon) & (b[:, 2] > config.epsilon)
        direction = (b[:, 0] > a[:, 0]) & (b[:, 1] > a[:, 1]) & (b[:, 2] < a[:, 2])
        valid += ok
        hits += ok & direction
    # PSEUDOCODE: count failed replicates conservatively -> rank stable modules -> allow an empty result.
    stability = hits / config.bootstrap_repetitions
    # PSEUDOCODE: exchange paired condition labels -> use the largest candidate statistic for each null draw.
    # This project extension assumes within-person condition exchangeability under the global null.
    observed = _directional_statistic(base, pre, config.epsilon)
    null_max = np.empty(config.permutation_repetitions)
    permutation_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    for j in range(config.permutation_repetitions):
        swap = permutation_rng.integers(0, 2, size=(len(ids), 1)).astype(bool)
        a = _components(np.where(swap, transition, stable), weights)
        b = _components(np.where(swap, stable, transition), weights)
        null_max[j] = np.max(_directional_statistic(a, b, config.epsilon)) if np.isfinite(a).all() and np.isfinite(b).all() else np.inf
    adjusted_p = (1 + np.sum(null_max[:, None] >= observed[None, :], axis=0)) / (len(null_max) + 1)
    records = []
    for i, module in enumerate(modules):
        eligible = bool(full_direction[i] and stability[i] >= config.module_stability and adjusted_p[i] <= config.discovery_alpha)
        score0 = base[i, 0] * base[i, 1] / base[i, 2] if base[i, 2] > config.epsilon else None
        score1 = pre[i, 0] * pre[i, 1] / pre[i, 2] if pre[i, 2] > config.epsilon else None
        ratio = float(score1 / score0) if score0 is not None and score0 > 0 and score1 is not None else None
        records.append({'module': module, 'stability': float(stability[i]), 'valid_bootstraps': int(valid[i]),
                        'stability_monte_carlo_se': float(np.sqrt(stability[i] * (1 - stability[i]) / config.bootstrap_repetitions)),
                        'max_stat_adjusted_p': float(adjusted_p[i]),
                        'all_three_conditions': bool(full_direction[i]), 'eligible': eligible,
                        'score_ratio': ratio, 'stable_components': base[i].tolist(),
                        'pre_event_components': pre[i].tolist()})
    chosen = sorted((r for r in records if r['eligible']),
                    key=lambda r: (-r['stability'], -(r['score_ratio'] or 0), tuple(r['module'])))[:config.max_modules]
    payload = {'study_id': fingerprint(asdict(config)), 'reference_id': reference['id'], 'people': ids,
               'outcome_protocol_id': pairs[0].outcome_protocol_id,
               'cutoff': instant(cutoff).isoformat(), 'modules': [r['module'] for r in chosen],
               'status': 'frozen' if chosen else 'no_stable_modules', 'candidates': records,
               'bootstrap_repetitions': config.bootstrap_repetitions, 'seed': config.seed,
               'permutation_repetitions': config.permutation_repetitions,
               'permutation_p_resolution': 1 / (config.permutation_repetitions + 1),
               'discovery_alpha': config.discovery_alpha,
               'null_assumption': 'paired_condition_exchangeability; global-null familywise control',
               'selection': 'three_DNB_conditions_plus_person_bootstrap_plus_max_stat_permutation'}
    return {**payload, 'id': fingerprint(payload)}
