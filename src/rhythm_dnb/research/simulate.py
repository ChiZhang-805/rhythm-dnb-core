"""Mechanism tests and fully labeled synthetic integration cohorts, never real validation."""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import numpy as np
from ..contracts import DailyPanel, FeatureValue, Provenance, PredictionRequest, OutcomeLabel
from ..config import StudyConfig
from ..dnb.classic import dnb_components
from ..dnb.reference import ReferenceCandidate
from ..measures.panel import get_panel, UNITS, CLOCKS
from ..provenance import fingerprint
from .discover import DiscoveryPair


def simulate_mechanism(*, seeds=range(30), samples=200):
    # PSEUDOCODE: contrast coordinated critical-like fluctuations with mean/noise/phase confounds across seeds.
    names = tuple('abcdef'); module = ('a', 'b', 'c'); results = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        common = rng.normal(size=(samples, 1))
        stable = .5 * common + rng.normal(size=(samples, 6))
        critical = rng.normal(size=(samples, 6))
        latent = rng.normal(size=samples) * 3
        critical[:, :3] = latent[:, None] * [1, -1, 1] + rng.normal(scale=.15, size=(samples, 3))
        time = np.arange(samples)
        periodic = np.column_stack([np.sin(2 * np.pi * time / 24 + j) for j in range(6)]) + rng.normal(scale=.2, size=(samples, 6))
        scenarios = {'stable': stable, 'coordinated_critical': critical,
                     'mean_shift_only': stable + np.array([4, 4, 4, 0, 0, 0]),
                     'independent_noise': stable + rng.normal(scale=3, size=stable.shape),
                     'stable_periodic': periodic, 'stable_late_phase': np.roll(periodic, 4, axis=0),
                     'shared_model_error': stable + np.column_stack([latent, latent, latent, np.zeros((samples, 3))]),
                     'abrupt_shock_no_precursor': np.vstack((stable[:-1], stable[-1] + 10))}
        missing = stable.copy(); missing[::3, :3] = np.nan; scenarios['missingness'] = missing
        for name, matrix in scenarios.items():
            result = dnb_components(matrix, names, module)
            results.setdefault(name, []).append(result)
    return {'domain': 'synthetic_mechanism_only', 'module': module, 'seeds': list(seeds),
            'scenarios': {name: {'valid': sum(r['valid'] for r in rows),
                'median_score': float(np.median([r['score'] for r in rows if r['valid']])),
                'score_range': np.quantile([r['score'] for r in rows if r['valid']], [.1, .9]).tolist()}
                for name, rows in results.items()},
            'interpretation': 'Shared text-model error can mimic coordination; DNB alone does not establish causality.'}


def _physical(values, features):
    # PSEUDOCODE: map latent simulation coordinates to bounded physical-looking units without claiming observations.
    means = [4., 7., 8., 20., 10., .55, 7000., 65., 35., 35., 40., 35.]
    scales = [.35, .25, .35, .35, .4, .03, 250., 2., 4., 4., 4., 4.]
    x = np.asarray(means[:len(features)]) + np.asarray(values) * scales[:len(features)]
    for j, name in enumerate(features):
        if name in CLOCKS:
            x[..., j] %= 24
        elif name == 'activity_ra':
            x[..., j] = np.clip(x[..., j], .01, .99)
        elif name.startswith('text_'):
            x[..., j] = np.clip(x[..., j], 0, 100)
        elif name == 'sleep_duration_h':
            x[..., j] = np.clip(x[..., j], 1, 15)
        elif name == 'resting_hr_bpm':
            x[..., j] = np.clip(x[..., j], 25, 180)
        else:
            x[..., j] = np.maximum(x[..., j], 1)
    return x


def _panel(person, day, values, features, model_id=None):
    # PSEUDOCODE: mark every simulated cell and its true availability explicitly.
    measured = datetime.combine(day + timedelta(days=1), datetime.min.time(), timezone.utc).replace(hour=4)
    provenance = Provenance('synthetic', person + ':' + str(day), fingerprint([person, str(day), list(values)]), method='simulation-v1')
    return DailyPanel(person, day, 'UTC', tuple(FeatureValue(name, float(value), UNITS[name], measured,
        provenance, measured_until=measured, model_id=model_id if name.startswith('text_') else None)
        for name, value in zip(features, values)))


def simulate_cohort(*, seed=20261001, panel_id='objective8', people_per_role=20, development_people=40):
    """Generate four disjoint roles on ordered calendars for an executable research example."""
    # PSEUDOCODE: generate independent references -> paired group transition -> daily calibration/test cases.
    from ..workflows.develop import LabeledCase
    rng = np.random.default_rng(seed); features = get_panel(panel_id); p = len(features)
    config = StudyConfig(panel_id=panel_id, simulation=True, seed=seed, bootstrap_repetitions=40)
    model_id = 'synthetic-semantic-v1' if panel_id == 'joint12' else None
    reference = []
    for i in range(config.reference_min_people):
        panel = _panel(f'ref-{i}', date(2025, 1, 5), _physical(rng.normal(size=p), features), features, model_id)
        reference.append(ReferenceCandidate(panel, True, f'synthetic-stable-{i}', datetime(2025, 1, 15, tzinfo=timezone.utc)))
    pairs = []
    for i in range(development_people):
        common = rng.normal(); stable = .8 * common + rng.normal(scale=.6, size=p)
        pre = rng.normal(size=p); latent = rng.normal() * 3
        pre[:3] = [latent, -latent, latent] + rng.normal(scale=.08, size=3)
        pairs.append(DiscoveryPair(f'dev-{i}', tuple(_physical(stable, features)), tuple(_physical(pre, features)),
                                    datetime(2025, 3, 20, tzinfo=timezone.utc), f'synthetic-event-{i}'))
    result = {'config': config, 'reference': reference, 'pairs': pairs, 'text_model_id': model_id,
              'reference_cutoff': datetime(2025, 1, 20, tzinfo=timezone.utc),
              'discovery_cutoff': datetime(2025, 3, 30, tzinfo=timezone.utc),
              'calibration_cutoff': datetime(2025, 6, 30, tzinfo=timezone.utc)}
    for role, start in [('calibration', date(2025, 5, 1)), ('test', date(2025, 8, 1))]:
        cases = []
        for i in range(people_per_role):
            person = f'{role}-{i}'; positive = i % 2 == 0
            onset = datetime.combine(start + timedelta(days=24), datetime.min.time(), timezone.utc).replace(hour=12)
            for offset in range(24):
                issued = datetime.combine(start + timedelta(days=offset), datetime.min.time(), timezone.utc).replace(hour=12)
                x = rng.normal(scale=.6, size=p)
                label = int(positive and 17 <= offset <= 23)
                if label:
                    x[:3] = [7., -7., 7.] + rng.normal(scale=.3, size=3)
                panel = _panel(person, issued.date() - timedelta(days=1), _physical(x, features), features, model_id)
                request = PredictionRequest(person, issued, 'UTC', (panel,))
                outcome = OutcomeLabel(label, 'synthetic_scenario', onset if label else None)
                cases.append(LabeledCase(request, outcome, issued + timedelta(days=9)))
        result[role] = cases
    return result
