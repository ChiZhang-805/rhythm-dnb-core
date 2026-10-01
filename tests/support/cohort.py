"""Artificial integration fixtures. Observed tags exercise validation contracts, not genuine data.

These builders are test-only, are never packaged, and provide no research evidence.
"""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import numpy as np
from rhythm_dnb.contracts import DailyPanel, FeatureValue, Provenance, PredictionRequest, OutcomeLabel, OutcomeEvent, MonitoringPeriod
from rhythm_dnb.config import StudyConfig
from rhythm_dnb.dnb.classic import dnb_components
from rhythm_dnb.dnb.reference import ReferenceCandidate
from rhythm_dnb.measures.panel import get_panel, UNITS, CLOCKS
from rhythm_dnb.provenance import fingerprint
from rhythm_dnb.research.discover import DiscoveryPair


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
    # PSEUDOCODE: construct test-only observed-format values with clearly artificial source receipts.
    measured = datetime.combine(day + timedelta(days=1), datetime.min.time(), timezone.utc).replace(hour=4)
    provenance = Provenance('observed', 'unit-test-only:' + person + ':' + str(day), fingerprint([person, str(day), list(values)]), method='artificial_software_test_fixture')
    return DailyPanel(person, day, 'UTC', tuple(FeatureValue(name, float(value), UNITS[name], measured,
        provenance, measured_until=measured, model_id=model_id if name.startswith('text_') else None)
        for name, value in zip(features, values)))


def make_cohort(*, seed=20261001, panel_id='objective8', people_per_role=20, development_people=40):
    """Construct artificial software fixtures; the records must never enter a real study."""
    # PSEUDOCODE: build artificial contract fixtures with disjoint people, complete registries and explicit calendars.
    from rhythm_dnb.workflows.develop import LabeledCase
    rng = np.random.default_rng(seed); features = get_panel(panel_id); p = len(features)
    config = StudyConfig(panel_id=panel_id, seed=seed, reference_min_people=60, endpoint_min_people=60,
                         calibration_min_events=10, bootstrap_repetitions=40, permutation_repetitions=999)
    model_id = 'unit-test-semantic-model' if panel_id == 'joint12' else None
    reference = []
    for i in range(config.reference_min_people):
        panel = _panel(f'ref-{i}', date(2025, 1, 5), _physical(rng.normal(size=p), features), features, model_id)
        reference.append(ReferenceCandidate(panel, True, f'synthetic-stable-{i}', datetime(2025, 1, 15, tzinfo=timezone.utc)))
    pairs = []
    for i in range(development_people):
        common = rng.normal(); stable = .8 * common + rng.normal(scale=.6, size=p)
        pre = rng.normal(size=p); latent = rng.normal() * 3
        pre[:3] = [latent, -latent, latent] + rng.normal(scale=.08, size=3)
        first = _panel(f'dev-{i}', date(2025, 2, 1), _physical(stable, features), features, model_id)
        second = _panel(f'dev-{i}', date(2025, 3, 10), _physical(pre, features), features, model_id)
        pairs.append(DiscoveryPair(f'dev-{i}', tuple(_physical(stable, features)), tuple(_physical(pre, features)),
            datetime(2025, 3, 20, tzinfo=timezone.utc), f'unit-test-event-{i}', first, second,
            datetime(2025, 3, 15, 12, tzinfo=timezone.utc), 'unit-test-endpoint-protocol'))
    result = {'config': config, 'reference': reference, 'pairs': pairs, 'text_model_id': model_id,
              'reference_cutoff': datetime(2025, 1, 20, tzinfo=timezone.utc),
              'discovery_cutoff': datetime(2025, 3, 30, tzinfo=timezone.utc),
              'calibration_cutoff': datetime(2025, 6, 30, tzinfo=timezone.utc)}
    for role, start in [('calibration', date(2025, 5, 1)), ('test', date(2025, 8, 1))]:
        cases, events, monitoring = [], [], []
        for i in range(people_per_role):
            person = f'{role}-{i}'; positive = i % 2 == 0
            onset = datetime.combine(start + timedelta(days=24), datetime.min.time(), timezone.utc).replace(hour=12)
            if positive:
                events.append(OutcomeEvent(person, onset, onset + timedelta(days=2)))
            monitoring.append(MonitoringPeriod(person, start, start + timedelta(days=23), 'UTC'))
            for offset in range(24):
                issued = datetime.combine(start + timedelta(days=offset), datetime.min.time(), timezone.utc).replace(hour=12)
                x = rng.normal(scale=.6, size=p)
                label = int(positive and 17 <= offset <= 23)
                if label:
                    x[:3] = [7., -7., 7.] + rng.normal(scale=.3, size=3)
                panel = _panel(person, issued.date() - timedelta(days=1), _physical(x, features), features, model_id)
                request = PredictionRequest(person, issued, 'UTC', (panel,))
                outcome = OutcomeLabel(label, 'synthetic_scenario', onset if label else None)
                cases.append(LabeledCase(request, outcome, issued + timedelta(days=9), 'unit-test-endpoint-protocol'))
        result[role] = cases
        result[role + '_events'] = events
        result[role + '_monitoring'] = monitoring
    return result
