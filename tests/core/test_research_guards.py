"""Fitting cutoffs, nonlinear alternatives and reference eligibility."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest
import numpy as np
from rhythm_dnb.config import StudyConfig
from rhythm_dnb.contracts import EvaluationDay
from rhythm_dnb.dnb.reference import fit_reference
from rhythm_dnb.measures.panel import get_panel
from support.cohort import make_cohort
from rhythm_dnb.research.baselines import fit_baselines, deviation_and_trend
from rhythm_dnb.research.calibrate import calibrate
from rhythm_dnb.warning.windows import select_window


class ResearchGuardTests(unittest.TestCase):
    def test_invalid_study_choices_rejected(self):
        for settings in ({'reference_min_people': 8}, {'rolling_min_days': 29}, {'module_sizes': (2, 2)},
                         {'epsilon': float('nan')}, {'seed': True}, {'seed': -1}, {'min_lead_hours': 200}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                StudyConfig(**settings)

    def test_reference_ignores_unavailable_stability_evidence(self):
        cohort = make_cohort(people_per_role=4)
        rows = list(cohort['reference'])
        rows[0] = replace(rows[0], stability_available_at=datetime(2030, 1, 1, tzinfo=timezone.utc))
        with self.assertRaises(ValueError):
            fit_reference(rows, get_panel('objective8'), cohort['reference_cutoff'])
        with self.assertRaises(ValueError):
            fit_reference(cohort['reference'] + cohort['reference'][:1], get_panel('objective8'), cohort['reference_cutoff'])

    def test_single_sample_calibration_cannot_use_unknown_labels(self):
        t = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
        row = EvaluationDay('p', t, 3., 0, label_available_at=t + timedelta(days=9))
        with self.assertRaises(ValueError):
            calibrate([row], StudyConfig(), {'people': [], 'id': 'r'},
                      {'people': [], 'id': 'd', 'modules': []}, t + timedelta(days=2))

    def test_rolling_gap_counts_missing_days_not_rows(self):
        cohort = make_cohort(people_per_role=4)
        base = cohort['test'][0].request.history[0]
        # Missing three calendar days cannot be hidden by supplying 23 dense rows.
        last = base.day + timedelta(days=27)
        history = []
        for i in range(28):
            if i in (10, 11, 12):
                continue
            panel = replace(base, day=base.day + timedelta(days=i), features=tuple(
                replace(f, available_at=f.available_at + timedelta(days=i), measured_until=f.measured_until + timedelta(days=i))
                for f in base.features))
            history.append(panel)
        _, reason = select_window(history, base.participant_id, 'UTC', last, get_panel('objective8'),
            datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(reason, 'missing_run_too_long')

    def test_nonlinear_comparator_is_selected_when_signal_is_curved(self):
        rng = np.random.default_rng(21)
        x = rng.normal(size=(180, 2)); y = (np.abs(x[:, 0]) > .8).astype(int)
        people = np.repeat(np.arange(30), 6)
        models, result = fit_baselines(x, y, people, folds=3)
        self.assertNotEqual(result['selected'], 'logistic')
        self.assertGreater(result['candidates']['spline_logistic']['grouped_cv_average_precision'],
                           result['candidates']['logistic']['grouped_cv_average_precision'] + .15)
        self.assertFalse(result['test_used'])
        self.assertEqual(deviation_and_trend([[1, 2], [2, 2], [3, 2]]).shape, (6,))
