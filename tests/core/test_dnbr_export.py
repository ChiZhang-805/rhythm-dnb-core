"""Guard participant isolation, clock alignment and fixed-scale export to upstream R."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from rhythm_dnb.measures.scaling import fit_scaler, transform


def exporter():
    # PSEUDOCODE: load the standalone bridge without changing the frozen calculation package.
    path = Path(__file__).resolve().parents[2] / 'tools/export_dnbr.py'
    spec = importlib.util.spec_from_file_location('export_dnbr', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs():
    # PSEUDOCODE: build small software boundary fixtures, never a scientific performance dataset.
    rng = np.random.default_rng(41)
    names = ['sleep_midpoint_h', 'sleep_duration_h', 'text_anxiety_intensity']
    ref_values = rng.normal(size=(12, 3)) + [23, 7, 50]
    ref_values[:, 0] %= 24
    reference = [dict(participant_id=f'ref-{i}', split='reference',
                      features=dict(zip(names, row))) for i, row in enumerate(ref_values)]
    train, pairs = [], []
    for i in range(10):
        person = f'dev-{i}'
        for stage, day in [('stable', 1), ('pre_event', 4)]:
            values = rng.normal(size=3) + [23, 7, 50]
            values[0] %= 24
            train.append(dict(record_id=person + '-' + stage, participant_id=person, split='train',
                              observed_at=f'2026-01-{day:02}T00:00:00+00:00',
                              issued_at=f'2026-01-{day:02}T12:00:00+00:00',
                              features=dict(zip(names, values))))
        pairs.append(dict(participant_id=person, stable_record=person + '-stable',
                          pre_event_record=person + '-pre_event'))
    return reference, train, pairs, names, fit_scaler(ref_values, names)


class DNBrExportTests(unittest.TestCase):
    def test_matrix_orientation_and_one_fixed_reference_scale(self):
        reference, train, pairs, names, scaler = inputs()
        original = deepcopy(train)
        matrix, meta = exporter().prepare_matrices(reference, train, pairs, names, scaler)
        self.assertEqual(matrix.shape, (3, 20))
        self.assertEqual([r['stage'] for r in meta], ['stable'] * 10 + ['pre_event'] * 10)
        by_id = {r['record_id']: r for r in train}
        expected = transform([[by_id[r['record_id']]['features'][k] for k in names] for r in meta], scaler).T
        np.testing.assert_array_equal(matrix, expected)
        self.assertEqual(train, original)
        # Separate per-stage normalization would erase the SD differences DNB needs.
        self.assertFalse(np.allclose(matrix[:, :10].std(axis=1, ddof=1), 1))

    def test_held_out_people_are_rejected(self):
        for role in ('validation', 'test', 'reference'):
            args = inputs()
            args[1][0]['split'] = role
            with self.assertRaisesRegex(ValueError, 'isolated training'):
                exporter().prepare_matrices(*args)

    def test_reference_overlap_and_duplicate_people_are_rejected(self):
        args = inputs()
        args[0][0]['participant_id'] = args[1][0]['participant_id']
        with self.assertRaisesRegex(ValueError, 'isolated training'):
            exporter().prepare_matrices(*args)
        args = inputs()
        args[2].append(deepcopy(args[2][0]))
        with self.assertRaisesRegex(ValueError, 'Repeated person'):
            exporter().prepare_matrices(*args)

    def test_missing_and_constant_features_do_not_become_correlations(self):
        args = inputs()
        args[1][0]['features'][args[3][1]] = None
        with self.assertRaisesRegex(ValueError, 'complete paired'):
            exporter().prepare_matrices(*args)
        args = inputs()
        for row in args[1]:
            row['features'][args[3][1]] = 7.
        with self.assertRaisesRegex(ValueError, 'Constant feature'):
            exporter().prepare_matrices(*args)

    def test_chronology_and_identity_must_match(self):
        args = inputs()
        args[1][1]['issued_at'] = args[1][0]['issued_at']
        with self.assertRaisesRegex(ValueError, 'chronological'):
            exporter().prepare_matrices(*args)
        args = inputs()
        args[2][0]['pre_event_record'] = args[2][1]['pre_event_record']
        with self.assertRaisesRegex(ValueError, 'identity'):
            exporter().prepare_matrices(*args)

    def test_existing_results_are_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            exporter().export('unread', 'unread', None, 'unread', Path(__file__).parent)
