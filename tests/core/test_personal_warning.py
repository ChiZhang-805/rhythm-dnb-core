"""Check personal-module topology, bounded fitting, person isolation and missing-predictor handling."""

import unittest

import numpy as np

from tools.personal_warning import branches, fit_model, predict_model, select_model


class PersonalWarningTests(unittest.TestCase):
    def test_negative_correlation_and_all_proper_branches(self):
        # PSEUDOCODE: anticorrelated features must cluster; enumerate nested branches without including the full universe.
        delta = np.array([[0, -.8, .1, .2, .01], [-.8, 0, .1, .1, .01],
                          [.1, .1, 0, .7, .01], [.2, .1, .7, 0, .01], [.01, .01, .01, .01, 0]])
        found = branches(delta, list('abcde'))
        self.assertEqual({tuple(m) for m in found}, {('a', 'b'), ('c', 'd'), ('a', 'b', 'c', 'd')})
        self.assertEqual(len(found), len(delta) - 2)

    def test_nonnegative_fit_rejects_reversed_risk_and_preserves_missing(self):
        # PSEUDOCODE: a reversed predictor gets no protective slope; a missing predictor is not turned into low risk.
        x = np.arange(20, dtype=float)[:, None]
        y = np.array([1] * 10 + [0] * 10)
        model = fit_model(x, y, [str(i) for i in range(20)], .01)
        self.assertAlmostEqual(model['coefficients'][0], 0., places=7)
        predicted = predict_model(model, [[1], [np.nan], [50]])
        self.assertTrue(np.isnan(predicted[1]))
        np.testing.assert_allclose(predicted[[0, 2]], [.5, .5], atol=1e-7)
        with self.assertRaises(ValueError):
            predict_model(model, [[-1]])

    def test_person_weighting_and_training_only_scale(self):
        # PSEUDOCODE: duplicating one person's records must not give that person more influence.
        x = np.array([[0., 1.], [1., 1.], [4., 1.], [8., 1.]])
        y = np.array([0, 0, 1, 1]); people = np.array(['a', 'b', 'c', 'd'])
        model = fit_model(x, y, people, .1)
        duplicated = fit_model(np.vstack([x, x[:1], x[:1]]), np.r_[y, 0, 0],
                               np.concatenate([people, ['a', 'a']]), .1)
        np.testing.assert_allclose(model['coefficients'], duplicated['coefficients'], atol=1e-7)
        np.testing.assert_allclose(model['center'], np.log1p(x).mean(axis=0), atol=1e-12)
        self.assertEqual(model['coefficients'][1], 0.)
        original_center = list(model['center'])
        predict_model(model, [[1e8, 1]])
        self.assertEqual(original_center, model['center'])

    def test_inner_folds_keep_whole_people_together(self):
        # PSEUDOCODE: verify all inner folds isolate people and the final model only uses the supplied development people.
        rng = np.random.default_rng(92)
        person_y = np.tile([0, 1], 12)
        y = np.repeat(person_y, 3); groups = np.repeat([str(i) for i in range(24)], 3)
        x = np.column_stack([np.maximum(0, y + rng.normal(0, .2, len(y))), rng.uniform(0, 2, len(y))])
        config = {'inner_folds': 3, 'penalties': [.01, .1]}
        model = select_model(x, y, groups, config, 19)
        tested = []
        for split in model['inner_splits']:
            self.assertFalse(set(split['fit_people']) & set(split['validation_people']))
            tested += split['validation_people']
        self.assertEqual(sorted(tested), sorted(set(groups)))
        self.assertEqual(model['training_people'], sorted(set(groups)))
        self.assertTrue(all(v >= 0 for v in model['coefficients']))
        self.assertIn(model['penalty'], config['penalties'])


if __name__ == '__main__':
    unittest.main()
