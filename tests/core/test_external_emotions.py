"""Keep human ordinal emotion references distinct from guessed continuous degree labels."""

import copy
import unittest
from rhythm_dnb.text.external import ordinal_emotion_report


class ExternalEmotionTests(unittest.TestCase):
    def test_perfect_presence_does_not_imply_correct_intensity_ordering(self):
        labels = [0, 0, 0, 1, 2, 3]
        scores = [0., 0., 0., 90., 60., 30.]
        rows = [{'example_id': str(i), 'category': 'emotion', 'external_scale': 'ordinal_0_3',
                 'external_reference': {'joy': value, 'sadness': value}} for i, value in enumerate(labels)]
        predictions = [{'scores': {'joy_intensity': value, 'sadness_intensity': value}} for value in scores]
        for metric in ordinal_emotion_report(rows, predictions)['metrics'].values():
            self.assertEqual(metric['presence_roc_auc'], 1.)
            self.assertEqual(metric['positive_intensity']['n'], 3)
            self.assertEqual(metric['positive_intensity']['spearman'], -1.)

    def test_ordering_imbalance_and_no_clinical_scale_conversion(self):
        rows = [{'example_id': str(i), 'category': 'emotion', 'external_scale': 'ordinal_0_3',
                 'external_reference': {'joy': i, 'sadness': 3-i}} for i in range(4)]
        predictions = [{'scores': {'joy_intensity': float(10+i*20), 'sadness_intensity': float(10+i*20)}} for i in range(4)]
        report = ordinal_emotion_report(rows, predictions)
        self.assertAlmostEqual(report['metrics']['joy_intensity']['spearman'], 1.)
        self.assertAlmostEqual(report['metrics']['sadness_intensity']['spearman'], -1.)
        self.assertIsNone(report['scale_conversion'])
        self.assertIsNone(report['clinical_accuracy'])
        self.assertEqual(report['metrics']['joy_intensity']['presence_roc_auc'], 1.)
        self.assertEqual(report['metrics']['sadness_intensity']['presence_roc_auc'], 0.)
        changed = copy.deepcopy(rows); changed[0]['external_reference']['joy'] = .5
        with self.assertRaises(ValueError): ordinal_emotion_report(changed, predictions)
        predictions[0]['scores']['joy_intensity'] = float('nan')
        with self.assertRaises(ValueError): ordinal_emotion_report(rows, predictions)

    def test_constant_references_do_not_create_undefined_statistics(self):
        rows = [{'example_id': str(i), 'category': 'emotion', 'external_scale': 'ordinal_0_3',
                 'external_reference': {'joy': 0, 'sadness': 0}} for i in range(3)]
        predictions = [{'scores': {'joy_intensity': 20., 'sadness_intensity': 30.}} for _ in rows]
        report = ordinal_emotion_report(rows, predictions)
        for metric in report['metrics'].values():
            self.assertIsNone(metric['spearman']); self.assertIsNone(metric['presence_average_precision'])
            self.assertEqual(metric['positive_intensity']['n'], 0)
            self.assertIsNone(metric['positive_intensity']['spearman'])
