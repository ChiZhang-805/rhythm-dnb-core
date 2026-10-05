"""Context diagnostics expose high absolute scores even when a paired direction is correct."""
import copy
import unittest
from rhythm_dnb.text.behavior import context_rows, context_report
from rhythm_dnb.text.schema import CATEGORIES


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.case = {'id': 'subject', 'category': 'sleep', 'metric': 'sleep_onset_difficulty',
                     'phenomenon': 'person', 'lower_is_explicit_absence': True,
                     'lower': {'anchor': '我很快睡着。', 'context': '同伴失眠，我很快睡着。'},
                     'higher': {'anchor': '我久久不能睡着。', 'context': '同伴很快睡着，我久久不能睡着。'}}
        self.guidance = {category: '评估本人相应时段的状态。' for category in CATEGORIES}

    def test_order_success_does_not_hide_context_error(self):
        predictions = [{'scores': {'sleep_onset_difficulty': v}} for v in [2., 92., 63., 95., 5., 94.]]
        result = context_report([self.case], predictions)['summary']
        self.assertEqual(result['original']['ordered'], 1)
        self.assertEqual(result['original']['mean_absence_score'], 63.)
        self.assertEqual(result['original']['mean_anchor_drift'], 32.)
        self.assertEqual(result['scoped']['mean_anchor_drift'], 2.5)
        self.assertEqual(result['original']['mean_contrast_drift'], 58.)

    def test_rows_preserve_family_and_instruction_order(self):
        rows = context_rows([self.case], self.guidance)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]['text'], self.case['lower']['anchor'])
        self.assertEqual(rows[2]['text'], self.case['lower']['context'])
        self.assertEqual(rows[4]['text'], self.guidance['sleep'] + '\n原始记录：' + self.case['lower']['context'])
        self.assertTrue(all(v is None for row in rows for v in row['scores'].values()))

    def test_duplicates_invalid_endpoints_and_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError): context_rows([self.case, self.case], self.guidance)
        invalid = copy.deepcopy(self.case); invalid['metric'] = 'sleep_quality'
        with self.assertRaises(ValueError): context_rows([invalid], self.guidance)
        invalid = copy.deepcopy(self.case); invalid['higher']['context'] = invalid['lower']['context']
        with self.assertRaises(ValueError): context_rows([invalid], self.guidance)
        for value in (float('nan'), float('inf'), -1, 101, True):
            with self.assertRaises(ValueError):
                context_report([self.case], [{'scores': {'sleep_onset_difficulty': value}}] * 6)


if __name__ == '__main__':
    unittest.main()
