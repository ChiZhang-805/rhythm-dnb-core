"""Keep future measurements, background people and exploratory proxy labels separate."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from rhythm_dnb.research.hospital_proxy import (METHODS, check_plan, circular_difference,
    held_out_scores, metrics, proxy_change)


class HospitalProxyTests(unittest.TestCase):
    def plan(self):
        return json.loads((Path(__file__).resolve().parents[2] / 'configs/hospital_proxy.json').read_text(encoding='utf-8'))

    def examples(self):
        plan = self.plan(); rng = np.random.default_rng(291)
        rows = []
        for person in range(40):
            for occasion in range(1, 5):
                values = dict(zip(plan['features'], 20 + rng.normal(size=len(plan['features']))))
                values.update(sleep_start_time=22*60, sleep_end_time=7*60,
                              breakfast_time=8*60, lunch_time=12*60, dinner_time=18*60)
                rows.append({'participant_id': f'p{person:02d}', 'record_id': f'p{person:02d}-{occasion}',
                             'occasion': occasion, 'values': values,
                             'date': {'status': 'year_missing', 'month_day': f'09-{occasion*3:02d}'}})
        return rows

    def test_midnight_and_two_domain_proxy_require_measured_later_values(self):
        rows = self.examples(); first, later = deepcopy(rows[0]), deepcopy(rows[1])
        later['values']['sleep_start_time'] += 60
        later['values']['sleep_end_time'] += 60
        for field in ('breakfast_time', 'lunch_time', 'dinner_time'):
            later['values'][field] += 60
        value = proxy_change(first, later)
        self.assertEqual(value['sleep_shift_min'], 60)
        self.assertEqual(value['joint_shift_min'], 60)
        self.assertEqual(value['gap_days'], 3)
        self.assertEqual(circular_difference(23*60+50, 10), 20)
        later['values']['breakfast_time'] = None
        self.assertIsNone(proxy_change(first, later)['joint_shift_min'])
        later['date']['status'] = 'ambiguous_day'
        self.assertEqual(proxy_change(first, later)['proxy_status'], 'unavailable_date')

    def test_no_year_inference_and_no_backward_or_same_date_pairs(self):
        rows = self.examples(); first, later = deepcopy(rows[0]), deepcopy(rows[1])
        later['date']['month_day'] = first['date']['month_day']
        self.assertEqual(proxy_change(first, later)['proxy_status'], 'nonincreasing_date')
        first['date']['month_day'], later['date']['month_day'] = '08-31', '09-01'
        self.assertEqual(proxy_change(first, later)['gap_days'], 1)
        later['date']['month_day'] = '09-21'
        self.assertIsNone(proxy_change(first, later)['gap_days'])

    def test_person_separation_and_targets_do_not_change_scores(self):
        rows = self.examples(); plan = self.plan()
        before, lineage = held_out_scores(rows, plan)
        self.assertEqual(len(before), 120)
        test_people = []
        for fold in lineage:
            fit, cal, test = (set(fold[k]) for k in ('reference_people', 'calibration_people', 'test_people'))
            self.assertFalse(fit & cal or fit & test or cal & test)
            self.assertFalse(fold['background_is_confirmed_stable'])
            self.assertTrue(all(r.endswith('-1') for r in fold['reference_records']))
            test_people.extend(test)
        self.assertEqual(len(set(test_people)), 40)
        changed = deepcopy(rows)
        for row in changed:
            row['values']['breakfast_time'] = 17*60
            row['values']['sleep_start_time'] = 13*60
            row['date']['month_day'] = None
            if row['occasion'] == 4:
                for field in plan['features']:
                    row['values'][field] += 10000
        after, later_lineage = held_out_scores(changed, plan)
        self.assertEqual(before, after)
        self.assertEqual(lineage, later_lineage)

    def test_missing_predictor_stays_abstention_not_zero_or_dropped(self):
        rows = self.examples(); rows[1]['values']['skin_temp_c'] = None
        scores, _ = held_out_scores(rows, self.plan())
        record = next(r for r in scores if r['record_id'] == rows[1]['record_id'])
        self.assertIsNone(record['scores'][METHODS[0]])
        self.assertIsNone(record['alerts'][METHODS[0]]['0.9'])
        self.assertEqual(len(scores), 120)

    def test_confusion_counts_and_unknown_proxy_denominator(self):
        rows = []
        for i, (shift, alert) in enumerate([(80, 1), (0, 0), (90, 0), (0, 1), (None, 1)]):
            rows.append({'participant_id': str(i), 'joint_shift_min': shift,
                         'scores': {m: float(i) for m in METHODS},
                         'alerts': {m: {'0.9': alert} for m in METHODS}})
        result = metrics(rows, METHODS[0], 60, .9)
        self.assertEqual(result['n'], 4)
        self.assertEqual([result[k] for k in ('tn', 'fp', 'fn', 'tp')], [1, 1, 1, 1])
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['balanced_accuracy'], .5)
        self.assertEqual(result['always_no_change_accuracy'], .5)

    def test_plan_rejects_using_label_clock_fields_as_predictors(self):
        plan = self.plan(); check_plan(plan)
        plan['features'].append('breakfast_time')
        with self.assertRaises(ValueError):
            check_plan(plan)


if __name__ == '__main__':
    unittest.main()
