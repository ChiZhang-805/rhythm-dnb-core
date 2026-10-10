"""Protect participant denominators, provisional labels and the first-record prediction boundary."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rhythm_dnb.research.hospital_participants import (annotate_person, run_participants,
                                                       summarize_people, validate_plan)
from rhythm_dnb.research.hospital_proxy import METHODS, held_out_scores
from rhythm_dnb.provenance import file_hash


ROOT = Path(__file__).resolve().parents[2]


class HospitalParticipantTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT / 'configs/hospital_participants.json').read_text(encoding='utf-8'))
        self.scoring = json.loads((ROOT / 'configs/hospital_proxy.json').read_text(encoding='utf-8'))

    def examples(self, n=40):
        rng = np.random.default_rng(291); rows = []
        for person in range(n):
            for occasion in range(1, 5):
                values = dict(zip(self.scoring['features'], 20 + rng.normal(size=len(self.scoring['features']))))
                values.update(sleep_start_time=22*60, sleep_end_time=7*60,
                              breakfast_time=8*60, lunch_time=12*60, dinner_time=18*60)
                rows.append({'participant_id': f'p{person:02d}', 'record_id': f'p{person:02d}-{occasion}',
                    'occasion': occasion, 'values': values, 'sheet': f'Sheet{person}',
                    'date': {'status': 'year_missing', 'month_day': f'09-{occasion*3:02d}'}})
        return rows

    @staticmethod
    def shift(row, sleep, meal):
        for key in ('sleep_start_time', 'sleep_end_time'):
            row['values'][key] = (row['values'][key] + sleep) % 1440
        for key in ('breakfast_time', 'lunch_time', 'dinner_time'):
            row['values'][key] = (row['values'][key] + meal) % 1440

    def test_label_uses_later_two_domain_evidence_and_crosses_midnight(self):
        rows = self.examples(1)
        self.shift(rows[3], 120, 60)
        label = annotate_person(rows, self.scoring)
        self.assertEqual(label['labels'], {'30': 1, '60': 1, '90': 0})
        self.assertEqual(label['strongest_pair']['sleep_shift_min'], 120)
        self.assertIsNone(label['clinical_label'])
        self.assertIsNone(label['clinical_onset'])
        self.shift(rows[0], 600, 600)
        self.assertEqual(label, annotate_person(rows, self.scoring))
        rows[0]['date']['status'] = 'ambiguous_day'
        after = annotate_person(rows, self.scoring)
        self.assertFalse(after['dates_verified'])
        self.assertEqual(after['labels'], label['labels'])

    def test_cannot_combine_sleep_and_meals_from_different_pairs(self):
        rows = self.examples(1)
        self.shift(rows[2], 100, 50); self.shift(rows[3], 50, 100)
        label = annotate_person(rows, self.scoring)
        self.assertEqual(label['joint_shift_min'], 50)
        self.assertEqual(label['labels']['60'], 0)

    def test_partial_evidence_is_explicit_and_insufficient_evidence_is_unknown(self):
        rows = self.examples(1); rows[1]['values']['breakfast_time'] = None
        label = annotate_person(rows, self.scoring)
        self.assertEqual(label['complete_outcome_occasions'], 2)
        self.assertEqual(label['labels']['60'], 0)
        self.assertEqual(label['annotation_kind'], 'rule_inferred_observed_variability')
        self.assertEqual(len(label['missing_clock_evidence']), 1)
        rows[2]['values']['sleep_start_time'] = None
        label = annotate_person(rows, self.scoring)
        self.assertEqual(label['complete_outcome_occasions'], 1)
        self.assertIsNone(label['labels']['60'])

    def test_scores_and_thresholds_do_not_see_any_followup_or_endpoint_fields(self):
        rows = self.examples()
        original, lineage = held_out_scores(rows, self.scoring, scored_occasions=(1,), calibration_occasions=(1,))
        changed = deepcopy(rows)
        for row in changed:
            self.shift(row, 400, 300)
            if row['occasion'] > 1:
                for field in self.scoring['features']:
                    row['values'][field] = None
        altered, altered_lineage = held_out_scores(changed, self.scoring, scored_occasions=(1,), calibration_occasions=(1,))
        self.assertEqual(original, altered); self.assertEqual(lineage, altered_lineage)
        self.assertEqual(len(original), 40)
        for fold in lineage:
            self.assertTrue(all(r.endswith('-1') for r in fold['reference_records'] + fold['calibration_records']))
            self.assertFalse(set(fold['test_people']) & (set(fold['reference_people']) | set(fold['calibration_people'])))
        own = original[0]; changed = deepcopy(rows)
        next(r for r in changed if r['record_id'] == own['record_id'])['values']['skin_temp_c'] += 100
        _, changed_lineage = held_out_scores(changed, self.scoring, scored_occasions=(1,), calibration_occasions=(1,))
        self.assertEqual(lineage[own['fold']], changed_lineage[own['fold']])

    def test_one_person_one_count_and_unknown_not_negative(self):
        rows = []
        for i, (shift, alert) in enumerate([(80, 1), (0, 0), (90, 0), (0, 1), (None, 1)]):
            rows.append({'participant_id': str(i), 'joint_shift_min': shift,
                'labels': {str(c): int(shift >= c) if shift is not None else None for c in [30, 60, 90]},
                'scores': {m: float(i) for m in METHODS},
                'alerts': {m: {str(q): alert for q in [.8, .9, .95]} for m in METHODS},
                'complete_outcome_occasions': 3 if shift is not None else 1, 'dates_verified': True})
        result = summarize_people(rows, self.scoring)
        self.assertEqual(result['people'], 5)
        self.assertEqual(result['label_counts'], {'1': 2, '0': 2, 'None': 1})
        self.assertEqual(result['primary'][METHODS[0]]['n'], 4)
        self.assertEqual(result['primary'][METHODS[0]]['accuracy'], .5)
        self.assertEqual(result['paired_accuracy_difference']['estimate'], 0)
        with self.assertRaises(ValueError):
            summarize_people(rows + rows[:1], self.scoring)

    def test_freeze_roundtrip_manifest_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'normalized.json'; output = root / 'result'
            source.write_text(json.dumps({'source_sha256': 'fixture', 'rows': self.examples()}), encoding='utf-8')
            digest = file_hash(source)
            result = run_participants(source, ROOT / 'configs/hospital_participants.json', output)
            self.assertEqual(result['people'], 40)
            self.assertEqual(file_hash(source), digest)
            manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
            self.assertTrue(all(file_hash(output / f) == h for f, h in manifest['files'].items()))
            saved = json.loads((output / 'participants.json').read_text(encoding='utf-8'))
            self.assertEqual(len(saved), 40)
            self.assertTrue(all(r['clinical_label'] is None for r in saved))
            with self.assertRaises(FileExistsError):
                run_participants(source, ROOT / 'configs/hospital_participants.json', output)

    def test_rejects_changing_the_prediction_boundary(self):
        validate_plan(self.plan, self.scoring)
        self.plan['prediction_occasions'].append(3)
        with self.assertRaises(ValueError):
            validate_plan(self.plan, self.scoring)
        with self.assertRaises(ValueError):
            held_out_scores(self.examples(), self.scoring, scored_occasions=(4,))


if __name__ == '__main__':
    unittest.main()
