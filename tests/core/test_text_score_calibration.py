"""Check that monotone score correction preserves order, missingness and held-out group boundaries."""

import unittest
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import numpy as np

from rhythm_dnb.provenance import fingerprint
from rhythm_dnb.research.experiment_data import TEXT
from tools.text_score_calibration import fit_curve, apply_curve, select_curves, development_rows
from tools.run_dnbr_calibration import read_decisions, verify_plan


class ScoreCalibrationTests(unittest.TestCase):
    def test_saved_plan_accepts_json_sequences_but_rejects_changed_content(self):
        # PSEUDOCODE: a disk round trip preserves plan identity; changed settings or forged content must still fail.
        plan = {'rotations': (('train', 'validation', 'test'),), 'strengths': (0., .25, .5, .75)}
        saved = json.loads(json.dumps({**plan, 'id': fingerprint(plan)}))
        verify_plan(saved, plan)
        saved['strengths'][-1] = 1.
        with self.assertRaises(ValueError):
            verify_plan(saved, plan)
        saved['id'] = fingerprint({k: v for k, v in saved.items() if k != 'id'})
        with self.assertRaises(ValueError):
            verify_plan(saved, plan)

    def test_saved_forecasts_restore_comparable_instants(self):
        # PSEUDOCODE: a saved offset timestamp must align with the same instant used by live forecast evaluation.
        row={'participant_id':'a','issued_at':'2026-01-01T08:00:00+08:00','score':1,'label':1}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'days.json';path.write_text(json.dumps([row]),encoding='utf-8')
            self.assertEqual(read_decisions(path)[0].issued_at,datetime(2026,1,1,tzinfo=timezone.utc))

    def test_blending_cannot_flatten_the_original_order(self):
        # PSEUDOCODE: even a constant fitted curve must retain strictly increasing distinct input scores.
        curve=fit_curve([10,20,30,40],[25,25,25,25],list('abcd'))
        values=apply_curve(curve,[0,10,20,30,40,100],.75)
        self.assertTrue(np.all(np.diff(values)>0))
        self.assertTrue(np.isnan(apply_curve(curve,[np.nan],.75)[0]))
        np.testing.assert_array_equal(apply_curve(curve,[1,99],0),[1,99])
        with self.assertRaises(ValueError):apply_curve(curve,[20],1)

    def test_identity_wins_when_raw_scores_are_correct(self):
        # PSEUDOCODE: group-held-out calibration cannot replace a zero-error raw predictor.
        rows=[]; predictions={}
        for i in range(20):
            row={'example_id':str(i),'category':'test','text':str(i),'participant_id':str(i),
                 'calibration_fold':i%2,'scores':{k[5:]:i*5 for k in TEXT}}
            rows.append(row);predictions[fingerprint(['test',str(i)])]=row['scores']
        models=select_curves(rows,predictions)
        self.assertTrue(all(v['strength']==0 for v in models.values()))
        rows[-1]['participant_id']=rows[0]['participant_id']
        with self.assertRaises(ValueError):select_curves(rows,predictions)

    def test_connected_text_families_stay_in_one_fold(self):
        # PSEUDOCODE: different people sharing an authored family cannot appear on both calibration sides.
        rows=[]
        for i in range(40):
            for j in range(2):
                rows.append({'example_id':f'{i}-{j}','group_id':f'family-{i}','participant_id':f'{i}-{j}',
                             'category':'emotion','text':chr(0x4e00+i)+('甲' if j else '乙'), 'split':'validation',
                             'scores':{'anxiety_intensity':20}})
        result=development_rows({'text-development':{'rows':rows}})
        for i in range(40):
            self.assertEqual(len({r['calibration_fold'] for r in result if r['group_id']==f'family-{i}'}),1)


if __name__=='__main__':unittest.main()
