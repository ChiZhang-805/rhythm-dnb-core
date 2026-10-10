"""Exercise outcome isolation, availability, frozen bundle integrity and independent evaluation gates."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rhythm_dnb.measures.scaling import fit_scaler, transform
from rhythm_dnb.provenance import file_hash, fingerprint
from rhythm_dnb.research.experiment_data import save_json
from rhythm_dnb.research.frozen_warning import (PURPOSE, implementation_receipt, load_bundle,
    predict, prepare_scores, predict_prepared, require_unseen_people, validate_request)
from rhythm_dnb.research.reference_stability import reference_draws
from tools.run_dnbr_warning import read_json
from tools.use_research_warning import evaluate


class FrozenWarningTests(unittest.TestCase):
    def inputs(self):
        rng=np.random.default_rng(521); raw=rng.uniform(1,4,(30,3)); names=['a','b','c']
        scaler=fit_scaler(raw,names)
        meta={'purpose':PURPOSE,'method':'reference_robust_dnb','scaler':scaler,'reference_features':names,
              'reference_people':[f'r{i}' for i in range(30)],'replicates':4,'epsilon':1e-8,'history_days':3,
              'measurement_contract_id':'fixed-measurement','feature_contract_id':'fixed-reference',
              'implementation':implementation_receipt(),'previously_used_people':['old-person'],
              'half_life':2,'threshold':.8,'alarm_consecutive':2,'cooldown_days':7,
              'protocol':{'forecast_days':5,'horizon_days':7,'min_lead_hours':24,'confirmation_days':2,
                          'alarm_consecutive':2,'cooldown_days':7}}
        bundle={'metadata':meta,'reference':transform(raw,scaler),'draws':reference_draws(meta['reference_people'],4,515),
                'model':None,'id':'frozen-bundle'}
        start=datetime(2026,1,1,12,tzinfo=timezone.utc)
        rows=[{'record_id':f'{p}-{i}','participant_id':p,'observed_at':(start+timedelta(days=i)).isoformat(),
               'available_at':(start+timedelta(days=i+1)).isoformat(),'issued_at':(start+timedelta(days=i+1)).isoformat(),
               'features':dict(zip(names,rng.uniform(2,6,3).tolist()))} for p in ('new-a','new-b') for i in range(5)]
        payload={'source_id':'unit-test-only','source_sha256':'a'*64,'domain':'authored_simulation_not_clinical_validation',
                 'measurement_contract_id':'fixed-measurement','as_of':(start+timedelta(days=20)).isoformat(),'rows':rows}
        return bundle,payload

    def test_prediction_is_causal_and_truth_cannot_be_passed_to_scoring(self):
        bundle,payload=self.inputs(); first=predict(bundle,payload)
        changed=deepcopy(payload)
        for row in changed['rows']:
            if row['record_id']=='new-a-4' or row['participant_id']=='new-b':
                row['features']={'a':30.,'b':8.,'c':2.}
        second=predict(bundle,changed)
        for a,b in zip(first['rows'][:4],second['rows'][:4]):
            self.assertEqual(a,b)
        bad=deepcopy(payload); bad['rows'][0]['outcome']={'label':1}
        with self.assertRaisesRegex(ValueError,'outcome'):
            predict(bundle,bad)
        self.assertEqual(first['rows'][0]['state']['last_day'].isoformat(),'2026-01-01')

    def test_missing_current_data_abstains_and_cannot_be_imputed_as_stable(self):
        bundle,payload=self.inputs(); payload['rows'][1]['features']['a']=None
        actual=predict(bundle,payload)['rows'][1]
        self.assertIsNone(actual['score']); self.assertIsNone(actual['warning'])
        self.assertIsNone(actual['risk_exceeds_threshold']); self.assertEqual(actual['status'],'insufficient_data')
        self.assertEqual(actual['state']['consecutive'],0)

    def test_late_arrival_duplicate_wrong_units_and_model_are_rejected(self):
        bundle,payload=self.inputs()
        for change in ('late','duplicate','measurement','nan','negative','wrong_time','cutoff'):
            bad=deepcopy(payload)
            if change=='late': bad['rows'][0]['available_at']='2026-01-09T12:00:00+00:00'
            elif change=='duplicate': bad['rows'].append(bad['rows'][0])
            elif change=='measurement': bad['measurement_contract_id']='another-model'
            elif change=='nan': bad['rows'][0]['features']['a']=float('nan')
            elif change=='negative': bad['rows'][0]['features']['a']=-1
            elif change=='wrong_time': bad['rows'][0]['issued_at']='2026-01-02T13:00:00+00:00'
            elif change=='cutoff': bad['as_of']='2026-01-03T12:00:00+00:00'
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_request(bad,bundle['metadata'])

    def test_known_people_and_prepared_features_from_another_bundle_are_rejected(self):
        bundle,payload=self.inputs(); prepared=prepare_scores(bundle,payload)
        other=deepcopy(bundle); other['metadata']['feature_contract_id']='different-reference'
        with self.assertRaisesRegex(ValueError,'another reference'):
            predict_prepared(other,prepared)
        self.assertEqual(require_unseen_people(payload,[bundle]),{'new-a','new-b'})
        payload['rows'][0]['participant_id']='old-person'
        with self.assertRaisesRegex(ValueError,'previously used'):
            require_unseen_people(payload,[bundle])

    def test_bundle_round_trip_and_tamper_detection(self):
        bundle,payload=self.inputs()
        with tempfile.TemporaryDirectory() as path:
            folder=Path(path); np.savez_compressed(folder/'reference.npz',reference=bundle['reference'],draws=bundle['draws'])
            save_json(folder/'metadata.json',bundle['metadata'])
            files={p.name:file_hash(p) for p in folder.iterdir()}
            save_json(folder/'manifest.json',{'files':files,'id':fingerprint(files)})
            restored=load_bundle(folder)
            a,b=predict(bundle,payload),predict(restored,payload)
            np.testing.assert_allclose([r['score'] for r in a['rows']],[r['score'] for r in b['rows']],atol=0,rtol=0)
            (folder/'reference.npz').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'changed'):
                load_bundle(folder)

    def test_independent_evaluator_runs_without_refitting_and_keeps_censoring_unknown(self):
        bundle,payload=self.inputs(); start=datetime(2026,1,1,12,tzinfo=timezone.utc)
        rows=[]
        for p in ('new-a','new-b'):
            for i in range(14):
                row={'record_id':f'{p}-{i}','participant_id':p,'observed_at':(start+timedelta(days=i)).isoformat(),
                     'issued_at':(start+timedelta(days=i+1)).isoformat(),
                     'outcome':{'event_onset_at':None,'followup_end_at':(start+timedelta(days=20)).isoformat(),
                                'label_observed_at':(start+timedelta(days=20)).isoformat()}}
                if p=='new-b':
                    row['outcome']['followup_end_at']=(start+timedelta(days=3)).isoformat()
                rows.append(row)
        truth={'source_id':'separate-unit-test-only','source_sha256':'b'*64,'domain':payload['domain'],
               'protocol_id':fingerprint(bundle['metadata']['protocol']),'as_of':payload['as_of'],'rows':rows}
        with tempfile.TemporaryDirectory() as path:
            for change in ('future_followup','unconfirmed_event','wrong_schedule','observed_endpoint'):
                bad=deepcopy(truth); inputs=deepcopy(payload)
                if change=='future_followup': bad['rows'][0]['outcome']['followup_end_at']='2027-01-01T12:00:00+00:00'
                elif change=='unconfirmed_event': bad['rows'][0]['outcome']['event_onset_at']='2026-01-21T12:00:00+00:00'
                elif change=='wrong_schedule': bad['rows'][-1]['issued_at']='2026-01-15T13:00:00+00:00'
                elif change=='observed_endpoint':
                    bad['domain']=inputs['domain']='observed_exploratory_not_validated'
                with self.subTest(change=change),self.assertRaises(ValueError):
                    evaluate(bundle,inputs,bad,Path(path)/change)
                self.assertFalse((Path(path)/change).exists())
            output=Path(path)/'evaluation'; result=evaluate(bundle,payload,truth,output)
            self.assertEqual(result['metrics']['label_coverage'],.5)
            self.assertEqual(result['label_reasons']['right_censored_or_unconfirmed'],5)
            self.assertFalse(read_json(output/'plan.json')['refit'])
            self.assertEqual(read_json(output/'input.json'),payload)
            self.assertEqual(read_json(output/'outcomes.json'),truth)
            self.assertTrue((output/'manifest.json').is_file())
            bundle['metadata']['previously_used_people'].append('new-a')
            refused=Path(path)/'refused'
            with self.assertRaisesRegex(ValueError,'previously used'):
                evaluate(bundle,payload,truth,refused)
            self.assertFalse(refused.exists())


if __name__=='__main__':
    unittest.main()
