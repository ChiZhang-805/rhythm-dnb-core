"""Export, score and independently evaluate frozen exploratory warning predictors."""

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from rhythm_dnb.provenance import canonical_json, file_hash, fingerprint
from rhythm_dnb.research.experiment import forecast_rows, policy_settings
from rhythm_dnb.research.experiment_data import DOMAIN, load_packet, save_json
from rhythm_dnb.research.evaluate import event_metrics, replay
from rhythm_dnb.research.frozen_warning import (METHODS, PURPOSE, implementation_receipt, load_bundle,
    predict, predict_prepared, prepare_scores, require_unseen_people, validate_request)
from rhythm_dnb.timebase import instant
from tools.run_dnbr_cross_validation import verify_parent
from tools.run_dnbr_warning import read_json
from tools.run_reference_stability import forecast_prefix
from tools.run_temporal_warning import attach_preserved_predictions


def export_bundles(experiment, packet, predictions, output):
    # PSEUDOCODE: package every frozen fold, without refitting or choosing a fold by its test accuracy.
    experiment, output = Path(experiment).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve previous bundles; use a new output directory.')
    verify_parent(experiment)
    plan = read_json(experiment/'plan.json'); data, manifest = load_packet(packet); inferred = read_json(predictions)
    if (plan['packet_id'] != manifest['id'] or plan['inference_id'] != inferred['id'] or
            fingerprint({k:v for k,v in inferred.items() if k != 'id'}) != inferred['id']):
        raise ValueError('Packet or text predictions differ from the verified experiment.')
    # Export only the already validated arithmetic; a new algorithm needs a new evaluation first.
    root = Path(__file__).resolve().parents[1]
    for name in ('research/reference_stability.py', 'research/temporal_warning.py', 'measures/scaling.py'):
        if file_hash(root/'src/rhythm_dnb'/name) != file_hash(experiment/'source/src/rhythm_dnb'/name):
            raise ValueError('Scoring arithmetic changed after the preserved experiment.')
    scaler = read_json(experiment/'scaler.json'); identities = read_json(experiment/'reference-identities.json')
    with np.load(experiment/'reference-inputs.npz', allow_pickle=False) as saved:
        reference, draws = saved['reference'].copy(), saved['draws'].copy()
    exposed = set().union(*(set(p) for p in plan['people'].values()))
    for row in data['text-development']['rows']:
        person = row.get('participant_id')
        if person:
            exposed.update((person, person.removeprefix('rhythm:')))
    measurement = {'features': scaler['features'], 'text_model_id': inferred['model_id'],
                   'text_precision': inferred['precision'], 'text_inference_profile': inferred['profile'],
                   'objective_clock': 'midpoint_of_aware_sleep_interval_UTC',
                   'units': {'sleep_midpoint_h':'UTC_hours', 'sleep_duration_h':'hours', 'meal_interval_cv':'ratio',
                             'exercise_minutes':'minutes', 'resting_hr_bpm':'bpm', 'screen_time_min':'minutes',
                             'text_features':'0_to_100_uncalibrated_experimental_estimates'},
                   'scope': 'Six objective measures and four experimental text estimates; not the formal joint12 panel.'}
    common = {'purpose': PURPOSE, 'experiment_plan_id': plan['id'], 'scaler': scaler,
              'measurement_contract': measurement, 'measurement_contract_id': fingerprint(measurement),
              'reference_features': identities['features'], 'reference_people': identities['reference_people'],
              'replicates': plan['config']['replicates'], 'epsilon': plan['protocol']['epsilon'],
              'history_days': plan['temporal_config']['history_days'], 'protocol': plan['protocol'],
              'alarm_consecutive': plan['protocol']['alarm_consecutive'], 'cooldown_days': plan['protocol']['cooldown_days'],
              'previously_used_people': sorted(exposed), 'implementation': implementation_receipt(),
              'historical_exploration_only': True, 'clinical_accuracy_established': False}
    feature_binding = {'measurement': measurement, 'scaler': scaler, 'reference': reference.tolist(),
                       'draws': draws.tolist(), 'features': identities['features'], 'epsilon': common['epsilon'],
                       'history_days': common['history_days'], 'implementation': common['implementation']}
    common['feature_contract_id'] = fingerprint(feature_binding)
    output.mkdir(parents=True); exported = []
    for number, roles in enumerate(plan['rotations'], 1):
        folder = experiment/f'fold-{number}'
        models, policy = read_json(folder/'models.json'), read_json(folder/'frozen-policy.json')
        if policy['test_used'] is not False:
            raise ValueError('Threshold was not frozen independently of testing.')
        for method in METHODS:
            destination = output/f'fold-{number}'/method; destination.mkdir(parents=True)
            meta = {**common, 'fold': number, 'roles': roles, 'method': method,
                    'threshold': policy['methods'][method]['threshold'],
                    'fit_people': plan['people'][roles[0]], 'calibration_people': plan['people'][roles[1]],
                    'historical_test_people': plan['people'][roles[2]],
                    'parent_archive_sha256': file_hash(experiment/'archive-manifest.json')}
            if method.startswith('history_'):
                learned = models['learners'][method]
                if set(learned['training_people']) != set(meta['fit_people']):
                    raise ValueError('Saved learner has the wrong training cohort.')
                model = folder/(method+'.joblib')
                if file_hash(model) != learned['model_sha256']:
                    raise ValueError('Saved learner hash differs from the selection receipt.')
                shutil.copy2(model, destination/'model.joblib'); meta['columns'] = learned['columns']
            else:
                meta['half_life'] = models['smoothing'][method]['half_life']
            np.savez_compressed(destination/'reference.npz', reference=reference, draws=draws)
            save_json(destination/'metadata.json', meta)
            files = {p.name:file_hash(p) for p in destination.iterdir() if p.is_file()}
            save_json(destination/'manifest.json', {'files':files, 'id':fingerprint(files)})
            loaded = load_bundle(destination)
            exported.append({'fold':number,'method':method,'bundle_id':loaded['id'],'path':destination.relative_to(output).as_posix()})
    receipt = {'created_at':datetime.now(timezone.utc).isoformat(), 'plan_id':plan['id'], 'bundles':exported,
               'refitted':False, 'selected_by_test':False,
               'standalone_default':'fold-1/history_robust_dnb',
               'default_reason':'Original train/calibration/test order; not chosen by test rank.',
               'measurement_contract_id':common['measurement_contract_id']}
    save_json(output/'export.json', receipt)
    return receipt


def replay_bundles(experiment, packet, predictions, bundles, output):
    # PSEUDOCODE: regenerate all historical held-out predictions through the standalone API and compare every result.
    experiment, bundles, output = map(Path, (experiment, bundles, output))
    if output.exists():
        raise FileExistsError('Use a new replay receipt directory.')
    verify_parent(experiment); plan = read_json(experiment/'plan.json'); original = read_json(experiment/'result.json')
    data, manifest = load_packet(packet)
    binding = {'packet_id':plan['packet_id'],'inference_id':plan['inference_id'],
               'implementation_id':plan['prediction_implementation_id']}
    attach_preserved_predictions(data,manifest,predictions,binding)
    output.mkdir(parents=True); max_error=0.; total=0; alarms={}; people=set()
    for number, roles in enumerate(plan['rotations'], 1):
        selected = forecast_prefix(data[roles[2]],plan['protocol']['forecast_days'])
        source = Path(packet)/(roles[2]+'.json'); loaded={m:load_bundle(bundles/f'fold-{number}'/m) for m in METHODS}
        payload={'source_id':manifest['id']+'/'+roles[2], 'source_sha256':file_hash(source),'domain':manifest['domain'],
                 'measurement_contract_id':loaded[METHODS[0]]['metadata']['measurement_contract_id'],
                 'as_of':max(r['issued_at'] for r in selected),
                 'rows':[{**{k:r[k] for k in ('record_id','participant_id','observed_at','issued_at','features')},
                          'available_at':r['issued_at']} for r in selected]}
        save_json(output/(roles[2]+'-input.json'),payload)
        prepared=prepare_scores(loaded[METHODS[0]],payload,
            progress=lambda done,total: print(f'Replay fold {number}: {done}/{total} reference draws',flush=True) if done%50==0 else None)
        for method,bundle in loaded.items():
            response=predict_prepared(bundle,prepared); scores={r['record_id']:r['score'] for r in response['rows']}
            days,events,periods,_=forecast_rows(data[roles[2]],scores,plan['protocol'])
            expected=read_json(experiment/f'fold-{number}'/('test-'+method+'-forecasts.json'))
            if len(days)!=len(expected) or len(response['rows'])!=len(expected):
                raise ValueError('Standalone replay changed the scheduled forecast count.')
            for day,saved in zip(days,expected):
                actual=json.loads(canonical_json(asdict(day)))
                a,b=actual.pop('score'),dict(saved)['score']; previous={k:v for k,v in saved.items() if k!='score'}
                if actual!=previous or (a is None)!=(b is None):
                    raise ValueError('Replay changed a person, date, outcome or coverage.')
                if a is not None:
                    np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10); max_error=max(max_error,abs(a-b))
            metrics=event_metrics(days,bundle['metadata']['threshold'],events=events,monitoring=periods,**policy_settings(plan['protocol']))
            if metrics != original['folds'][number-1]['methods'][method]['test']:
                raise ValueError('Standalone predictor failed to reproduce frozen test metrics.')
            expected_alarms=replay(days,bundle['metadata']['threshold'],
                consecutive=plan['protocol']['alarm_consecutive'],cooldown_days=plan['protocol']['cooldown_days'])
            for actual,(day,warning,status) in zip(response['rows'],expected_alarms):
                if ((actual['participant_id'],instant(actual['issued_at']),actual['warning'],actual['status'])
                        != (day.participant_id,instant(day.issued_at),warning,status)):
                    raise ValueError('Standalone alarm or suppression status differs from the evaluated policy.')
            alarms[f'{number}:{method}']=sum(r['warning']==1 for r in response['rows'])
            save_json(output/(f'fold-{number}-'+method+'-predictions.json'),response)
            total+=len(response['rows']); people.update(r['participant_id'] for r in response['rows'])
        # The independent-evaluation API must refuse this already-inspected cohort.
        try:
            require_unseen_people(payload,list(loaded.values()))
        except ValueError:
            pass
        else:
            raise ValueError('Previously inspected test people were incorrectly accepted as unseen.')
        print('Standalone replay verified: fold '+str(number),flush=True)
    receipt={'created_at':datetime.now(timezone.utc).isoformat(),'rows_replayed':total,'methods':list(METHODS),
             'people':len(people),'max_absolute_score_error':max_error,'event_metrics_match':True,'every_alarm_matches':True,
             'known_people_rejected_by_independent_gate':True,'alarm_counts':alarms,'new_independent_cohort':False}
    save_json(output/'verification.json',receipt)
    files={p.name:file_hash(p) for p in output.iterdir() if p.is_file()}
    save_json(output/'manifest.json',{'files':files,'id':fingerprint(files)})
    return receipt


def evaluate(bundle, request, truth, output):
    # PSEUDOCODE: freeze prediction first; evaluate new people only against separate event/follow-up records.
    output=Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh independent evaluation directory.')
    validate_request(request,bundle['metadata']); people=require_unseen_people(request,[bundle])
    if request['domain']!=DOMAIN:
        raise ValueError('This outcome adapter is for the authored protocol only; observed outcomes require the formal validate workflow.')
    required={'source_id','source_sha256','domain','protocol_id','as_of','rows'}
    if set(truth)!=required or truth['domain']!=request['domain'] or truth['protocol_id']!=fingerprint(bundle['metadata']['protocol']):
        raise ValueError('Outcome provenance or endpoint protocol differs from the frozen study.')
    if (not isinstance(truth['source_id'],str) or not truth['source_id'].strip() or
            not isinstance(truth['source_sha256'],str) or len(truth['source_sha256'])!=64 or
            any(c not in '0123456789abcdef' for c in truth['source_sha256'])):
        raise ValueError('Independent outcome source identity and SHA256 required.')
    if not isinstance(truth['rows'],list) or not truth['rows'] or {r['participant_id'] for r in truth['rows']}!=people:
        raise ValueError('Outcome registry must cover exactly the forecast people.')
    lookup={}; times=set(); cutoff=instant(truth['as_of'])
    for row in truth['rows']:
        if set(row)!={'record_id','participant_id','observed_at','issued_at','outcome'} or set(row['outcome'])!={'event_onset_at','followup_end_at','label_observed_at'}:
            raise ValueError('Malformed outcome timeline.')
        if any(not isinstance(row[k],str) or not row[k].strip() for k in ('record_id','participant_id')):
            raise ValueError('Outcome records require stable nonempty identities.')
        issued=instant(row['issued_at']); key=(row['participant_id'],issued)
        if (issued.hour,issued.minute,issued.second,issued.microsecond)!=(12,0,0,0):
            raise ValueError('Outcome timeline differs from the daily UTC-noon schedule.')
        if row['record_id'] in lookup or key in times or instant(row['observed_at'])>issued or issued>cutoff:
            raise ValueError('Repeated or future outcome timeline record.')
        available=instant(row['outcome']['label_observed_at'])
        followup=instant(row['outcome']['followup_end_at'])
        if available>cutoff or followup>available:
            raise ValueError('Outcome not yet available at evaluation cutoff.')
        onset=row['outcome']['event_onset_at']
        if onset is not None and instant(onset)+timedelta(days=bundle['metadata']['protocol']['confirmation_days'])>min(followup,available):
            raise ValueError('Event lacks completed confirmation follow-up.')
        lookup[row['record_id']]=row; times.add(key)
    for row in request['rows']:
        saved=lookup.get(row['record_id'])
        if saved is None or any(saved[k]!=row[k] for k in ('participant_id','observed_at','issued_at')):
            raise ValueError('Prediction history and outcome timeline identities differ.')
    prefix=forecast_prefix(truth['rows'],bundle['metadata']['protocol']['forecast_days'])
    if not {r['record_id'] for r in request['rows']} <= {r['record_id'] for r in prefix}:
        raise ValueError('This fixed protocol evaluates the first five forecast days only.')
    output.mkdir(parents=True)
    save_json(output/'plan.json',{'bundle_id':bundle['id'],'request_sha256':fingerprint(request),
              'truth_sha256':fingerprint(truth),'protocol_id':truth['protocol_id'],'refit':False,'new_people_only':True})
    save_json(output/'input.json',request); save_json(output/'outcomes.json',truth)
    response=predict(bundle,request); save_json(output/'predictions.json',response)
    scores={r['record_id']:r['score'] for r in response['rows']}
    days,events,monitoring,reasons=forecast_rows(truth['rows'],scores,bundle['metadata']['protocol'])
    metrics=event_metrics(days,bundle['metadata']['threshold'],events=events,monitoring=monitoring,
                          **policy_settings(bundle['metadata']['protocol']))
    result={'bundle_id':bundle['id'],'domain':request['domain'],'metrics':metrics,'label_reasons':reasons,
            'independent_identifiers_verified':True,'source_authenticity_automatically_certified':False,
            'clinical_accuracy_established':False}
    save_json(output/'result.json',result)
    files={p.name:file_hash(p) for p in output.iterdir() if p.is_file()}
    save_json(output/'manifest.json',{'files':files,'id':fingerprint(files)})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    for name in ('export','replay'):
        command=sub.add_parser(name)
        for key in ('experiment','packet','predictions','output'):
            command.add_argument('--'+key,required=True)
        if name=='replay':
            command.add_argument('--bundles',required=True)
    for name in ('score','evaluate'):
        command=sub.add_parser(name)
        for key in ('bundle','input','output'):
            command.add_argument('--'+key,required=True)
        if name=='evaluate':
            command.add_argument('--truth',required=True)
    args=parser.parse_args()
    if args.command=='export':
        result=export_bundles(args.experiment,args.packet,args.predictions,args.output)
    elif args.command=='replay':
        result=replay_bundles(args.experiment,args.packet,args.predictions,args.bundles,args.output)
    elif args.command=='evaluate':
        result=evaluate(load_bundle(args.bundle),read_json(args.input),read_json(args.truth),args.output)
    else:
        if Path(args.output).exists():
            raise FileExistsError('Preserve previous predictions; use a new output path.')
        result=predict(load_bundle(args.bundle),read_json(args.input)); save_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','lineage','bundles')},ensure_ascii=False,default=str))
