"""Independent numerical, censoring, leakage and interruption guards for the authored experiment."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from rhythm_dnb.contracts import EvaluationDay
from rhythm_dnb.dnb.classic import dnb_components
from rhythm_dnb.provenance import fingerprint, file_hash
from rhythm_dnb.research.discover import discover_matrices
from rhythm_dnb.research.evaluate import event_metrics
from rhythm_dnb.research.experiment import validate_protocol, forecast_rows, score_cohort, choose_threshold
from rhythm_dnb.research.experiment_data import (DOMAIN, OBJECTIVE, load_packet, save_json, split_text_groups, cohort_rows)
from rhythm_dnb.research.experiment_gpu import inference_tasks, attach_predictions, implementation_id, verify_new_checkpoint
from rhythm_dnb.measures.scaling import fit_scaler, transform
from rhythm_dnb.text.schema import CATEGORIES


def protocol():
    return json.loads((Path(__file__).resolve().parents[2] / 'configs/experiment.json').read_text())


def longitudinal(person='person', positive=True, days=14):
    start = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    return [{'participant_id': person, 'record_id': f'{person}-{i}', 'split': 'test',
             'observed_at': (start + timedelta(days=i)).isoformat(),
             'issued_at': (start + timedelta(days=i + 1)).isoformat(),
             'features': {k: float(i + j + 1) for j, k in enumerate(OBJECTIVE)},
             'texts': {c: person + ' text ' + c for c in ('emotion', 'diet', 'sleep', 'social')},
             'outcome': {'event_onset_at': (start + timedelta(days=6)).isoformat() if positive else None,
                         'followup_end_at': (start + timedelta(days=21)).isoformat(),
                         'label_observed_at': (start + timedelta(days=21)).isoformat()}} for i in range(days)]


class EndToEndExperimentTests(unittest.TestCase):
    def test_accuracy_is_not_notification_count_or_imputed_abstention(self):
        start = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        onset = start + timedelta(days=5)
        rows = [EvaluationDay('a', start + timedelta(days=i), 3., 1, onset) for i in range(3)]
        rows += [EvaluationDay('b', start, 0., 0), EvaluationDay('c', start, None, 0)]
        measured = event_metrics(rows, 1.)
        self.assertEqual(measured['risk_accuracy'], 1.)
        self.assertEqual(measured['risk_balanced_accuracy'], 1.)
        self.assertEqual(measured['true_alarms'], 1)
        self.assertEqual(measured['classified_days'], 4)
        self.assertEqual(measured['classification_coverage'], .8)
        abstained = event_metrics(rows, None)
        self.assertIsNone(abstained['risk_accuracy'])
        self.assertIsNone(abstained['risk_balanced_accuracy'])
        self.assertEqual(abstained['classified_days'], 0)

    def test_imbalance_does_not_create_high_balanced_accuracy(self):
        start = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        rows = [EvaluationDay(str(i), start, 0., 0) for i in range(9)]
        rows += [EvaluationDay('positive', start, 0., 1, start + timedelta(days=2))]
        measured = event_metrics(rows, 1., consecutive=1)
        self.assertEqual(measured['risk_accuracy'], .9)
        self.assertEqual(measured['risk_balanced_accuracy'], .5)

    def test_forecast_target_is_future_not_current_state(self):
        rows = longitudinal() + longitudinal('control', False)
        forecasts, events, periods, reasons = forecast_rows(rows, {r['record_id']: 1. for r in rows}, protocol())
        self.assertEqual(len(forecasts), 10)
        self.assertEqual([r.label for r in forecasts if r.participant_id == 'person'], [1] * 5)
        self.assertEqual([r.label for r in forecasts if r.participant_id == 'control'], [0] * 5)
        self.assertTrue(all(timedelta(days=1) <= r.onset - r.issued_at <= timedelta(days=5) for r in forecasts if r.label == 1))

    def test_claimed_followup_cannot_replace_missing_days(self):
        rows = longitudinal('control', False, days=10)
        forecasts, *_ = forecast_rows(rows, {}, protocol())
        self.assertTrue(any(r.label is None for r in forecasts))
        self.assertTrue(all(r.score is None for r in forecasts))
        rows = [r for r in longitudinal('control', False) if r['record_id'] != 'control-8']
        forecasts, *_ = forecast_rows(rows, {}, protocol())
        self.assertTrue(all(r.label is None for r in forecasts))

    def test_monitoring_calendar_does_not_shrink_when_forecast_is_missing(self):
        rows = longitudinal('control', False)
        rows = [r for r in rows if r['record_id'] != 'control-2']
        forecasts, _, periods, _ = forecast_rows(rows, {r['record_id']: 1. for r in rows}, protocol())
        self.assertEqual(len(forecasts), 5)
        self.assertIsNone(forecasts[2].score)
        self.assertEqual((periods[0].last_issue_day - periods[0].first_issue_day).days, 4)

    def test_text_identity_components_stay_together(self):
        rows = [dict(example_id='a', group_id='one', participant_id='p', text='睡了6小时'),
                dict(example_id='b', group_id='two', participant_id='q', text='睡了9小时'),
                dict(example_id='c', group_id='three', participant_id='q', text='今天吃饭挺正常')]
        result = split_text_groups(rows, 20)
        self.assertEqual(len({r['split'] for r in result}), 1)
        self.assertTrue(all('split' not in r for r in rows))

    def test_matrices_use_same_classic_formula_and_preserve_negative_findings(self):
        config = replace(validate_protocol(protocol()), bootstrap_repetitions=20, permutation_repetitions=19)
        rng = np.random.default_rng(23)
        stable = rng.normal(size=(30, 6)); pre = stable.copy()
        result = discover_matrices(stable, pre, list(OBJECTIVE), config)
        self.assertEqual(result['chosen'], [])
        for candidate in result['candidates']:
            manual = dnb_components(stable, OBJECTIVE, candidate['module'])
            self.assertAlmostEqual(manual['score'], np.prod(candidate['stable_components'][:2]) / candidate['stable_components'][2])
            self.assertFalse(candidate['all_three_conditions'])

    def test_future_values_and_labels_do_not_change_earlier_scores(self):
        config = validate_protocol(protocol())
        reference = np.random.default_rng(82).normal(size=(60, 6))
        scaler = fit_scaler(reference, OBJECTIVE); reference = transform(reference, scaler)
        rows = longitudinal()
        discovery = {'chosen': [{'module': list(OBJECTIVE[:2])}]}
        first, _ = score_cohort(rows, OBJECTIVE, reference, scaler, discovery, config)
        changed = deepcopy(rows)
        changed[-1]['features'] = dict.fromkeys(OBJECTIVE, 9999.)
        for row in changed:
            row['outcome'] = {'event_onset_at': None}
        second, _ = score_cohort(changed, OBJECTIVE, reference, scaler, discovery, config)
        for method in first:
            self.assertEqual(first[method][rows[0]['record_id']], second[method][rows[0]['record_id']])

    def test_change_comparator_does_not_bridge_missing_days(self):
        config = validate_protocol(protocol())
        reference = np.random.default_rng(82).normal(size=(60, 6))
        scaler = fit_scaler(reference, OBJECTIVE)
        rows = longitudinal(); rows.pop(1)
        scores, _ = score_cohort(rows, OBJECTIVE, transform(reference, scaler), scaler, {'chosen': []}, config)
        self.assertIsNone(scores['previous_day_change'][rows[1]['record_id']])
        self.assertTrue(all(v is None for v in scores['dnb'].values()))

    def test_no_threshold_is_not_default_half(self):
        forecasts, events, periods, _ = forecast_rows(longitudinal(), {}, protocol())
        setting = {**protocol(), 'calibration_min_event_people': 1}
        result = choose_threshold(forecasts, events, periods, setting)
        self.assertIsNone(result['threshold'])
        self.assertEqual(result['status'], 'no_evaluable_scores')

    def test_positive_only_calibration_cannot_certify_false_alarm_budget(self):
        source = longitudinal()
        forecasts, events, periods, _ = forecast_rows(source, {r['record_id']: 2. for r in source}, protocol())
        setting = {**protocol(), 'calibration_min_event_people': 1}
        result = choose_threshold(forecasts, events, periods, setting)
        self.assertIsNone(result['threshold'])
        self.assertEqual(result['status'], 'insufficient_calibration_negative_days')

    def test_packet_change_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); save_json(root / 'test.json', [])
            files = {'test.json': file_hash(root / 'test.json')}
            save_json(root / 'manifest.json', {'domain': DOMAIN, 'files': files, 'id': fingerprint(files)})
            load_packet(root)
            (root / 'test.json').write_text('[1]')
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_packet(root)

    def test_existing_or_continued_checkpoint_is_rejected_before_inference(self):
        model = {'initialization': {'old': 'model'}, 'dataset': {'id': 'corpus'}}
        with patch('rhythm_dnb.text.checkpoint.inspect_checkpoint', return_value=(model, 'id')):
            with self.assertRaisesRegex(ValueError, 'directly'):
                verify_new_checkpoint('unused', {'text-development': {'manifest': {'id': 'corpus'}}}, {})

    def test_prediction_join_checks_exact_text_and_scores(self):
        data = {role: longitudinal(role, days=1) for role in ('reference', 'train', 'validation', 'test')}
        data['text-development'] = {'manifest': {'id': 'corpus'}}
        tasks = inference_tasks(data)
        predictions = {key: {k: 25. for k in CATEGORIES[task['category']][1]} for key, task in tasks.items()}
        result = {'packet_id': 'packet', 'tasks_id': fingerprint(tasks), 'checkpoint_dataset_id': 'corpus',
                  'initialization': None, 'implementation_id': implementation_id(),
                  'predictions': predictions, 'experimental_uncalibrated': True}
        result['id'] = fingerprint(result)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.json'; save_json(path, result)
            attach_predictions(data, {'id': 'packet'}, path)
            self.assertEqual(data['test'][0]['features']['text_anxiety_intensity'], 25.)
            data['test'][0]['texts']['emotion'] = 'changed text'
            with self.assertRaisesRegex(ValueError, 'do not belong'):
                attach_predictions(data, {'id': 'packet'}, path)


if __name__ == '__main__':
    unittest.main()
