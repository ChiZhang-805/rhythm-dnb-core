"""Regression checks for evidence masking, scope learning, blinded review and fair warning comparisons."""

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from rhythm_dnb.text.labels import evidence_target, validate_labels
from rhythm_dnb.text.model import regression_loss
from rhythm_dnb.text.schema import CATEGORIES, METRICS
from rhythm_dnb.text.review import coverage_report, training_readiness, export_blind_review, compare_blind_review
from rhythm_dnb.text.annotations import scope_token_targets
from rhythm_dnb.text.evaluate import longitudinal_report, within_person_network
from rhythm_dnb.text.experiment import seal_experiment, validate_holdout, PURPOSE, REFERENCE
from rhythm_dnb.provenance import fingerprint
from rhythm_dnb.config import StudyConfig
from rhythm_dnb.research.comparison import window_sensitivity, compare_methods
from rhythm_dnb.contracts import EvaluationDay, OutcomeEvent, MonitoringPeriod
from rhythm_dnb.text.study import summarize_repetitions


def record(identity, *, split='train', category='stress', value=40.):
    # PSEUDOCODE: construct an explicitly artificial software fixture with independent identity.
    return {'example_id': identity, 'group_id': identity, 'participant_id': identity,
        'split': split, 'origin': 'authored_simulation', 'annotation_source': REFERENCE,
        'category': category, 'text': '测试记录' + identity, 'scores': {k: value for k in CATEGORIES[category][1]}}


class ResearchRepairsTests(unittest.TestCase):
    def test_missing_labels_are_not_negative_evidence(self):
        row = record('missing', value=None)
        self.assertIsNone(evidence_target(row, 'stress_intensity'))
        outputs = {c: torch.full((1, len(keys)), .4, requires_grad=True) for c, (_, keys) in CATEGORIES.items()}
        outputs['_evidence'] = {c: torch.zeros((1, len(keys)), requires_grad=True) for c, (_, keys) in CATEGORIES.items()}
        regression_loss(outputs, torch.full((1, len(METRICS)), float('nan')), ['stress']).backward()
        self.assertEqual(outputs['_evidence']['stress'].grad.item(), 0.)
        row['label_states'] = {'stress_intensity': 'insufficient_evidence'}
        validate_labels(row)
        self.assertEqual(evidence_target(row, 'stress_intensity'), 0.)
        row['scores']['stress_intensity'] = 0
        with self.assertRaises(ValueError): validate_labels(row)
        row['label_states']['stress_intensity'] = 'explicit_absence'
        validate_labels(row)
        self.assertEqual(evidence_target(row, 'stress_intensity'), 1.)

    def test_scope_supervises_only_annotated_spans(self):
        row = record('scope', category='sleep')
        row['text'] = '他失眠。我很快睡着。'
        row['scope_targets'] = {'sleep_onset_difficulty': {'experiencer': 'self', 'period': 'latest_sleep',
            'evidence': [{'start': 4, 'end': 10, 'quote': row['text'][4:10]}],
            'excluded': [{'start': 0, 'end': 4, 'quote': row['text'][:4], 'reason': 'other_person'}]}}
        offsets = [(0, 0)] + [(i, i + 1) for i in range(len(row['text']))] + [(0, 0)]
        targets = np.asarray(scope_token_targets(row, offsets))
        column = [m[0] for m in METRICS].index('sleep_onset_difficulty')
        self.assertTrue(np.isnan(targets[0]).all())
        self.assertTrue(np.isnan(targets[-1]).all())
        self.assertEqual(targets[1:5, column].tolist(), [0.] * 4)
        self.assertEqual(targets[5:11, column].tolist(), [1.] * 6)
        self.assertEqual(np.isfinite(targets).sum(), 10)

    def test_supported_but_unscored_only_teaches_evidence(self):
        from rhythm_dnb.text.labels import has_supervision
        row = record('supported', value=None)
        row['label_states'] = {'stress_intensity': 'supported_unscored'}
        validate_labels(row)
        self.assertTrue(has_supervision(row))
        self.assertEqual(evidence_target(row, 'stress_intensity'), 1.)
        outputs = {c: torch.full((1, len(keys)), .4, requires_grad=True) for c, (_, keys) in CATEGORIES.items()}
        outputs['_evidence'] = {c: torch.zeros((1, len(keys)), requires_grad=True) for c, (_, keys) in CATEGORIES.items()}
        targets = torch.full((1, len(METRICS)), float('nan'))
        evidence = targets.clone(); evidence[0, 5] = 1.
        regression_loss(outputs, targets, ['stress'], evidence_labels=evidence).backward()
        self.assertEqual(outputs['stress'].grad.item(), 0.)
        self.assertLess(outputs['_evidence']['stress'].grad.item(), 0.)
        row['scores']['stress_intensity'] = 50.
        with self.assertRaises(ValueError): validate_labels(row)

    def test_repeated_selection_uses_all_seeds_and_rejects_test_scores(self):
        results = [{'arm': arm, 'seed': seed, 'validation_mae': value, 'test': None}
                   for arm, values in (('uneven', (1., 7.)), ('consistent', (3., 3.)))
                   for seed, value in enumerate(values)]
        result = summarize_repetitions(results)
        self.assertEqual(result['selected_arm'], 'consistent')
        self.assertFalse(result['automatic_model_promotion'])
        self.assertEqual(result['arms']['uneven']['validation_mean_mae'], 4.)
        with self.assertRaisesRegex(ValueError, 'same two'): summarize_repetitions(results[:-1])
        with self.assertRaisesRegex(ValueError, 'Duplicate'): summarize_repetitions(results + [results[0]])
        with self.assertRaisesRegex(ValueError, 'validation errors'):
            summarize_repetitions([{**r, 'test': 1.} for r in results])

    def test_refresh_preserves_partitions_times_and_read_only_source(self):
        import sqlite3
        from contextlib import closing
        from rhythm_dnb.provenance import canonical_json, file_hash
        from rhythm_dnb.text.expansion import refresh_source_corpus
        rows = [record(role + category, split=role, category=category)
                for role in ('train', 'validation', 'test') for category in CATEGORIES]
        selected = [r for r in rows if r['category'] == 'sleep']
        for row in selected:
            row.update(source_record_id='rhythm:' + row['example_id'], participant_id='rhythm:' + row['example_id'])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); database = root / 'source.sqlite'
            seal_experiment(rows, [], root / 'old')
            keys = CATEGORIES['sleep'][1]
            with closing(sqlite3.connect(database)) as db:
                db.execute('CREATE TABLE observations(record_id TEXT, participant_id TEXT, observed_at TEXT,' +
                           ','.join('text_' + k + ' REAL' for k in keys) + ')')
                db.execute('CREATE TABLE raw_inputs(record_id TEXT, payload TEXT)')
                db.execute('CREATE TABLE observation_provenance(record_id TEXT, payload TEXT)')
                for row in selected:
                    rid = row['example_id']
                    db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?)',
                               (rid, rid, '2026-01-02T08:00:00+08:00', 30., 65., 70., 75.))
                    db.execute('INSERT INTO raw_inputs VALUES (?,?)', (rid, canonical_json({
                        'sleep_description_text': rid + '昨晚入睡困难，夜里反复醒来，睡眠质量很差，醒后疲劳。'})))
                    db.execute('INSERT INTO observation_provenance VALUES (?,?)', (rid, canonical_json({
                        'provenance': {'text_' + k: {'method': 'manual_contextual_rewrite'} for k in keys}})))
                db.commit()
            digest = file_hash(database)
            result = refresh_source_corpus(root / 'old', database, root / 'new')
            self.assertEqual(file_hash(database), digest)
            self.assertEqual(result['changed_texts'], 3)
            self.assertEqual(result['restored_times'], 3)
            refreshed = [r for name in ('development', 'test') for r in json.loads((root / 'new' / (name + '.json')).read_text(encoding='utf-8'))['rows']]
            self.assertEqual({r['example_id']: r['split'] for r in refreshed}, {r['example_id']: r['split'] for r in rows})
            self.assertTrue(all(r['temporal_basis'] == 'source_record_timestamp_for_authored_text' for r in refreshed if r['category'] == 'sleep'))
            self.assertFalse(result['fresh_sealed_test'])

    def test_scope_head_trains_saves_and_reloads(self):
        from core.test_text_measurement import qwen_fixture
        from rhythm_dnb.text.train import train, predict_rows
        from rhythm_dnb.text.checkpoint import load_checkpoint, load_tokenizer
        from rhythm_dnb.text.dataset import ScoreDataset, Collator
        with tempfile.TemporaryDirectory() as folder:
            base, config, rows = qwen_fixture(folder)
            for row in rows:
                key = next(iter(row['scores']))
                row['scope_targets'] = {key: {'experiencer': 'self', 'period': 'current',
                    'evidence': [{'start': 0, 'end': 5, 'quote': row['text'][:5]}],
                    'excluded': [{'start': 5, 'end': len(row['text']), 'quote': row['text'][5:], 'reason': 'other_person'}]}}
            config.update(scope_loss_weight=.2, require_all_validation_metrics=True)
            result = train(rows, base, Path(folder) / 'run', config)
            model, tokenizer, _, _ = load_checkpoint(result['best_checkpoint'], base_path=base)
            self.assertIsNotNone(model.scope_head)
            predictions = predict_rows(model, tokenizer, rows[:2], config, torch.device('cpu'))
            self.assertGreater(predictions[0]['scope']['tokens'], 0)
            self.assertTrue(np.isfinite(predictions[0]['scope']['binary_cross_entropy']))
            self.assertGreater(result['test']['scope_supervision']['rows'], 0)
            weak = record('weak')
            dataset = ScoreDataset([weak], tokenizer, 512, explicit_evidence_only=True)
            batch = Collator(tokenizer)([dataset[0]])
            self.assertTrue(torch.isnan(batch['evidence_labels']).all())
            # Scope loss has a real gradient; it is not merely a stored annotation field.
            logits = torch.zeros((1, 2, len(METRICS)), requires_grad=True)
            outputs = {c: torch.full((1, len(keys)), .4, requires_grad=True) for c, (_, keys) in CATEGORIES.items()}
            outputs['_evidence'] = {c: torch.zeros((1, len(keys)), requires_grad=True) for c, (_, keys) in CATEGORIES.items()}
            outputs['_scope'] = logits
            labels = torch.full_like(logits, float('nan')); labels[0, 0, 5] = 1.; labels[0, 1, 5] = 0.
            regression_loss(outputs, torch.full((1, len(METRICS)), float('nan')), ['stress'], scope_labels=labels, scope_weight=.2).backward()
            self.assertLess(logits.grad[0, 0, 5], 0); self.assertGreater(logits.grad[0, 1, 5], 0)

    def test_missing_head_blocks_strict_training(self):
        partitions = {'train': [record('a')], 'validation': [record('b', split='validation')]}
        result = training_readiness(partitions, {'require_all_validation_metrics': True}, experimental=True)
        self.assertFalse(result['ready_for_training'])
        self.assertIn('sleep_onset_difficulty', result['blockers'][0])
        report = coverage_report(partitions['validation'])
        self.assertEqual(report['partitions']['validation']['metrics']['stress_intensity']['labeled'], 1)

    def test_blinded_reviews_exclude_old_answers_and_do_not_average(self):
        rows = [record(str(i), value=10. * i) for i in range(8)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'review'
            export_blind_review(rows, root, per_metric=6, seed=17)
            a, b, mapping = [json.loads((root / name).read_text()) for name in ('rater-A.json', 'rater-B.json', 'coordinator.json')]
            self.assertTrue(all(r['scores']['stress_intensity'] is None and 'example_id' not in r for r in a['rows']))
            with self.assertRaises(ValueError): compare_blind_review(mapping, a, b, tolerance=10)
            for slot, packet in (('A', a), ('B', b)):
                for row in packet['rows']:
                    row.update(rater_id=slot, human_reviewed=True, scores={'stress_intensity': 30. if slot == 'A' else 60.}, label_states={'stress_intensity': 'supported'})
            result = compare_blind_review(mapping, a, b, tolerance=10)
            self.assertEqual(result['adjudication_required'], 6)
            self.assertFalse(result['averaged_labels'])
            b['rows'][0]['rater_id'] = 'A'
            with self.assertRaises(ValueError): compare_blind_review(mapping, a, b, tolerance=10)

    def test_old_inspected_text_cannot_become_a_fresh_holdout(self):
        rows = [record(role + category, split=role, category=category) for role in ('train', 'validation', 'test') for category in CATEGORIES]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'earlier'):
                seal_experiment(rows, [rows[-1]], Path(folder) / 'bad')
            manifest = seal_experiment(rows, [], Path(folder) / 'good')
            self.assertTrue(manifest['fresh_sealed_test'])
            manifest['test_usage'] = 'development_regression_only_previously_exposed'
            with self.assertRaisesRegex(ValueError, 'already exposed'):
                validate_holdout({}, {'dataset': manifest})

    def test_longitudinal_changes_use_real_order_and_report_gaps(self):
        rows = [record(str(i), value=v) for i, v in enumerate((20., 50., 10.))]
        for i, row in enumerate(rows):
            row.update(participant_id='same', observed_at=f'2026-01-0{1 + i * 2}T12:00:00+00:00', temporal_basis='authored_timeline')
        predictions = [{'scores': {'stress_intensity': v}} for v in (20., 40., 20.)]
        report = longitudinal_report(rows[::-1], predictions[::-1])['stress_intensity']
        self.assertEqual(report['gap_hours_maximum'], 48.)
        self.assertEqual(report['direction_agreement_on_nonzero_reference_changes'], 1.)
        self.assertEqual(report['mae'], 15.)
        self.assertEqual(report['person_macro_change_mae'], 15.)
        self.assertAlmostEqual(report['person_macro_level_mae'], 20. / 3)
        with self.assertRaisesRegex(ValueError, 'one prediction'): longitudinal_report(rows, predictions[:-1])

    def test_change_error_weights_people_and_separates_levels_and_time_sources(self):
        rows, predictions = [], []
        for person, reference, estimated, basis in (
            ('offset', [10., 20., 30., 40.], [20., 30., 40., 50.], 'authored_timeline'),
            ('flat', [0., 40.], [20., 20.], 'source_record_timestamp_for_authored_text')):
            for i, (truth, estimate) in enumerate(zip(reference, estimated)):
                row = record(person + str(i), value=truth)
                row.update(participant_id=person, observed_at=f'2026-01-0{i+1}T12:00:00+00:00', temporal_basis=basis)
                rows.append(row); predictions.append({'scores': {'stress_intensity': estimate}})
        unrelated = record('unrelated', category='sleep', value=20.)
        unrelated.update(observed_at='2026-01-01T12:00:00+00:00', temporal_basis='unrelated_timestamp')
        rows.append(unrelated); predictions.append({'scores': unrelated['scores']})
        report = longitudinal_report(rows, predictions)['stress_intensity']
        self.assertEqual(report['mae'], 10.)  # Three unchanged differences and one 40-point change error.
        self.assertEqual(report['person_macro_change_mae'], 20.)
        self.assertEqual(report['person_macro_level_mae'], 15.)
        self.assertEqual({p['participant_id']: p['mae'] for p in report['per_person_changes']}, {'offset': 0., 'flat': 40.})
        self.assertNotIn('unrelated_timestamp', report['temporal_bases'])

    def test_error_summary_rejects_matrices_and_scalars(self):
        from rhythm_dnb.text.evaluate import errors
        for value in (1., [], [[1., 2.], [3., 4.]]):
            with self.assertRaises(ValueError): errors(value, value)

    def test_sensitivity_freezes_target_definition(self):
        base = StudyConfig()
        result = window_sensitivity(base, [{'rolling_days': 21, 'rolling_min_days': 18}, {'rolling_days': 35, 'rolling_min_days': 29}])
        self.assertEqual(len(result['candidates']), 2)
        self.assertTrue(all(c['study']['outcome_window'] == 7 for c in result['candidates']))
        with self.assertRaises(ValueError): window_sensitivity(base, [{'outcome_window': 14}])

    def test_network_checks_within_person_correlation_and_no_interpolation(self):
        rows, predictions = [], []
        keys = CATEGORIES['emotion'][1]
        for i, value in enumerate((10., 30., 70., 90.)):
            row = record(str(i), category='emotion', value=None)
            row.update(participant_id='same', observed_at=f'2026-01-0{i+1}T10:00:00+00:00')
            row['scores'].update(mood_valence=value, sadness_intensity=100. - value)
            rows.append(row)
            predictions.append({'scores': {k: value for k in keys}})
        result = within_person_network(rows, predictions)
        self.assertAlmostEqual(result['person_macro_absolute_correlation_error'], 2.)
        self.assertEqual(result['pairs'][0]['paired_instants'], 4)
        rows[0]['scores']['sadness_intensity'] = None
        self.assertEqual(within_person_network(rows, predictions)['pairs'][0]['paired_instants'], 3)
        rows[1]['scores']['sadness_intensity'] = None
        self.assertEqual(within_person_network(rows, predictions)['pairs'], [])
        with self.assertRaisesRegex(ValueError, 'one reference'):
            within_person_network(rows + [rows[-1]], predictions + [predictions[-1]])

    def test_warning_comparison_uses_same_calendars_and_never_selects_on_test(self):
        issued = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        methods = {}
        calendars, events, partitions = {}, {}, {}
        for role in ('calibration', 'test'):
            people = [role + '-event', role + '-stable']
            calendars[role] = [MonitoringPeriod(p, date(2026, 1, 1), date(2026, 1, 3), 'UTC') for p in people]
            onset = issued + timedelta(days=5)
            events[role] = [OutcomeEvent(people[0], onset, onset + timedelta(days=2))]
            partitions[role] = [EvaluationDay(p, issued + timedelta(days=i), 8. if j == 0 else 1., 1 if j == 0 else 0,
                onset if j == 0 else None, issued + timedelta(days=15), 'UTC') for j, p in enumerate(people) for i in range(3)]
        methods['dnb'] = partitions
        methods['trend'] = {k: [replace(r, score=r.score * 2) for r in v] for k, v in partitions.items()}
        args = dict(calibration_events=events['calibration'], test_events=events['test'], calibration_monitoring=calendars['calibration'],
            test_monitoring=calendars['test'], calibration_cutoff='2026-02-01T00:00:00+00:00', evaluation_as_of='2026-03-01T00:00:00+00:00',
            config=replace(StudyConfig(), calibration_min_events=1, calibration_min_negative_days=1), fitted_people=['training-only'])
        result = compare_methods(methods, **args)
        self.assertFalse(result['winner_selected_on_test'])
        self.assertEqual(result['methods']['dnb']['test']['event_sensitivity'], 1.)
        self.assertEqual(result['methods']['dnb']['threshold'] * 2, result['methods']['trend']['threshold'])
        paired = compare_methods(methods, **args, bootstrap_repetitions=40)
        comparison = paired['paired_differences']['comparisons'][0]
        self.assertEqual((comparison['left'], comparison['right']), ('dnb', 'trend'))
        for metric in comparison['metrics'].values():
            self.assertEqual(metric['estimate'], 0.)
            self.assertEqual(metric['percentile_95'], [0., 0.])
        self.assertFalse(paired['paired_differences']['thresholds_refitted'])
        self.assertLess(comparison['metrics']['event_sensitivity']['valid_replicates'], 40)
        self.assertEqual(comparison['metrics']['score_coverage']['valid_replicates'], 40)
        methods['trend']['test'] = [replace(r, score=None) for r in methods['trend']['test']]
        missing = compare_methods(methods, **args, bootstrap_repetitions=40)
        difference = missing['paired_differences']['comparisons'][0]['metrics']
        self.assertEqual(difference['event_sensitivity']['estimate'], 1.)
        self.assertEqual(difference['event_sensitivity']['percentile_95'], [1., 1.])
        self.assertEqual(difference['score_coverage']['estimate'], 1.)
        self.assertIsNone(difference['false_alarms_per_30_days']['estimate'])
        self.assertEqual(difference['false_alarms_per_30_days']['valid_replicates'], 0)
        methods['trend']['test'][0] = replace(methods['trend']['test'][0], label_available_at=issued + timedelta(days=16))
        with self.assertRaisesRegex(ValueError, 'identical'): compare_methods(methods, **args)


if __name__ == '__main__':
    unittest.main()
