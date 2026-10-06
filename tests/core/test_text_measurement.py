"""Decoder padding, masked labels, independent evidence calibration and complete quantification."""

from datetime import date, datetime, timezone, timedelta
import json
from pathlib import Path
import importlib.util
import tempfile
import unittest
import numpy as np
import torch
from rhythm_dnb.text.schema import CATEGORIES, METRICS
from rhythm_dnb.text.model import ScoringModel, regression_loss
from rhythm_dnb.text.evidence import calibrate_evidence, qualified_scores
from rhythm_dnb.text.corpus import prepare_corpus
from rhythm_dnb.provenance import file_hash
from rhythm_dnb.text.config import validate_config


def qwen_fixture(root):
    # PSEUDOCODE: create tiny local Qwen and a non-clinical contract fixture without downloading any weights.
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
    root = Path(root); base = root / 'base'; base.mkdir()
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator(['今天心情焦虑压力睡眠饮食社交 abcdefghijklmnopqrstuvwxyz'],
        trainers.BpeTrainer(vocab_size=300, special_tokens=['[UNK]', '[PAD]', '[EOS]'],
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
    tokenizer.model_input_names = ['input_ids', 'attention_mask']
    tokenizer.save_pretrained(base)
    torch.manual_seed(42)
    encoder = Qwen3Model(Qwen3Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=1024, attention_dropout=0.))
    encoder.save_pretrained(base)
    config = validate_config({'model_id': 'Qwen/Qwen3-8B', 'revision': 'b' * 40, 'seed': 42,
        'max_length': 512, 'batch_size': 3, 'gradient_accumulation': 2, 'epochs': 1, 'patience': 1,
        'encoder_lr': .001, 'head_lr': .001, 'weight_decay': .01, 'warmup_ratio': 0.,
        'huber_delta': .1, 'dropout': 0., 'max_grad_norm': 1., 'threads': 1, 'device': 'cpu',
        'precision': 'fp32', 'lora_rank': 2, 'lora_alpha': 4})
    receipt = {'model_id': config['model_id'], 'revision': config['revision'],
               'files': {p.name: file_hash(p) for p in base.iterdir() if p.is_file()}}
    (base / 'download.json').write_text(json.dumps(receipt), encoding='utf-8')
    rows = []
    for split in ('train', 'validation', 'calibration', 'test'):
        for category, (_, keys) in CATEGORIES.items():
            for known in (True, False):
                identity = split + category + str(known)
                rows.append({'example_id': identity, 'participant_id': identity, 'group_id': identity,
                    'split': split, 'category': category, 'text': '今天的记录' + identity,
                    'scores': {key: float(20 + index * 10) if known else None for index, key in enumerate(keys)},
                    'label_states': {key: 'supported' if known else 'insufficient_evidence' for key in keys},
                    'origin': 'observed', 'review_status': 'accepted', 'annotation_evidence_id': 'software-test-only'})
    return base, config, rows


class TextMeasurementTests(unittest.TestCase):
    def test_qwen_lora_training_reload_and_padding_invariance(self):
        from rhythm_dnb.text.train import train
        from rhythm_dnb.text.checkpoint import load_checkpoint
        from rhythm_dnb.text.dataset import encode
        with tempfile.TemporaryDirectory() as folder:
            base, config, rows = qwen_fixture(folder)
            result = train(rows, base, Path(folder) / 'run', config)
            model, tokenizer, manifest, _ = load_checkpoint(result['best_checkpoint'], base_path=base)
            self.assertEqual(manifest['storage'], 'adapter')
            self.assertEqual(result['test']['records'], 10)
            self.assertEqual(result['test']['per_metric']['stress_intensity']['n'], 1)
            self.assertEqual(set(manifest['dataset']['people']), {'train', 'validation', 'calibration', 'test'})
            from rhythm_dnb.text.plots import plot_evaluation
            points = json.loads((Path(folder) / 'run/test-predictions.json').read_text(encoding='utf-8'))
            plotted = plot_evaluation(result, points, Path(folder) / 'figures')
            self.assertEqual(len(plotted['figures']), 5)
            self.assertTrue(all(Path(p).stat().st_size > 0 for p in plotted['figures']))
            model.eval()
            sequences = [encode(tokenizer, 'stress', s, 512) for s in ('今天压力很大', '今天没有什么压力但是有点疲惫')]
            with torch.inference_mode():
                separate = [model(**tokenizer.pad([s], return_tensors='pt'))['stress'][0] for s in sequences]
                together = model(**tokenizer.pad(sequences, return_tensors='pt'))['stress']
                torch.testing.assert_close(together, torch.stack(separate), atol=1e-5, rtol=1e-5)
                tokenizer.padding_side = 'left'
                left = model(**tokenizer.pad(sequences, return_tensors='pt'))['stress']
                torch.testing.assert_close(left, together, atol=1e-5, rtol=1e-5)
            with self.assertRaisesRegex(ValueError, 'exact verified base'):
                load_checkpoint(result['best_checkpoint'])
            (base / 'config.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_checkpoint(result['best_checkpoint'], base_path=base)

    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec('bitsandbytes'), 'NF4 needs CUDA and optional bitsandbytes')
    def test_nf4_adapter_training_reload_and_frozen_base(self):
        from rhythm_dnb.text.train import train
        from rhythm_dnb.text.checkpoint import load_checkpoint
        from rhythm_dnb.text.dataset import encode
        with tempfile.TemporaryDirectory() as folder:
            base, config, rows = qwen_fixture(folder)
            config.update(device='cuda', precision='auto', quantization='nf4')
            result = train(rows, base, Path(folder) / 'run', config)
            self.assertGreater(result['history'][0]['optimizer_steps'], 0)
            model, tokenizer, _, _ = load_checkpoint(result['best_checkpoint'], base_path=base, device='cuda')
            self.assertTrue(model.encoder.is_loaded_in_4bit)
            self.assertEqual(model.encoder.get_input_embeddings().weight.dtype, torch.float32)
            self.assertTrue(any('lora_B' in n and torch.count_nonzero(p).item() for n, p in model.encoder.named_parameters()))
            self.assertFalse(any(p.requires_grad for n, p in model.encoder.named_parameters() if 'lora_' not in n))
            model.eval()
            encoded = encode(tokenizer, 'stress', '今天压力很大', 512)
            with torch.inference_mode():
                value = model(**{k: torch.tensor([v], device='cuda') for k, v in encoded.items()})['stress']
            self.assertTrue(torch.isfinite(value).all())

    def test_unknown_intensity_is_masked_but_evidence_learns(self):
        outputs = {c: torch.full((1, len(keys)), .4, requires_grad=True) for c, (_, keys) in CATEGORIES.items()}
        outputs['_evidence'] = {c: torch.zeros((1, len(keys)), requires_grad=True) for c, (_, keys) in CATEGORIES.items()}
        loss = regression_loss(outputs, torch.full((1, len(METRICS)), float('nan')), ['stress'],
                               evidence_labels=torch.zeros((1, len(METRICS))))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(outputs['stress'].grad.item(), 0)
        self.assertGreater(outputs['_evidence']['stress'].grad.item(), 0)

    def test_evidence_cutoff_comes_from_separate_data_and_can_fail(self):
        key = 'stress_intensity'
        rows = [{'example_id': str(i), 'participant_id': str(i), 'scores': {key: score},
                 'label_states': {key: 'supported' if score is not None else 'insufficient_evidence'}}
                for i, score in enumerate((None, None, 30., 70.))]
        predictions = [{'evidence': {key: value}} for value in (.1, .6, .7, .9)]
        calibration = [{'example_id': 'c' + str(i), 'participant_id': 'c' + str(i), 'scores': {key: 50.}} for i in range(114)]
        calibration_predictions = [{'evidence': {key: .8}} for _ in calibration]
        options = dict(selection_rows=rows, selection_predictions=predictions, target_precision=.95)
        policy = calibrate_evidence(calibration, calibration_predictions, **options)
        self.assertEqual(policy['minimum_accepted_if_all_correct'], 114)
        self.assertEqual(policy['thresholds'][key]['threshold'], .7)
        self.assertGreaterEqual(policy['thresholds'][key]['precision_lower'], .95)
        accepted, _ = qualified_scores({key: 0.}, {key: .8}, policy)
        self.assertEqual(accepted[key], 0.)
        rejected, reason = qualified_scores({key: 80.}, {key: .65}, policy)
        self.assertIsNone(rejected[key])
        self.assertEqual(reason[key], 'insufficient_text_evidence')
        failed = calibrate_evidence(calibration[:20], calibration_predictions[:20], **options)
        self.assertIsNone(failed['thresholds'][key]['threshold'])
        self.assertEqual(failed['thresholds'][key]['candidate_threshold'], .7)
        # Calibration failures cannot be repaired by searching a new cutoff on the same people.
        calibration[0]['scores'][key] = None
        calibration[0]['label_states'] = {key: 'insufficient_evidence'}
        failed = calibrate_evidence(calibration, calibration_predictions, **options)
        self.assertIsNone(failed['thresholds'][key]['threshold'])
        self.assertEqual(failed['thresholds'][key]['candidate_threshold'], .7)
        for row in calibration:
            row['participant_id'] = 'one-person'
        repeated = calibrate_evidence(calibration, calibration_predictions, **options)
        self.assertEqual(repeated['thresholds'][key]['n'], 1)
        self.assertIsNone(repeated['thresholds'][key]['threshold'])
        calibration[0]['participant_id'] = rows[0]['participant_id']
        with self.assertRaisesRegex(ValueError, 'independent'):
            calibrate_evidence(calibration, calibration_predictions, **options)

    def test_main_corpus_requires_independent_calibration_and_missing_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, rows = qwen_fixture(folder)
            partitions, _ = prepare_corpus(rows, require_calibration=True)
            self.assertEqual(set(partitions), {'train', 'validation', 'calibration', 'test'})
            rows[-1]['participant_id'] = rows[0]['participant_id']
            with self.assertRaisesRegex(ValueError, 'Cross-split'):
                prepare_corpus(rows, require_calibration=True)

    def test_evidence_search_preserves_ties_and_nonmonotone_precision(self):
        key = 'stress_intensity'
        rng = np.random.default_rng(914)
        examples = [[(.9, True), (.8, False), (.7, True), (.7, True), (.5, False)]]
        examples += [list(zip(rng.integers(0, 11, 80) / 10, rng.integers(0, 2, 80).astype(bool))) for _ in range(25)]
        for pairs in examples:
            rows = [{'example_id': str(i), 'participant_id': str(i), 'scores': {key: 50. if known else None},
                     'label_states': {key: 'supported' if known else 'insufficient_evidence'}}
                    for i, (_, known) in enumerate(pairs)]
            predictions = [{'evidence': {key: float(value)}} for value, _ in pairs]
            expected = None
            for threshold in sorted({value for value, _ in pairs}):
                accepted = [known for value, known in pairs if value >= threshold]
                if sum(accepted) / len(accepted) >= .75:
                    expected = threshold
                    break
            policy = calibrate_evidence([], [], selection_rows=rows, selection_predictions=predictions, target_precision=.75)
            self.assertEqual(policy['thresholds'][key]['candidate_threshold'], expected)

    def test_preflight_checks_real_contract_and_tokenizer_without_model_load(self):
        from unittest.mock import patch
        from rhythm_dnb.text.preflight import check_training_inputs
        from rhythm_dnb.text.dataset import ScoreDataset
        from rhythm_dnb.text.checkpoint import load_tokenizer
        with tempfile.TemporaryDirectory() as folder:
            base, config, rows = qwen_fixture(folder)
            with patch.object(ScoringModel, 'pretrained', side_effect=AssertionError('Preflight loaded a model')) as load:
                report = check_training_inputs(rows, base, config)
                self.assertEqual(report['tokens']['train']['records'], 10)
                self.assertEqual(report['tokens']['test']['people'], 10)
                self.assertLessEqual(report['tokens']['train']['max'], config['max_length'])
                self.assertFalse(report['gpu_memory_fit_verified'])
                self.assertFalse(report['model_loaded'])
                with self.assertRaisesRegex(ValueError, 'exclude calibration and test'):
                    check_training_inputs(rows, base, config, development_only=True)
                development = [r for r in rows if r['split'] in ('train', 'validation')]
                self.assertEqual(set(check_training_inputs(development, base, config, development_only=True)['tokens']), {'train', 'validation'})
                from rhythm_dnb.cli import main
                from contextlib import redirect_stdout
                from io import StringIO
                root = Path(folder)
                (root / 'corpus.json').write_text(json.dumps(development), encoding='utf-8')
                (root / 'config.json').write_text(json.dumps(config), encoding='utf-8')
                with redirect_stdout(StringIO()):
                    main(['check-text', '--corpus', str(root / 'corpus.json'), '--base', str(base),
                          '--config', str(root / 'config.json'), '--development-only', '--output', str(root / 'check.json')])
                self.assertEqual(json.loads((root / 'check.json').read_text(encoding='utf-8'))['status'], 'inputs_validated')
                load.assert_not_called()
            rows[0]['text'] = '超长真实输入检查' * 512
            with self.assertRaisesRegex(ValueError, rows[0]['example_id']):
                check_training_inputs(rows, base, config)
            with self.assertRaisesRegex(ValueError, rows[0]['example_id']):
                ScoreDataset([rows[0]], load_tokenizer(base), config['max_length'])

    def test_development_mode_cannot_consume_calibration_or_test(self):
        from rhythm_dnb.text.train import train
        with tempfile.TemporaryDirectory() as folder:
            base, config, rows = qwen_fixture(folder)
            with self.assertRaisesRegex(ValueError, 'exclude calibration and test'):
                train(rows, base, Path(folder) / 'invalid', config, development_only=True)
            development = [r for r in rows if r['split'] in ('train', 'validation')]
            result = train(development, base, Path(folder) / 'development', config, development_only=True)
            self.assertIsNone(result['test'])
            self.assertEqual(set(result['corpus']['counts']), {'train', 'validation'})
            self.assertFalse((Path(folder) / 'development' / 'test-predictions.json').exists())

    def test_every_text_category_reaches_quantification(self):
        from rhythm_dnb.contracts import Observation, Provenance
        from rhythm_dnb.workflows.prepare import quantify_day, prepare_day
        from rhythm_dnb.measures.panel import ALL_MEASURES
        class Predictor:
            def predict(self, category, text):
                return {'scores': {k: 23. for k in CATEGORIES[category][1]}, 'model_identity': 'test-only'}
        t = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
        observations = [Observation(c, 'p', 'text:' + c, '真实记录', 'text', t, t, t, 'UTC', Provenance('observed', c, 'hash'))
                        for c in CATEGORIES]
        measured = quantify_day(observations, 'p', t.date(), 'UTC', t + timedelta(days=1), text_predictor=Predictor())
        self.assertEqual(tuple(f.name for f in measured.features), ALL_MEASURES)
        self.assertEqual(sum(f.value == 23. for f in measured.features), 17)
        panel = prepare_day(observations, 'p', t.date(), 'UTC', t + timedelta(days=1), text_predictor=Predictor())
        self.assertEqual(len(panel.features), 12)

    def test_point_at_boundary_is_counted_only_in_next_day(self):
        from rhythm_dnb.contracts import Observation, Provenance
        from rhythm_dnb.workflows.prepare import quantify_day
        t = datetime(2025, 1, 2, 4, tzinfo=timezone.utc)
        observation = Observation('meal', 'p', 'caloric_event', 10., 'kcal', t, t, t, 'UTC', Provenance('observed', 'meal', 'hash'))
        before = quantify_day([observation], 'p', date(2025, 1, 1), 'UTC', t + timedelta(hours=8), eating_complete=True)
        after = quantify_day([observation], 'p', date(2025, 1, 2), 'UTC', t + timedelta(days=1, hours=8), eating_complete=True)
        self.assertIsNone(next(f.value for f in before.features if f.name == 'first_caloric_h'))
        self.assertEqual(next(f.value for f in after.features if f.name == 'first_caloric_h'), 4.)

    def test_low_mae_does_not_hide_constant_predictions(self):
        from rhythm_dnb.text.evaluate import errors
        report = errors([45., 50., 55.], [50., 50., 50.])
        self.assertEqual(report['sd_ratio'], 0.)
        self.assertIsNone(report['pearson'])
