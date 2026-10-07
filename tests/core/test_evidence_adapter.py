"""Evidence-only optimization, provenance isolation and safe composition with an unchanged scorer."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from rhythm_dnb.provenance import fingerprint, file_hash
from rhythm_dnb.text.schema import CATEGORIES
from rhythm_dnb.text.evidence_adapter import prepare_evidence, train_evidence, evaluate_evidence, EvidenceAdapter, refine_evidence_thresholds


def evidence_fixture(parent, roles=('train', 'validation')):
    # PSEUDOCODE: construct software-test labels for every metric; never use these fixtures in research training.
    rows = []
    for role in roles:
        for category, (_, keys) in CATEGORIES.items():
            for supported in (False, True):
                identity = f'evidence-fixture:{role}:{category}:{supported}'
                rows.append({'example_id': identity, 'group_id': identity, 'family_id': identity, 'split': role,
                    'category': category, 'text': '软件测试 ' + identity,
                    'scores': {k: None for k in keys}, 'label_states': {k: 'supported_unscored' if supported else 'insufficient_evidence' for k in keys}})
    return {'rows': rows, 'id': fingerprint(rows), 'initial_checkpoint_id': parent, 'independent_human_gold': False}


class EvidenceAdapterTests(unittest.TestCase):
    def test_individually_balanced_heads_do_not_hide_missing_partial_evidence(self):
        from rhythm_dnb.text.review import evidence_support_coverage
        rows = evidence_fixture('parent')['rows']
        coverage = evidence_support_coverage(rows)
        self.assertEqual(coverage['partitions']['train']['sleep']['mixed_support_rows'], 0)
        self.assertTrue(any('train/sleep' in warning for warning in coverage['warnings']))
        partial = copy.deepcopy(next(r for r in rows if r['split'] == 'train' and r['category'] == 'sleep'))
        partial['label_states']['sleep_onset_difficulty'] = 'explicit_absence'
        partial['scores']['sleep_onset_difficulty'] = 0.
        partial['label_states']['post_sleep_fatigue'] = 'unreviewed'
        coverage = evidence_support_coverage([*rows, partial])
        sleep = coverage['partitions']['train']['sleep']
        self.assertEqual(sleep['mixed_support_rows'], 1)
        self.assertEqual(sleep['patterns']['010?'], 1)
        self.assertFalse(any('train/sleep' in warning for warning in coverage['warnings']))

    def test_protocol_rejects_scores_missing_classes_test_and_tampering(self):
        payload = evidence_fixture('parent')
        prepare_evidence(payload)
        for kind in ('score', 'class', 'test', 'unreviewed'):
            changed = copy.deepcopy(payload)
            row = changed['rows'][0]
            if kind == 'score':
                row['scores'] = {k: 50. for k in row['scores']}; row['label_states'] = {k: 'supported' for k in row['scores']}
            elif kind == 'class':
                row['label_states'] = {k: 'supported_unscored' for k in row['scores']}
            elif kind == 'test':
                row['split'] = 'test'
            else:
                row['label_states'] = {k: 'unreviewed' for k in row['scores']}
            changed['id'] = fingerprint(changed['rows'])
            with self.assertRaises(ValueError):
                prepare_evidence(changed)
        payload['rows'][0]['text'] += 'changed'
        with self.assertRaisesRegex(ValueError, 'identity'):
            prepare_evidence(payload)

    def test_separate_training_reload_evaluation_and_composition(self):
        from core.test_text_experiment import experimental_fixture
        from rhythm_dnb.text.experiment import train_experiment
        from rhythm_dnb.text.checkpoint import inspect_checkpoint, load_checkpoint
        from rhythm_dnb.text.predict import TextPredictor
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, config, exported = experimental_fixture(root)
            main = train_experiment(json.loads((exported/'development.json').read_text(encoding='utf-8')), base, root/'main', config)
            parent = main['best_checkpoint']; _, identity = inspect_checkpoint(parent, allow_experimental=True)
            before = {p.relative_to(parent).as_posix(): file_hash(p) for p in Path(parent).rglob('*') if p.is_file()}
            setup = {**config, 'epochs': 2, 'train_experimental_evidence': True, 'evidence_loss_weight': 1., 'scope_loss_weight': 0.}
            result = train_evidence(evidence_fixture(identity), parent, base, root/'evidence', setup)
            selected = result['best_checkpoint']
            with self.assertRaises(ValueError):
                TextPredictor(selected, base_path=base, allow_experimental=True)
            with self.assertRaisesRegex(ValueError, 'exact scoring'):
                EvidenceAdapter(selected, 'wrong', base, 'cpu')
            old, _, _, _ = load_checkpoint(parent, base_path=base, allow_experimental=True)
            new, _, manifest, _ = load_checkpoint(selected, base_path=base, allow_evidence=True)
            self.assertTrue(all(torch.equal(v, new.heads.state_dict()[k]) for k, v in old.heads.state_dict().items()))
            self.assertTrue(any(not torch.equal(v, new.encoder.state_dict()[k]) for k, v in old.encoder.state_dict().items()))
            self.assertFalse(manifest['numeric_outputs_from_this_adapter_valid'])
            refined = refine_evidence_thresholds(evidence_fixture(identity), selected, base, root/'refined', device='cpu')
            self.assertFalse(refined['weights_changed'])
            self.assertFalse(refined['test_used'])
            newer, _ = inspect_checkpoint(refined['checkpoint'], allow_evidence=True)
            self.assertEqual(manifest['files'], newer['files'])
            self.assertEqual(manifest['validation']['false_acceptances'], newer['validation']['false_acceptances'])
            plain = TextPredictor(parent, base_path=base, device='cpu', allow_experimental=True)
            combined = TextPredictor(parent, base_path=base, device='cpu', allow_experimental=True, evidence_checkpoint=selected)
            first = plain.predict('stress', '我最近很焦虑')
            second = combined.predict('stress', '我最近很焦虑')
            self.assertEqual(first['estimates'], second['estimates'])
            self.assertIsNone(second['scores']['stress_intensity'])
            self.assertFalse(second['acceptance_certified'])
            self.assertTrue(second['evidence_trained'])
            self.assertEqual(before, {p.relative_to(parent).as_posix(): file_hash(p) for p in Path(parent).rglob('*') if p.is_file()})
            heldout = evidence_fixture(identity, ('test',))
            evaluated = evaluate_evidence(heldout, selected, base, root/'evaluation', device='cpu')
            self.assertEqual(len(evaluated['heads']), 17)
            self.assertFalse(evaluated['independent_human_gold'])
            overlap = evidence_fixture(identity)
            for row in overlap['rows']:
                row['split'] = 'test'
            overlap['id'] = fingerprint(overlap['rows'])
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                evaluate_evidence(overlap, selected, base, root/'invalid', device='cpu', regression_only=True)
            inherited = copy.deepcopy(heldout)
            inherited['rows'][0]['text'] = json.loads((exported/'development.json').read_text(encoding='utf-8'))['rows'][0]['text']
            inherited['id'] = fingerprint(inherited['rows'])
            with self.assertRaisesRegex(ValueError, 'text was used'):
                evaluate_evidence(inherited, selected, base, root/'ancestor-leak', device='cpu')
