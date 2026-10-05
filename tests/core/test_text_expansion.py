"""Sparse weak references, participant isolation, and trainable checkpoint continuation."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from rhythm_dnb.provenance import fingerprint
from rhythm_dnb.text.experiment import prepare_development, train_experiment
from rhythm_dnb.text.expansion import person_roles, template_identity, text_identity, MENTIONS
from rhythm_dnb.text.checkpoint import inspect_checkpoint, load_checkpoint
from core.test_text_experiment import experimental_fixture


class TextExpansionTests(unittest.TestCase):
    def test_identity_preserves_negation_and_person_visits(self):
        import re
        self.assertNotEqual(text_identity('我开心'), text_identity('我不开心'))
        self.assertEqual(template_identity('睡了7小时'), template_identity('睡了8小时'))
        self.assertFalse(re.search(MENTIONS['sleep_onset_difficulty'], '凌晨一点睡，七点醒。'))
        self.assertTrue(re.search(MENTIONS['sleep_onset_difficulty'], '昨晚并不难以入睡。'))
        rows = [{'participant_id': f'p{i}', 'source_dataset': 'DelSoM-baseline'} for i in range(20)]
        roles = person_roles(rows, 42)
        self.assertEqual(roles, person_roles(rows + [{'participant_id': 'p0', 'source_dataset': 'DelSoM-treatment'}], 42))
        self.assertEqual(set(roles.values()), {'train', 'validation', 'test'})

    def test_new_stage_keeps_weights_trainable_and_rejects_old_validation(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, config, exported = experimental_fixture(root)
            payload = json.loads((exported / 'development.json').read_text(encoding='utf-8'))
            original = train_experiment(payload, base, root / 'initial', config)
            previous, identity = inspect_checkpoint(original['best_checkpoint'], allow_experimental=True)
            new = copy.deepcopy(payload)
            for row in new['rows']:
                if row['split'] == 'validation':
                    row['example_id'] = row['group_id'] = 'new-' + row['example_id']
                    row['text'] += '新的独立场景'
                if len(row['scores']) > 1:
                    row['scores'][next(iter(row['scores']))] = None
            rows = sorted(new['rows'], key=lambda r: r['example_id'])
            ordered = [r for r in rows if r['split'] == 'train'] + [r for r in rows if r['split'] == 'validation']
            new['manifest'].update(id=fingerprint(rows), development_id=fingerprint(ordered),
                                   initial_checkpoint_id=identity, allow_missing=True)
            prepare_development(new)
            config = {**config, 'balance_categories': True}
            continued = train_experiment(new, base, root / 'continued', config,
                                         initialize_from=original['best_checkpoint'], save_resume_state=True)
            manifest, _ = inspect_checkpoint(continued['best_checkpoint'], allow_experimental=True)
            self.assertEqual(manifest['initialization']['checkpoint_id'], identity)
            self.assertTrue((root / 'continued' / 'initial-validation.json').exists())
            model, _, _, _ = load_checkpoint(original['best_checkpoint'], base_path=base,
                                             allow_experimental=True, trainable=True)
            self.assertTrue(any(p.requires_grad for n,p in model.encoder.named_parameters() if 'lora_' in n))
            bad = copy.deepcopy(new)
            bad['manifest']['initial_checkpoint_id'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'exact initial'):
                train_experiment(bad, base, root / 'bad', config, initialize_from=original['best_checkpoint'])
            bad = copy.deepcopy(payload)
            bad['manifest']['initial_checkpoint_id'] = identity
            with self.assertRaisesRegex(ValueError, 'overlaps previous'):
                train_experiment(bad, base, root / 'leak', config, initialize_from=original['best_checkpoint'])
