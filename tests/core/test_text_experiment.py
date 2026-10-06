"""Experimental provenance, held-out isolation and usable adapter reload without primary-policy bypass."""

import copy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from rhythm_dnb.text.experiment import export_corpus, prepare_development, train_experiment, evaluate_experiment, validate_holdout
from rhythm_dnb.text.checkpoint import inspect_checkpoint
from rhythm_dnb.text.predict import TextPredictor
from core.test_text_measurement import qwen_fixture


def experimental_fixture(root):
    # PSEUDOCODE: use a tiny software-only model and SQLite references to exercise the real export/training path.
    base, config, source = qwen_fixture(root)
    database = Path(root) / 'references.sqlite'
    with closing(sqlite3.connect(database)) as connection:
        connection.execute('CREATE TABLE records(record_id TEXT, dataset TEXT, subject_id TEXT)')
        connection.execute('CREATE TABLE chinese_examples(example_id TEXT,source_record_id TEXT,category TEXT,text TEXT,scores TEXT,origin TEXT,split TEXT,group_id TEXT)')
        for row in source:
            if row['split'] == 'calibration' or any(value is None for value in row['scores'].values()):
                continue
            origin = 'authored_simulation' if row['split'] == 'train' else 'translated_source'
            connection.execute('INSERT INTO chinese_examples VALUES(?,?,?,?,?,?,?,?)',
                (row['example_id'], None, row['category'], row['text'], json.dumps(row['scores']), origin, row['split'], row['group_id']))
        connection.commit()
    exported = Path(root) / 'corpus'
    export_corpus(database, exported)
    return base, config, exported


class TextExperimentTests(unittest.TestCase):
    def test_export_preserves_origins_and_development_rejects_test_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, exported = experimental_fixture(folder)
            development = json.loads((exported / 'development.json').read_text(encoding='utf-8'))
            heldout = json.loads((exported / 'test.json').read_text(encoding='utf-8'))
            partitions, manifest = prepare_development(development)
            self.assertEqual(manifest['origins']['authored_simulation'], 5)
            self.assertNotIn('participant_id', partitions['train'][0])
            self.assertFalse(manifest['independent_human_gold'])
            with self.assertRaises(ValueError):
                prepare_development({**development, 'rows': development['rows'] + heldout['rows']})
            changed = copy.deepcopy(development)
            key = next(iter(changed['rows'][0]['scores']))
            changed['rows'][0]['scores'][key] += 1
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                prepare_development(changed)

    def test_experimental_training_reload_and_sealed_test(self):
        with tempfile.TemporaryDirectory() as folder:
            base, config, exported = experimental_fixture(folder)
            development = json.loads((exported / 'development.json').read_text(encoding='utf-8'))
            heldout = json.loads((exported / 'test.json').read_text(encoding='utf-8'))
            result = train_experiment(development, base, Path(folder) / 'training', config)
            self.assertIsNone(result['test'])
            with self.assertRaises(ValueError):
                TextPredictor(result['best_checkpoint'], base_path=base)
            manifest, _ = inspect_checkpoint(result['best_checkpoint'], allow_experimental=True)
            tampered = copy.deepcopy(heldout)
            key = next(iter(tampered['rows'][0]['scores']))
            tampered['rows'][0]['scores'][key] += 1
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                validate_holdout(tampered, manifest)
            evaluated = evaluate_experiment(heldout, result['best_checkpoint'], base, Path(folder) / 'evaluation', device='cpu')
            self.assertEqual(evaluated['test']['records'], 5)
            self.assertEqual(len(evaluated['test']['per_metric']), 17)
            self.assertFalse(evaluated['test']['independent_human_gold'])
            with self.assertRaises(ValueError):
                TextPredictor(evaluated['best_checkpoint'], base_path=base)
            predictor = TextPredictor(evaluated['best_checkpoint'], base_path=base, allow_experimental=True)
            prediction = predictor.predict('压力', '今天压力很大')
            self.assertEqual(set(prediction['scores']), {'stress_intensity'})
            self.assertTrue(0 <= prediction['estimates']['stress_intensity'] <= 100)
            self.assertIsNone(prediction['scores']['stress_intensity'])
            self.assertFalse(prediction['evidence_calibrated'])
            self.assertFalse(prediction['eligible_for_primary_dnb'])
            self.assertNotIn('evidence', prediction)
            points = json.loads((Path(folder) / 'evaluation/test-predictions.json').read_text(encoding='utf-8'))
            self.assertNotIn('evidence', points['rows'][0])
