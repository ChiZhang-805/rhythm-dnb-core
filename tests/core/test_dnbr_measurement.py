"""Check that reference-score diagnostics cannot silently change text, outcomes or their source."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from rhythm_dnb.provenance import fingerprint
from rhythm_dnb.research.experiment_data import TEXT
from tools.run_dnbr_measurement import attach_references, verify_source_row


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        # PSEUDOCODE: prepare one explicitly authored example with a future outcome and provenance.
        self.row = {'record_id': 'r', 'participant_id': 'p', 'split': 'train',
                    'observed_at': '2026-01-01T00:00:00+00:00', 'texts': {'emotion': '今天心情平静。'},
                    'features': {'sleep_duration_h': 8, **{k: 21 for k in TEXT}},
                    'outcome': {'event_onset_at': None}}
        self.source = {k: self.row[k] for k in ('record_id', 'participant_id', 'split', 'observed_at')}
        self.source.update(source_dataset='SIMULATED-RHYTHM',
                           label_source='synthetic_scenario_not_clinical_ground_truth', event_onset_at=None,
                           raw=json.dumps({'emotion_description_text': self.row['texts']['emotion']}),
                           provenance=json.dumps({'provenance': {k: {'method': 'manual_contextual_rewrite'} for k in TEXT}}))
        self.source.update({k: 20 for k in TEXT})

    def test_identity_text_outcome_and_provenance_are_bound(self):
        # PSEUDOCODE: correct references pass; different person, text, future outcome or provenance must fail.
        self.assertEqual(verify_source_row(self.row, self.source), {k: 20 for k in TEXT})
        changes = {'participant_id': 'other', 'raw': json.dumps({'emotion_description_text': '换了文本'}),
                   'event_onset_at': '2026-01-08', 'provenance': json.dumps({'provenance': {}})}
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_source_row(self.row, {**self.source, key: value})

    def test_counterfactual_changes_only_text_measurements(self):
        # PSEUDOCODE: apply a verified export on a copy; refuse even rehashed exports with extra or missing records.
        data = {'reference': [], 'train': [deepcopy(self.row)], 'validation': [], 'test': []}
        original = deepcopy(data)
        manifest = {'id': 'packet', 'source_sha256': 'source', 'domain': 'authored'}
        payload = {'packet_id': 'packet', 'source_sha256': 'source', 'domain': 'authored',
                   'independent_human_gold': False,
                   'purpose': 'authored_reference_counterfactual_not_deployable_predictions',
                   'scores': {'r': {k: 20 for k in TEXT}}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'reference.json'
            path.write_text(json.dumps({**payload, 'id': fingerprint(payload)}), encoding='utf-8')
            result, _ = attach_references(data, manifest, path)
            self.assertEqual(data, original)
            self.assertEqual(result['train'][0]['outcome'], self.row['outcome'])
            self.assertEqual(result['train'][0]['features']['sleep_duration_h'], 8)
            self.assertTrue(all(result['train'][0]['features'][k] == 20 for k in TEXT))
            payload['scores']['extra'] = payload['scores']['r']
            path.write_text(json.dumps({**payload, 'id': fingerprint(payload)}), encoding='utf-8')
            with self.assertRaises(ValueError):
                attach_references(data, manifest, path)


if __name__ == '__main__':
    unittest.main()
