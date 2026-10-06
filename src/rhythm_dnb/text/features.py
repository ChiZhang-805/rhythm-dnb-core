"""Cache immutable checkpoint representations for independent evidence learning without score drift."""

import json
from pathlib import Path
import numpy as np
from ..provenance import canonical_json, fingerprint, file_hash
from .schema import CATEGORIES, METRICS, validate_input
from .labels import validate_labels


def validate_feature_rows(rows):
    # PSEUDOCODE: require unique explicit identities and labels; retain semantic scope and source grouping.
    from ..io.splits import validate_splits
    seen, text_roles = set(), {}
    if not rows:
        raise ValueError('Feature rows cannot be empty.')
    for row in rows:
        category, text = validate_input(row['category'], row['text'])
        if text != row['text'] or category != row['category'] or not row.get('example_id') or row['example_id'] in seen:
            raise ValueError('Feature rows require normalized text and unique explicit identities.')
        seen.add(row['example_id'])
        text_key = ''.join(text.split())
        if text_key in text_roles and text_roles[text_key] != row.get('split'):
            raise ValueError('Cross-split leakage: duplicate text')
        text_roles[text_key] = row.get('split')
        if not row.get('group_id') or row.get('split') not in ('train', 'validation', 'calibration', 'test'):
            raise ValueError('Feature rows require leakage groups and a declared split.')
        if not isinstance(row.get('label_states'), dict) or set(row['scores']) != set(CATEGORIES[category][1]):
            raise ValueError('Evidence targets must be explicitly annotated for every category metric.')
        validate_labels(row)
    validate_splits(rows, group_keys=('group_id', 'participant_id', 'family_id'))


def extract_features(rows, checkpoint, base, output_dir, *, device='cuda', batch_size=32):
    # PSEUDOCODE: verify a fixed checkpoint -> encode each original text once -> save aligned vectors, predictions and hashes.
    import torch
    from torch.utils.data import DataLoader
    from .checkpoint import load_checkpoint
    from .dataset import ScoreDataset, Collator
    from .train import _inputs
    validate_feature_rows(rows)
    if type(batch_size) is not int or batch_size < 1 or not rows:
        raise ValueError('Feature extraction needs nonempty rows and an explicit positive batch size.')
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    model, tokenizer, manifest, identity = load_checkpoint(checkpoint, base_path=base, device=device, allow_experimental=True)
    model.eval(); model.requires_grad_(False)
    target = next(model.heads.parameters()).device
    loader = DataLoader(ScoreDataset(rows, tokenizer, manifest['config']['max_length']), batch_size=batch_size,
                        shuffle=False, collate_fn=Collator(tokenizer), num_workers=0)
    vectors, scores, prior_evidence = [], [], []
    with torch.inference_mode():
        for batch in loader:
            pooled, _ = model.representation(**_inputs(batch, target))
            vectors.append(pooled.cpu().numpy())
            for i, category in enumerate(batch['categories']):
                indices = [j for j, metric in enumerate(METRICS) if metric[1] == category]
                score, evidence = np.full(len(METRICS), np.nan), np.full(len(METRICS), np.nan)
                score[indices] = (torch.sigmoid(model.heads[category](pooled[i])) * 100).cpu().numpy()
                evidence[indices] = torch.sigmoid(model.evidence_heads[category](pooled[i])).cpu().numpy()
                scores.append(score); prior_evidence.append(evidence)
    matrix = np.concatenate(vectors)
    if not np.isfinite(matrix).all():
        raise ValueError('Frozen encoder produced nonfinite features.')
    np.savez_compressed(output/'features.npz', vectors=matrix, scores=np.asarray(scores), prior_evidence=np.asarray(prior_evidence))
    (output/'rows.json').write_text(canonical_json(rows), encoding='utf-8')
    receipt = {'checkpoint_id': identity, 'rows_id': fingerprint(rows), 'rows': len(rows), 'width': matrix.shape[1],
        'representation': 'unchanged_checkpoint_full_text_pooled_float32', 'score_model_modified': False,
        'files': {name: file_hash(output/name) for name in ('features.npz', 'rows.json')}}
    (output/'manifest.json').write_text(canonical_json(receipt), encoding='utf-8')
    return receipt


def load_features(directory):
    # PSEUDOCODE: reject modified vectors or row order before fitting or evaluating a separate evidence head.
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if set(manifest['files']) != {'features.npz', 'rows.json'}:
        raise ValueError('Unexpected feature-cache file inventory.')
    for name, digest in manifest['files'].items():
        if file_hash(directory/name) != digest:
            raise ValueError('Feature cache changed: ' + name)
    rows = json.loads((directory/'rows.json').read_text(encoding='utf-8'))
    validate_feature_rows(rows)
    if fingerprint(rows) != manifest['rows_id'] or len(rows) != manifest['rows']:
        raise ValueError('Feature rows differ from their sealed cache.')
    with np.load(directory/'features.npz', allow_pickle=False) as arrays:
        vectors, scores, prior = (arrays[k].copy() for k in ('vectors', 'scores', 'prior_evidence'))
    if vectors.shape != (len(rows), manifest['width']) or scores.shape != (len(rows), len(METRICS)) or prior.shape != scores.shape or not np.isfinite(vectors).all():
        raise ValueError('Feature-cache dimensions or values differ from the manifest.')
    for i, row in enumerate(rows):
        selected = np.asarray([metric[1] == row['category'] for metric in METRICS])
        if (not np.isfinite(scores[i, selected]).all() or not np.isfinite(prior[i, selected]).all()
            or np.any((scores[i, selected] < 0) | (scores[i, selected] > 100))
            or np.any((prior[i, selected] < 0) | (prior[i, selected] > 1))
            or not np.isnan(scores[i, ~selected]).all() or not np.isnan(prior[i, ~selected]).all()):
            raise ValueError('Cached category predictions are invalid or misaligned.')
    return rows, vectors, scores, prior, manifest
