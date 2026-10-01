"""Freeze semantic training rows and leakage groups independently from rhythm outcomes."""

import math
from ..provenance import fingerprint
from ..io.splits import validate_splits
from .schema import CATEGORIES, validate_input


def prepare_corpus(rows, *, require_all_categories=True, require_calibration=False, development_only=False):
    # PSEUDOCODE: validate semantic labels -> enforce human review and split groups -> fingerprint exact rows.
    rows = [dict(r) for r in rows]
    ids, identities = set(), {}
    for row in rows:
        category, text = validate_input(row['category'], row['text'])
        row.update(category=category, text=text)
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in ('example_id', 'group_id')) or row['example_id'] in ids:
            raise ValueError('Unique example IDs and leakage groups are required.')
        if not isinstance(row.get('participant_id'), str) or not row['participant_id'].strip():
            raise ValueError('Every corpus row needs a participant identity for independent evaluation.')
        ids.add(row['example_id'])
        if row.get('split') not in ('train', 'validation', 'calibration', 'test'):
            raise ValueError('Unknown corpus split.')
        if development_only and row['split'] not in ('train', 'validation'):
            raise ValueError('Development-only input must exclude calibration and test rows.')
        if not isinstance(row.get('scores'), dict) or set(row['scores']) != set(CATEGORIES[category][1]) or any(v is not None and (type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100) for v in row['scores'].values()):
            raise ValueError('Semantic scores must cover the category heads with 0-100 or null for insufficient evidence.')
        if row.get('review_status') != 'accepted' or not isinstance(row.get('annotation_evidence_id'), str) or not row['annotation_evidence_id'].strip():
            raise ValueError('Unreviewed semantic reference labels.')
        if row.get('observed_at') is not None:
            from ..timebase import instant
            row['observed_at'] = instant(row['observed_at']).isoformat()
        real = row.get('origin') in ('observed', 'translated_observed')
        if not real:
            raise ValueError('Constructed text is excluded from primary training/evaluation.')
        identity = fingerprint(''.join(text.split()))
        if identity in identities and identities[identity] != row['split']:
            raise ValueError('Duplicate text crosses corpus splits.')
        identities[identity] = row['split']
    validate_splits(rows)
    rows.sort(key=lambda row: row['example_id'])
    roles = ('train', 'validation') if development_only else (('train', 'validation', 'calibration', 'test') if require_calibration or any(r['split'] == 'calibration' for r in rows) else ('train', 'validation', 'test'))
    partitions = {split: [r for r in rows if r['split'] == split] for split in roles}
    for split, items in partitions.items():
        if not items or require_all_categories and {r['category'] for r in items} != set(CATEGORIES):
            raise ValueError('Empty or category-incomplete partition: ' + split)
        if require_all_categories:
            for category, (_, keys) in CATEGORIES.items():
                for key in keys:
                    present = [r['scores'][key] is not None for r in items if r['category'] == category]
                    if not any(present) or require_calibration and split == 'train' and all(present):
                        raise ValueError(f'{split}:{key} requires observed scores and training examples of insufficient evidence.')
    manifest = {'id': fingerprint(sorted(rows, key=lambda r: r['example_id'])),
                'counts': {k: len(v) for k, v in partitions.items()},
                'people': {k: sorted({r['participant_id'] for r in v}) for k, v in partitions.items()},
                'partitions': {k: [r['example_id'] for r in v] for k, v in partitions.items()}}
    return partitions, manifest
