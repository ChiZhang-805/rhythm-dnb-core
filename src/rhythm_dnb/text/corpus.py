"""Freeze semantic training rows and leakage groups independently from rhythm outcomes."""

import math
from ..provenance import fingerprint
from ..io.splits import validate_splits
from .schema import CATEGORIES, validate_input


def prepare_corpus(rows, *, allow_constructed_training=False, require_all_categories=True):
    # PSEUDOCODE: validate semantic labels -> enforce human review and split groups -> fingerprint exact rows.
    rows = [dict(r) for r in rows]
    ids, identities = set(), {}
    for row in rows:
        category, text = validate_input(row['category'], row['text'])
        row.update(category=category, text=text)
        if not row.get('example_id') or row['example_id'] in ids or not row.get('group_id'):
            raise ValueError('Unique example IDs and leakage groups are required.')
        ids.add(row['example_id'])
        if row.get('split') not in ('train', 'validation', 'test'):
            raise ValueError('Unknown corpus split.')
        if set(row['scores']) != set(CATEGORIES[category][1]) or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100 for v in row['scores'].values()):
            raise ValueError('Semantic scores must cover exactly the category heads on 0-100.')
        if row.get('review_status') != 'accepted' or not row.get('annotation_evidence_id'):
            raise ValueError('Unreviewed semantic reference labels.')
        real = row.get('origin') in ('observed', 'translated_observed')
        if not real and (row['split'] != 'train' or not allow_constructed_training):
            raise ValueError('Constructed text is excluded from primary training/evaluation.')
        identity = fingerprint(''.join(text.split()))
        if identity in identities and identities[identity] != row['split']:
            raise ValueError('Duplicate text crosses corpus splits.')
        identities[identity] = row['split']
    validate_splits(rows)
    partitions = {split: [r for r in rows if r['split'] == split] for split in ('train', 'validation', 'test')}
    for split, items in partitions.items():
        if not items or require_all_categories and {r['category'] for r in items} != set(CATEGORIES):
            raise ValueError('Empty or category-incomplete partition: ' + split)
    manifest = {'id': fingerprint(sorted(rows, key=lambda r: r['example_id'])),
                'counts': {k: len(v) for k, v in partitions.items()},
                'allow_constructed_training': allow_constructed_training,
                'partitions': {k: [r['example_id'] for r in v] for k, v in partitions.items()}}
    return partitions, manifest
