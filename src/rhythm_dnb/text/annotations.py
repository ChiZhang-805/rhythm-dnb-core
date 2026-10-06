"""Blind annotation comparison; disagreement is reviewed rather than auto-averaged."""

import numpy as np
from .schema import CATEGORIES, METRICS, validate_input
from ..provenance import fingerprint


def compare_annotations(first, second, *, tolerance=10):
    # PSEUDOCODE: verify same item/different raters -> compare every head -> request adjudication for disagreement.
    if type(tolerance) not in (int, float) or not np.isfinite(tolerance) or not 0 <= tolerance <= 100 or not first.get('rater_id') or not second.get('rater_id'):
        raise ValueError('Invalid adjudication tolerance or rater identity.')
    if first['example_id'] != second['example_id'] or first['category'] != second['category'] or first['rater_id'] == second['rater_id']:
        raise ValueError('Two independent raters of the same example are required.')
    keys = CATEGORIES[first['category']][1]
    if set(first['scores']) != set(keys) or set(second['scores']) != set(keys):
        raise ValueError('Incomplete annotation heads.')
    raw = [r['scores'][k] for r in (first, second) for k in keys]
    if any(v is not None and (type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 100) for v in raw):
        raise ValueError('Annotation score outside 0-100.')
    differences = {k: abs(first['scores'][k] - second['scores'][k]) if first['scores'][k] is not None and second['scores'][k] is not None else None for k in keys}
    evidence_disagreement = [k for k in keys if (first['scores'][k] is None) != (second['scores'][k] is None)]
    comparable = [v for v in differences.values() if v is not None]
    return {'example_id': first['example_id'], 'evidence_id': fingerprint([first, second]),
            'mean_absolute_disagreement': float(np.mean(comparable)) if comparable else None,
            'adjudication_required': bool(evidence_disagreement or any(v > tolerance for v in comparable)),
            'evidence_disagreement': evidence_disagreement, 'per_head': differences}


def validate_scope_annotations(records):
    # PSEUDOCODE: verify quoted evidence offsets and family splits without pretending to verify their meaning.
    if not records:
        raise ValueError('Structured scope annotations cannot be empty.')
    identities, families, texts = {}, {}, {}
    for row in records:
        category, text = validate_input(row['category'], row['text'])
        if text != row['text'] or row['metric'] not in CATEGORIES[category][1]:
            raise ValueError('Annotation category, metric or source text differs.')
        if not isinstance(row.get('record_id'), str) or not row['record_id'] or row['record_id'] in identities:
            raise ValueError('Structured annotations require unique nonempty record identities.')
        if row['split'] not in ('train', 'validation', 'test', 'diagnostic') or not row.get('family_id'):
            raise ValueError('Structured annotations require a family and a declared role.')
        family = row['family_id']
        if family in families and families[family] != row['split']:
            raise ValueError('A semantic family crosses annotation partitions.')
        families[family] = row['split']
        key = fingerprint(''.join(text.split()))
        if key in texts and texts[key] != row['split']:
            raise ValueError('The same text crosses annotation partitions.')
        texts[key] = row['split']
        target, expectation, provenance = row['target'], row['expectation'], row['annotation']
        if target['experiencer'] != 'self' or not isinstance(target.get('period'), str) or not target['period']:
            raise ValueError('Each target must name its experiencer and observation period.')
        direction = next(m[3] for m in METRICS if m[0] == row['metric'])
        if expectation['scale_direction'] != direction or expectation['order'] not in ('lower', 'higher'):
            raise ValueError('The annotation must preserve the metric scale direction and paired ordering.')
        if type(expectation['explicit_absence']) is not bool or expectation['explicit_absence'] and direction != 'absent_to_extreme':
            raise ValueError('Explicit absence is valid only on an intensity scale with an absent endpoint.')
        if provenance.get('author_kind') not in ('assistant', 'human') or type(provenance.get('human_reviewed')) is not bool:
            raise ValueError('Annotation authorship and human review must be explicit.')
        if provenance['human_reviewed'] and not provenance.get('human_reviewer_id'):
            raise ValueError('A human-reviewed claim needs a human reviewer identity.')
        evidence, excluded = target['evidence'], target['excluded']
        if not evidence:
            raise ValueError('A scored target needs quoted evidence; unknown evidence must not be invented.')
        for span in evidence + excluded:
            start, end = span['start'], span['end']
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text) or text[start:end] != span['quote']:
                raise ValueError('Evidence character offsets must match the exact source substring.')
        if any(span.get('reason') not in ('other_person', 'past_resolved', 'negated_claim', 'ironic_literal') for span in excluded):
            raise ValueError('Every excluded span needs a stated scope reason.')
        if any(max(a['start'], b['start']) < min(a['end'], b['end']) for a in evidence for b in excluded):
            raise ValueError('Evidence cannot overlap an explicitly excluded span.')
        identities[row['record_id']] = row
    for row in records:
        anchor_id = row['expectation']['equivalent_to']
        if anchor_id not in identities:
            raise ValueError('An equivalence link must resolve inside the annotation file.')
        anchor = identities[anchor_id]
        keys = ('family_id', 'split', 'category', 'metric')
        if any(row[key] != anchor[key] for key in keys) or row['expectation']['order'] != anchor['expectation']['order']:
            raise ValueError('Equivalent records must share the target and family.')
        if any(row['target'][key] != anchor['target'][key] for key in ('experiencer', 'period')):
            raise ValueError('Equivalent records cannot silently change the person or time period.')
        if anchor['expectation']['equivalent_to'] != anchor_id:
            raise ValueError('An equivalence anchor must be a self-linked clean statement.')
    return {'records': len(records), 'families': len(families),
            'human_reviewed': sum(row['annotation']['human_reviewed'] for row in records),
            'validation_scope': 'format, references and split integrity only; semantic correctness is not certified'}
