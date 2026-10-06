"""Audit every scoring head and prepare blinded review without manufacturing reference labels."""

from collections import Counter, defaultdict
from pathlib import Path
from ..provenance import canonical_json, fingerprint
from .schema import CATEGORIES, METRICS
from .labels import STATES, evidence_target, validate_labels


def review_guide():
    # PSEUDOCODE: expose scale endpoints and scope questions without revealing any previous numerical answer.
    endpoints = {'absent_to_extreme': ['明确没有该感受或症状', '该指标的极强端点'],
        'very_bad_to_very_good': ['非常差', '非常好'], 'none_to_strong': ['完全没有意愿', '非常强的意愿']}
    return {'metrics': [{'key': key, 'category': category, 'label': label, 'zero': endpoints[direction][0],
            'hundred': endpoints[direction][1]} for key, category, label, direction in METRICS],
        'states': {'supported': '有本人、对应时段的依据，并给出程度分数',
            'supported_unscored': '有依据，但程度未定；分数留空', 'explicit_absence': '症状明确不存在；仅适用程度量表零端点',
            'insufficient_evidence': '原文无法判断；分数留空', 'unreviewed': '尚未完成；分数留空', 'disputed': '仍有分歧；分数留空'},
        'questions': ['描述的是谁？', '对应当前还是过去哪个时段？', '否定、转折、引用和反话改变了哪部分含义？',
            '哪些原文提供依据，哪些片段应排除？', '能否确定程度？不能时不要猜分或填中间值。'],
        'rules': ['不从情绪好坏推断所有情绪强度；悲伤和愉快可同时存在。',
            '主观睡眠质量不等于跨日节律稳定；缺少多日证据时不推断规律性。',
            '两名标注者先独立填写，再由仲裁解决分歧；不可看旧分数互相迁就。'],
        'intervals_are_not_clinical_cutoffs': True, 'independent_review_completed': False}


def coverage_report(rows):
    # PSEUDOCODE: count each metric, reference range, evidence state and time basis separately in every role.
    roles = sorted({r['split'] for r in rows})
    report = {}
    for role in roles:
        items = [r for r in rows if r['split'] == role]
        metrics = {}
        for key, category, *_ in METRICS:
            selected = [r for r in items if r['category'] == category]
            values = [r['scores'][key] for r in selected if r['scores'][key] is not None]
            metrics[key] = {'rows': len(selected), 'labeled': len(values),
                'people_with_labels': len({r['participant_id'] for r in selected if r['scores'][key] is not None and r.get('participant_id')}),
                'minimum': min(values) if values else None, 'maximum': max(values) if values else None,
                'zero': sum(v == 0 for v in values), 'hundred': sum(v == 100 for v in values),
                'score_bands': dict(Counter(str(min(int(v // 20), 4)) for v in values)),
                'evidence_states': dict(Counter(r.get('label_states', {}).get(key, 'legacy_unspecified') for r in selected)),
                'supported_evidence': sum(evidence_target(r, key) == 1 for r in selected),
                'unsupported_evidence': sum(evidence_target(r, key) == 0 for r in selected)}
        report[role] = {'rows': len(items), 'metrics': metrics,
            'origins': dict(Counter(r['origin'] for r in items)),
            'timestamped': sum(r.get('observed_at') is not None for r in items),
            'time_bases': dict(Counter(r.get('temporal_basis', 'unspecified') for r in items)),
            'scope_rows': sum(bool(r.get('scope_targets')) for r in items)}
    return {'partitions': report, 'id': fingerprint(rows),
            'score_band_definition': '[0,20),[20,40),[40,60),[60,80),[80,100]; descriptive only, not clinical cutoffs',
            'semantic_validity_certified': False}


def training_readiness(partitions, config, *, experimental):
    # PSEUDOCODE: expose absent validation heads and require actual targets for each requested auxiliary loss.
    coverage = coverage_report([r for items in partitions.values() for r in items])
    blockers, warnings = [], []
    for role in ('train', 'validation'):
        metrics = coverage['partitions'].get(role, {}).get('metrics', {})
        missing = [key for key, *_ in METRICS if not metrics.get(key, {}).get('labeled')]
        if missing:
            (blockers if config.get('require_all_validation_metrics') else warnings).append('No ' + role + ' scores: ' + ', '.join(missing))
    if config.get('scope_loss_weight', 0):
        for role in ('train', 'validation'):
            if not coverage['partitions'][role]['scope_rows']:
                blockers.append(role + ': scope loss requested but no scope annotations')
    if experimental and config.get('train_experimental_evidence'):
        for role in ('train', 'validation'):
            selected = [r for r in partitions[role] if r.get('label_states')]
            if not any(evidence_target(r, k) == 0 for r in selected for k in r['scores']):
                blockers.append(role + ': evidence training requires explicit insufficient_evidence annotations')
            if not any(evidence_target(r, k) == 1 for r in selected for k in r['scores']):
                blockers.append(role + ': evidence training requires explicit supported annotations')
            for key, category, *_ in METRICS:
                targets = {evidence_target(r, key) for r in selected if r['category'] == category}
                if not {0., 1.} <= targets:
                    warnings.append(role + ': explicit evidence classes incomplete for ' + key)
    return {'blockers': blockers, 'warnings': warnings, 'coverage': coverage,
            'ready_for_training': not blockers, 'primary_dnb_validated': False}


def export_blind_review(rows, output_dir, *, per_metric, seed):
    # PSEUDOCODE: sample endpoints, unscored and ordinary cases -> hide old scores -> separate rater packets and coordinator map.
    if type(per_metric) is not int or per_metric < 1 or type(seed) is not int:
        raise ValueError('Provide an explicit positive review budget and integer seed.')
    if len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('Review inputs need unique example identities.')
    chosen, selected_for = {}, defaultdict(list)
    for key, category, *_ in METRICS:
        candidates = [r for r in rows if r['category'] == category]
        strata = defaultdict(list)
        for row in candidates:
            value = row['scores'][key]
            band = 'missing' if value is None else str(min(int(value // 20), 4))
            strata[band].append(row)
        for items in strata.values():
            items.sort(key=lambda r: fingerprint([seed, key, r['example_id']]))
        selections = []
        while len(selections) < per_metric and any(strata.values()):
            for band in sorted(strata):
                if strata[band] and len(selections) < per_metric:
                    selections.append(strata[band].pop(0))
        for row in selections:
            chosen[row['example_id']] = row
            selected_for[row['example_id']].append(key)
    if not chosen:
        raise ValueError('No review candidates.')
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    coordinator, packets = [], {'A': [], 'B': []}
    for identity, row in sorted(chosen.items()):
        mapping = {'example_id': identity, 'original': row, 'selected_for': selected_for[identity], 'blind_ids': {}}
        for slot in packets:
            blind = fingerprint([seed, slot, identity])
            mapping['blind_ids'][slot] = blind
            packets[slot].append({'blind_id': blind, 'category': row['category'], 'text': row['text'],
                'rater_id': None, 'human_reviewed': False, 'scores': {k: None for k in row['scores']},
                'label_states': {k: 'unreviewed' for k in row['scores']}, 'scope_targets': {}})
        coordinator.append(mapping)
    for slot, items in packets.items():
        items.sort(key=lambda r: fingerprint([seed, slot, r['blind_id'], 'order']))
        (output / ('rater-' + slot + '.json')).write_text(canonical_json({'rows': items}), encoding='utf-8')
    (output / 'coordinator.json').write_text(canonical_json({'source_id': fingerprint(rows), 'rows': coordinator}), encoding='utf-8')
    (output / 'guide.json').write_text(canonical_json(review_guide()), encoding='utf-8')
    receipt = {'source_id': fingerprint(rows), 'review_rows': len(chosen), 'per_metric_budget': per_metric,
        'seed': seed, 'completed_human_reviews': 0, 'status': 'awaiting_independent_review',
        'states': STATES, 'clinical_gold': False}
    (output / 'receipt.json').write_text(canonical_json(receipt), encoding='utf-8')
    return receipt


def compare_blind_review(coordinator, packet_a, packet_b, *, tolerance):
    # PSEUDOCODE: require exact blind inventories and different humans -> expose disagreements without averaging scores.
    from .annotations import compare_annotations, validate_scope_targets
    if type(tolerance) not in (float, int) or not 0 <= tolerance <= 100:
        raise ValueError('Explicit finite adjudication tolerance in score points is required.')
    indexed = []
    for slot, packet in (('A', packet_a), ('B', packet_b)):
        lookup = {r['blind_id']: r for r in packet['rows']}
        if len(lookup) != len(packet['rows']) or set(lookup) != {r['blind_ids'][slot] for r in coordinator['rows']}:
            raise ValueError('Blind review inventory differs from the assigned packet.')
        indexed.append(lookup)
    results = []
    for mapping in coordinator['rows']:
        annotations = []
        for i, slot in enumerate(('A', 'B')):
            r = indexed[i][mapping['blind_ids'][slot]]
            if r['text'] != mapping['original']['text'] or r['category'] != mapping['original']['category']:
                raise ValueError('Review text or category changed.')
            validate_labels(r); validate_scope_targets(r)
            if r.get('human_reviewed') is not True or any(s == 'unreviewed' for s in r['label_states'].values()):
                raise ValueError('Review is incomplete or not independently human-reviewed.')
            annotations.append({**r, 'example_id': mapping['example_id']})
        compared = compare_annotations(*annotations, tolerance=tolerance)
        different_states = [k for k in annotations[0]['scores'] if annotations[0]['label_states'][k] != annotations[1]['label_states'][k]]
        unresolved = any('disputed' in r['label_states'].values() for r in annotations)
        scope_disagreement = annotations[0].get('scope_targets', {}) != annotations[1].get('scope_targets', {})
        results.append({**compared, 'state_disagreement': different_states, 'scope_disagreement': scope_disagreement,
            'adjudication_required': compared['adjudication_required'] or bool(different_states) or scope_disagreement or unresolved})
    return {'items': results, 'cases': len(results), 'adjudication_required': sum(r['adjudication_required'] for r in results),
            'automatic_label_writeback': False, 'tolerance': tolerance, 'averaged_labels': False}
