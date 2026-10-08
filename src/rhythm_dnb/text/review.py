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
    scope = {
        'mood_valence': ('本人适用时段内的总体心情方向；有明确中性表述才可判断为中性。', '不能用愉快减悲伤自动计算；信息不足不填 50。'),
        'joy_intensity': ('本人表达的开心、愉悦或喜悦有多强。', '没有坏事不代表开心；悲伤与愉快可同时存在。'),
        'sadness_intensity': ('本人表达的悲伤、失落或难过有多强。', '不开心不一定是悲伤；转述他人的悲伤不计入。'),
        'anxiety_intensity': ('本人表达的担忧、紧张或不安有多强。', '不能把所有压力、恐惧或忙碌直接换成焦虑程度。'),
        'irritability_intensity': ('本人表达的烦躁、不耐烦或易受激惹有多强。', '描述冲突或生气事件不自动证明本人持续烦躁。'),
        'stress_intensity': ('本人感到要求、负担或应对压力有多强。', '任务多、日程满不自动等于主观压力高。'),
        'appetite_loss_intensity': ('本人进食欲望下降或难以产生食欲的程度。', '吃得少可能由时间、供应或主动控制造成，不能自动算食欲差。'),
        'excess_intake_intensity': ('原文明确表达本人吃得过多、超出需要或失控进食的程度。', '单个食物名称或份量没有个人参照时，不能自行判定过量。'),
        'meal_irregularity_intensity': ('本人在文字中明确描述的进餐时间不稳定、随意变动或紊乱程度。', '稳定晚吃与多日时间漂移不同；单个钟点不能证明不规律。'),
        'sleep_quality': ('本人对目标睡眠段整体睡得好坏的主观评价。', '只知道入睡速度、时长或醒来次数时，不自动补全总体质量。'),
        'sleep_onset_difficulty': ('本人目标睡眠段从准备睡觉到入睡的困难程度。', '晚睡、主动熬夜、他人睡不着或睡后中断不自动等于入睡困难。'),
        'sleep_disruption_intensity': ('本人目标睡眠段入睡后醒来、反复中断或难以再睡的程度。', '入睡前难睡或正常起床不能自动算睡眠中断。'),
        'post_sleep_fatigue': ('本人从目标睡眠段醒来后的疲劳、乏力或未恢复感。', '白天工作后累或睡前疲劳不能自动算醒后疲劳。'),
        'social_willingness': ('本人在适用时段想要或愿意与人互动的程度。', '实际见了几个人不等于愿意社交；没有机会也不等于没有意愿。'),
        'social_satisfaction': ('本人对目标互动或关系体验感到满意的程度。', '互动多、对方评价好或独处，不能自动证明本人满意或不满意。'),
        'loneliness_intensity': ('本人感到孤独、缺少连接或无人理解的程度。', '独处、朋友少不一定孤独；热闹中也可能孤独。'),
        'social_burden_intensity': ('本人感到互动带来负担、消耗或勉强应付的程度。', '社交少或偏好独处不自动等于社交负担高。'),
    }
    return {'metrics': [{'key': key, 'category': category, 'label': label, 'zero': endpoints[direction][0],
            'hundred': endpoints[direction][1], 'meaning': scope[key][0], 'do_not_infer': scope[key][1]}
            for key, category, label, direction in METRICS],
        'states': {'supported': '有本人、对应时段的依据，并给出程度分数',
            'supported_unscored': '有依据，但程度未定；分数留空', 'explicit_absence': '症状明确不存在；仅适用程度量表零端点',
            'insufficient_evidence': '原文无法判断；分数留空', 'unreviewed': '尚未完成；分数留空', 'disputed': '仍有分歧；分数留空'},
        'questions': ['描述的是谁？', '对应当前还是过去哪个时段？', '否定、转折、引用和反话改变了哪部分含义？',
            '哪些原文提供依据，哪些片段应排除？', '能否确定程度？不能时不要猜分或填中间值。'],
        'rules': ['不从情绪好坏推断所有情绪强度；悲伤和愉快可同时存在。',
            '主观睡眠质量不等于跨日节律稳定；缺少多日证据时不推断规律性。',
            '先固定本人及目标时段；已经缓解的过去状态不与当前状态取平均。时间或人物无法确定时留空。',
            '量表是文本表达程度的研究约定；不是临床诊断、发生概率或由关键词直接查出的分数。',
            '只有顺序或有无依据、不能确定具体程度时，用 supported_unscored；不要为凑齐指标猜分。',
            '两名标注者先独立填写，再由仲裁解决分歧；不可看旧分数互相迁就。'],
        'intervals_are_not_clinical_cutoffs': True, 'independent_review_completed': False}


def evidence_support_coverage(rows):
    # PSEUDOCODE: count jointly known support patterns; warn when balanced individual heads hide missing partial-information cases.
    partitions, warnings = {}, []
    for role in sorted({r['split'] for r in rows}):
        categories = {}
        for category, (_, keys) in CATEGORIES.items():
            selected = [r for r in rows if r['split'] == role and r['category'] == category]
            if not selected:
                continue
            patterns = Counter(''.join('?' if evidence_target(r, k) is None else str(int(evidence_target(r, k))) for k in keys) for r in selected)
            mixed = sum(n for pattern, n in patterns.items() if '0' in pattern and '1' in pattern)
            categories[category] = {'metric_order': list(keys), 'rows': len(selected), 'patterns': dict(sorted(patterns.items())),
                'mixed_support_rows': mixed, 'multiple_supported_rows': sum(n for p, n in patterns.items() if p.count('1') > 1),
                'all_targets_explicit_rows': sum(n for p, n in patterns.items() if '?' not in p)}
            if len(keys) > 1 and mixed == 0:
                warnings.append(role + '/' + category + ': no row explicitly distinguishes supported from unsupported category metrics')
        partitions[role] = categories
    return {'partitions': partitions, 'warnings': warnings,
            'pattern_legend': {'1': 'supported including explicit symptom absence', '0': 'insufficient evidence', '?': 'unreviewed or disputed'},
            'semantic_validity_certified': False}


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
    return {'partitions': report, 'id': fingerprint(rows), 'joint_evidence_coverage': evidence_support_coverage(rows),
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
