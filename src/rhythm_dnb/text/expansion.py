"""Read-only export of authored rhythm narratives with sparse references and new person holdouts."""

from collections import Counter, defaultdict
from contextlib import closing
import json
import math
from pathlib import Path
import re
import sqlite3

from ..provenance import canonical_json, fingerprint
from .checkpoint import inspect_checkpoint
from .experiment import PURPOSE, REFERENCE, validate_rows
from .schema import CATEGORIES, FORMAL_CATEGORIES

# These gates only discard unsupported targets; they never generate or change a numerical score.
# A lexical match is a weak-reference eligibility check, not an independent semantic review.
MENTIONS = {
    'mood_valence': r'心情|情绪',
    'joy_intensity': r'开心|高兴|快乐|兴奋|愉快|喜悦|开心不起来',
    'sadness_intensity': r'低落|难过|伤心|沮丧|失落|悲伤',
    'anxiety_intensity': r'焦虑|紧张|担心|忧虑|不安',
    'irritability_intensity': r'烦|生气|发火|恼火|脾气',
    'appetite_loss_intensity': r'胃口|食欲|吃不下|不想吃',
    'excess_intake_intensity': r'吃撑|过量|吃得太多|吃太多|过饱|暴食|停不下|吃多了|吃得过多',
    'meal_irregularity_intensity': r'饭点|餐点|按时吃|规律|进餐时间|吃饭时间|餐间隔|饭点间隔|漏餐|漏吃|没吃早餐|没吃午餐|没吃晚餐',
    'sleep_quality': r'睡眠质量|睡得|睡的质量|睡觉质量|睡眠感受',
    'sleep_onset_difficulty': r'入睡困难|难以入睡|睡不着|翻来覆去|入睡顺利|很快睡着|入睡很快|入睡容易|难入睡|入睡费劲',
    'sleep_disruption_intensity': r'夜醒|早醒|中途醒|半夜醒|醒了|醒过|断断续续|一觉到|睡到天亮|睡眠中断',
    'post_sleep_fatigue': r'(?:醒后|醒来|起床|早上|起床后).{0,16}(?:疲|累|困|精神)',
    'social_willingness': r'愿|不想.{0,5}(?:聊|见|社交)|想.{0,5}(?:聊|见|社交)|主动|回避|拒绝邀约',
    'social_satisfaction': r'满意|愉快|舒服|开心|聊得|交流.{0,5}顺|沟通.{0,5}顺',
    'loneliness_intensity': r'孤独|寂寞|疏离|被忽略|没人陪',
    'social_burden_intensity': r'负担|压力|累|疲|勉强|应付|消耗|耗费',
}


def text_identity(text):
    # PSEUDOCODE: ignore whitespace/punctuation but retain wording, numbers and negation.
    return re.sub(r'\W+', '', text).lower()


def template_identity(text):
    # PSEUDOCODE: collapse numeric slot substitutions without erasing other semantic words.
    text = re.sub(r'\d+(?:[.:：点]\d+)*', '#', text)
    text = re.sub(r'[零〇一二两三四五六七八九十百千万]+(?=小时|分钟|点|毫升|毫克|次|人|杯|口|千卡)', '#', text)
    return re.sub(r'[^\w#]+', '', text).lower()


def person_roles(rows, seed):
    # PSEUDOCODE: group all visits by person -> stratify cohorts -> reserve roughly 10% each for validation/test.
    sources = defaultdict(set)
    for row in rows:
        source = 'DelSoM' if row['source_dataset'].startswith('DelSoM') else row['source_dataset']
        sources[row['participant_id']].add(source)
    strata = defaultdict(list)
    for person, names in sources.items():
        strata[tuple(sorted(names))].append(person)
    roles = {}
    for people in strata.values():
        people.sort(key=lambda p: fingerprint([seed, p]))
        n = max(1, round(len(people) * .1)) if len(people) >= 5 else 0
        for i, person in enumerate(people):
            roles[person] = 'test' if i < n else 'validation' if i < 2 * n else 'train'
    return roles


def export_expansion(database, legacy_development, legacy_test, checkpoint, output_dir, *, seed=20261006, supplement=None):
    # PSEUDOCODE: verify prior data -> snapshot source rows -> mask unsupported labels -> freeze new person-separated partitions.
    prior, checkpoint_id = inspect_checkpoint(checkpoint, allow_experimental=True)
    old_dev = json.loads(Path(legacy_development).read_text(encoding='utf-8'))
    old_test = json.loads(Path(legacy_test).read_text(encoding='utf-8'))
    if old_dev['manifest']['id'] != prior['dataset']['id'] or old_test['manifest']['id'] != prior['dataset']['id']:
        raise ValueError('Legacy reference files do not belong to the initial model.')
    from .experiment import prepare_development, validate_holdout
    prepare_development(old_dev)
    validate_holdout(old_test, prior)
    old_rows = old_dev['rows'] + old_test['rows']
    old_templates = {template_identity(r['text']) for r in old_rows}
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        observations = [dict(r) for r in db.execute('SELECT o.*,r.payload AS raw,p.payload AS provenance '
            'FROM observations o JOIN raw_inputs r USING(record_id) JOIN observation_provenance p USING(record_id) ORDER BY o.record_id')]
        if len(observations) != db.execute('SELECT count(*) FROM observations').fetchone()[0]:
            raise ValueError('Some observations lack raw text or provenance.')
    roles = person_roles(observations, seed)
    rows, rejected, masked = [], [], Counter()
    source_snapshot = []
    for record in observations:
        raw, provenance = json.loads(record['raw']), json.loads(record['provenance'])['provenance']
        for category in FORMAL_CATEGORIES:
            text = raw.get(category + '_description_text')
            keys = CATEGORIES[category][1]
            original = {k: record.get('text_' + k) for k in keys}
            rid = 'rhythm:' + record['record_id'] + ':' + category
            source_snapshot.append([rid, text, original, record['participant_id']])
            if not isinstance(text, str) or not text.strip() or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100 for v in original.values()):
                rejected.append([rid, 'invalid_text_or_scores']); continue
            if any(provenance.get('text_' + k, {}).get('method') != 'manual_contextual_rewrite' for k in keys):
                rejected.append([rid, 'unverified_rewrite_provenance']); continue
            if template_identity(text) in old_templates:
                rejected.append([rid, 'overlap_with_previous_experiment']); continue
            scores = {k: v if re.search(MENTIONS[k], text) else None for k, v in original.items()}
            masked.update(k for k, v in scores.items() if v is None)
            if not any(v is not None for v in scores.values()):
                rejected.append([rid, 'no_explicit_metric_mention']); continue
            rows.append({'example_id': rid, 'source_record_id': 'rhythm:' + record['record_id'],
                'participant_id': 'rhythm:' + record['participant_id'], 'group_id': 'rhythm:' + record['participant_id'],
                'category': category, 'text': text.strip(), 'scores': scores, 'origin': 'authored_simulation',
                'annotation_source': REFERENCE, 'source_dataset': record['source_dataset'],
                'split': roles[record['participant_id']], 'reference_method': 'manual_contextual_rewrite_with_mention_filter'})
    groups = defaultdict(list)
    for row in rows:
        groups[(row['category'], text_identity(row['text']))].append(row)
    unique = []
    for items in groups.values():
        if len({canonical_json(r['scores']) for r in items}) > 1:
            rejected.extend([r['example_id'], 'identical_text_conflicting_scores'] for r in items)
        else:
            # Choose by identity alone, never by score or eventual model error.
            items.sort(key=lambda r: fingerprint([seed, r['example_id']]))
            unique.append(items[0])
            rejected.extend([r['example_id'], 'duplicate_text'] for r in items[1:])
    templates = defaultdict(list)
    for row in unique:
        templates[(row['category'], template_identity(row['text']))].append(row)
    accepted = []
    for items in templates.values():
        items.sort(key=lambda r: fingerprint([seed, r['example_id']]))
        accepted.append(items[0])
        rejected.extend([r['example_id'], 'numeric_template_duplicate'] for r in items[1:])
    accepted.extend(r for r in old_dev['rows'] if r['split'] == 'train')
    supplemental = []
    if supplement is not None:
        supplemental = [json.loads(line) for line in Path(supplement).read_text(encoding='utf-8').splitlines() if line.strip()]
        for i, item in enumerate(supplemental):
            if set(item['scores']) - set(CATEGORIES[item['category']][1]):
                raise ValueError('Supplement contains an unsupported score.')
            if template_identity(item['text']) in old_templates or any(template_identity(r['text']) == template_identity(item['text']) for r in accepted):
                raise ValueError('Supplement overlaps existing data.')
            accepted.append({'example_id': f'complex-training:{i}', 'group_id': 'complex-training:' + item['case_id'],
                'source_record_id': 'complex-training:' + item['case_id'], 'split': 'train', 'origin': 'authored_simulation',
                'category': item['category'], 'text': item['text'], 'annotation_source': REFERENCE,
                'scores': {k: item['scores'].get(k) for k in CATEGORIES[item['category']][1]},
                'phenomenon': item['phenomenon'], 'reference_method': 'assistant_authored_semantic_reference'})
    categories = {'train': list(CATEGORIES), 'validation': list(FORMAL_CATEGORIES), 'test': list(FORMAL_CATEGORIES)}
    accepted, partitions = validate_rows(accepted, ('train', 'validation', 'test'), categories=categories, allow_missing=True)
    for role in ('validation', 'test'):
        for key, values in prior['dataset']['development_groups'].items():
            if set(values) & {r[key] for r in partitions[role] if r.get(key)}:
                raise ValueError('New holdout overlaps previous model exposure: ' + key)
        if {fingerprint(''.join(r['text'].split())) for r in partitions[role]} & set(prior['dataset']['development_text_hashes']):
            raise ValueError('New holdout contains previous development text.')
    development = partitions['train'] + partitions['validation']
    manifest = {'purpose': PURPOSE, 'reference': REFERENCE, 'id': fingerprint(accepted),
        'development_id': fingerprint(development), 'test_id': fingerprint(partitions['test']),
        'initial_checkpoint_id': checkpoint_id, 'source_snapshot_id': fingerprint(source_snapshot),
        'counts': {r: len(items) for r, items in partitions.items()}, 'categories': categories, 'allow_missing': True,
        'origins': dict(Counter(r['origin'] for r in accepted)), 'independent_human_gold': False,
        'independent_people_verified': False, 'split_unit': 'recorded_participant_ids_within_source_cohorts',
        'evidence_calibration': None, 'seed': seed, 'holdout_fraction_per_source': .1,
        'known_limitations': ['authored_weak_references', 'lexical_mention_is_not_semantic_verification',
                             'semantic_paraphrase_duplicates_not_exhaustively_excluded', 'no_new_stress_holdout'],
        'mention_patterns': MENTIONS}
    manifest.update(supplement_id=fingerprint(supplemental), supplement_texts=len(supplemental))
    audit = {'source_records': len(observations), 'source_texts': len(source_snapshot),
        'source_snapshot_id': manifest['source_snapshot_id'], 'masked_targets': dict(masked),
        'excluded_reasons': dict(Counter(reason for _, reason in rejected)), 'excluded_rows': rejected,
        'partitions': {role: {'texts': len(items), 'people': len({r['participant_id'] for r in items if r.get('participant_id')}),
            'categories': dict(Counter(r['category'] for r in items)),
            'labeled_metrics': dict(Counter(k for r in items for k,v in r['scores'].items() if v is not None))}
            for role, items in partitions.items()}}
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    for name, items in (('development', development), ('test', partitions['test'])):
        (output / (name + '.json')).write_text(canonical_json({'manifest': manifest, 'rows': items}), encoding='utf-8')
    (output / 'manifest.json').write_text(canonical_json(manifest), encoding='utf-8')
    (output / 'audit.json').write_text(canonical_json(audit), encoding='utf-8')
    return {**manifest, 'audit_summary': {k: v for k,v in audit.items() if k != 'excluded_rows'}}
