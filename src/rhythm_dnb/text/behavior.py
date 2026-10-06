"""Independent sentence-pair diagnostics; ranking checks are not population accuracy estimates."""

from collections import defaultdict
import math
from statistics import mean, median
from .schema import CATEGORIES, validate_input
from .train import predict_rows


def evaluate_pairs(model, tokenizer, cases, config, device):
    # PSEUDOCODE: validate distinct paired sentences -> score full contexts -> report expected-order failures by phenomenon.
    rows, ids = [], set()
    for case in cases:
        category = case['category']
        if case['id'] in ids or case['metric'] not in CATEGORIES[category][1]:
            raise ValueError('Behavior checks need unique identities and a compatible metric.')
        ids.add(case['id'])
        if case['lower'] == case['higher']:
            raise ValueError('A directional pair must contain different sentences.')
        for side in ('lower', 'higher'):
            _, text = validate_input(category, case[side])
            rows.append({'example_id': case['id'] + ':' + side, 'category': category, 'text': text,
                         'scores': {k: None for k in CATEGORIES[category][1]}})
    predictions = predict_rows(model, tokenizer, rows, config, device)
    results, groups = [], defaultdict(list)
    for i, case in enumerate(cases):
        low, high = (predictions[2*i+j]['scores'][case['metric']] for j in (0,1))
        passed = high > low
        result = {**case, 'lower_prediction': low, 'higher_prediction': high, 'difference': high-low, 'passed': passed}
        results.append(result); groups[case['phenomenon']].append(passed)
    return {'reference': 'assistant_authored_directional_expectations_not_human_gold',
            'checks': len(results), 'passed': sum(r['passed'] for r in results),
            'by_phenomenon': {k: {'checks': len(v), 'passed': sum(v)} for k,v in groups.items()}, 'results': results}


def context_rows(cases, guidance):
    # PSEUDOCODE: keep each semantic family together and construct clean, original and instructed inputs.
    from .schema import METRICS
    if not cases or set(guidance) != set(CATEGORIES) or any(not isinstance(v, str) or not v.strip() for v in guidance.values()):
        raise ValueError('Context diagnostics need cases and explicit guidance for every category.')
    rows, identifiers, sentences = [], set(), set()
    for case in cases:
        category, metric = case['category'], case['metric']
        if category not in CATEGORIES or metric not in CATEGORIES[category][1] or case['id'] in identifiers:
            raise ValueError('Invalid or duplicate context family.')
        if not isinstance(case['lower_is_explicit_absence'], bool) or not case.get('phenomenon'):
            raise ValueError('Context references require a phenomenon and an explicit boolean absence flag.')
        if case['lower_is_explicit_absence'] and next(m[3] for m in METRICS if m[0] == metric) != 'absent_to_extreme':
            raise ValueError('Absence references apply only to intensity scales with an absent endpoint.')
        identifiers.add(case['id'])
        for mode in ('anchor', 'original', 'scoped'):
            for side in ('lower', 'higher'):
                text = case[side]['anchor' if mode == 'anchor' else 'context']
                _, text = validate_input(category, text)
                if mode != 'scoped':
                    normalized = ''.join(text.split())
                    if normalized in sentences:
                        raise ValueError('Context families must contain distinct reference and challenge sentences.')
                    sentences.add(normalized)
                else:
                    text = guidance[category] + '\n原始记录：' + text
                rows.append({'example_id': case['id'] + ':' + mode + ':' + side, 'category': category,
                             'text': text, 'scores': {k: None for k in CATEGORIES[category][1]}})
    return rows


def context_report(cases, predictions):
    # PSEUDOCODE: compare equivalent meanings continuously; do not invent a drift cutoff or human score labels.
    if not cases or len(predictions) != len(cases) * 6:
        raise ValueError('Every context family requires six predictions in the documented input order.')
    results = []
    for index, case in enumerate(cases):
        scores = {}
        for offset, mode in enumerate(('anchor', 'original', 'scoped')):
            pair = [predictions[index * 6 + offset * 2 + j]['scores'][case['metric']] for j in (0, 1)]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 100 for v in pair):
                raise ValueError('Context predictions must be finite scores on the declared 0-100 scale.')
            scores[mode] = {'lower': pair[0], 'higher': pair[1], 'margin': pair[1] - pair[0], 'ordered': pair[1] > pair[0]}
        anchor = scores['anchor']
        for mode in ('original', 'scoped'):
            scores[mode]['anchor_drift'] = mean(abs(scores[mode][side] - anchor[side]) for side in ('lower', 'higher'))
            scores[mode]['contrast_drift'] = abs(scores[mode]['margin'] - anchor['margin'])
        results.append({'id': case['id'], 'phenomenon': case['phenomenon'], 'category': case['category'],
                        'metric': case['metric'], 'lower_is_explicit_absence': case['lower_is_explicit_absence'],
                        'scores': scores})

    def summarize(items):
        # PSEUDOCODE: retain absolute absence scores alongside ordering so a correctly ranked high error stays visible.
        absent = [item for item in items if item['lower_is_explicit_absence']]
        summary = {'families': len(items), 'explicit_absence_cases': len(absent)}
        for mode in ('anchor', 'original', 'scoped'):
            summary[mode] = {'ordered': sum(item['scores'][mode]['ordered'] for item in items),
                'mean_absence_score': mean(item['scores'][mode]['lower'] for item in absent) if absent else None}
            if mode != 'anchor':
                drift = [item['scores'][mode]['anchor_drift'] for item in items]
                summary[mode].update(mean_anchor_drift=mean(drift), median_anchor_drift=median(drift),
                    mean_contrast_drift=mean(item['scores'][mode]['contrast_drift'] for item in items))
        return summary

    return {'reference': 'assistant_authored_semantic_equivalence_not_human_gold', 'summary': summarize(results),
            'by_phenomenon': {name: summarize([r for r in results if r['phenomenon'] == name])
                              for name in sorted({r['phenomenon'] for r in results})}, 'results': results,
            'interpretation': 'Anchor drift is consistency with a clean paraphrase, not clinical or population accuracy.'}


def scope_consistency_report(records, predictions):
    # PSEUDOCODE: resolve evidence-linked anchors and retain drift, absence scores and paired ordering together.
    from .annotations import validate_scope_annotations
    validate_scope_annotations(records)
    if len(predictions) != len(records):
        raise ValueError('Each structured annotation needs exactly one prediction.')
    scores = {}
    for row, prediction in zip(records, predictions):
        value = prediction['scores'][row['metric']]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError('Structured prediction must be finite and within the metric bounds.')
        scores[row['record_id']] = value
    results, pairs = [], defaultdict(dict)
    for row in records:
        expected = row['expectation']
        anchor_id = expected['equivalent_to']
        if row['record_id'] == anchor_id:
            continue
        prediction, anchor = scores[row['record_id']], scores[anchor_id]
        results.append({'record_id': row['record_id'], 'family_id': row['family_id'], 'category': row['category'],
            'metric': row['metric'], 'variant': row['variant'], 'order': expected['order'], 'score': prediction,
            'anchor_score': anchor, 'anchor_drift': abs(prediction - anchor),
            'explicit_absence': expected['explicit_absence'], 'excluded_reasons': sorted({s['reason'] for s in row['target']['excluded']})})
        group = pairs[(row['family_id'], row['variant'])]
        if expected['order'] in group:
            raise ValueError('A context variant has duplicate lower or higher members.')
        group[expected['order']] = prediction
    if not results or any(set(pair) != {'lower', 'higher'} for pair in pairs.values()):
        raise ValueError('Structured consistency checks need complete lower/higher context pairs.')
    absent = [r['score'] for r in results if r['explicit_absence']]
    return {'contexts': len(results), 'pairs': len(pairs),
            'ordered_pairs': sum(pair['higher'] > pair['lower'] for pair in pairs.values()),
            'mean_anchor_drift': mean(r['anchor_drift'] for r in results),
            'median_anchor_drift': median(r['anchor_drift'] for r in results),
            'explicit_absence_contexts': len(absent), 'mean_absence_score': mean(absent) if absent else None,
            'results': results, 'independent_human_gold': False,
            'interpretation': 'Consistency against the same model on a clean equivalent statement, not population accuracy.'}
