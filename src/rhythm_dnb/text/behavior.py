"""Independent sentence-pair diagnostics; ranking checks are not population accuracy estimates."""

from collections import defaultdict
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
