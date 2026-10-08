"""External emotion and stress diagnostics without inventing personal 0-100 reference scores."""

import math
import numpy as np

MAPPING = {'joy': 'joy_intensity', 'sadness': 'sadness_intensity'}


def binary_stress_report(rows, predictions):
    # PSEUDOCODE: verify aligned binary stress references -> evaluate score ranking without selecting a threshold or changing label scale.
    from sklearn.metrics import roc_auc_score, average_precision_score
    if not rows or len(rows) != len(predictions) or len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('External stress evaluation requires unique aligned records.')
    for row, prediction in zip(rows, predictions):
        if (row.get('external_scale') != 'binary_stress' or row.get('category') != 'stress'
                or type(row.get('external_reference')) is not int or row['external_reference'] not in (0, 1)
                or not isinstance(row.get('group_id'), str) or not row['group_id'].strip()
                or prediction.get('example_id') != row['example_id']):
            raise ValueError('Stress references must retain binary labels, source groups and exact prediction identities.')
        score = prediction.get('estimate')
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError('Stress estimates must be finite values on the project 0-100 scale.')
    truth = np.asarray([r['external_reference'] for r in rows])
    scores = np.asarray([p['estimate'] for p in predictions])
    both_classes = len(set(truth)) == 2
    return {'metric': 'stress_intensity', 'n': len(rows), 'source_groups': len({r['group_id'] for r in rows}),
        'positive': int(truth.sum()), 'negative': int((truth == 0).sum()), 'prevalence': float(truth.mean()),
        'roc_auc': float(roc_auc_score(truth, scores)) if both_classes else None,
        'average_precision': float(average_precision_score(truth, scores)) if both_classes else None,
        'score_by_label': {str(label): {'n': int((truth == label).sum()),
            'quartiles': np.quantile(scores[truth == label], [.25, .5, .75]).tolist() if (truth == label).any() else None}
            for label in (0, 1)},
        'scale_conversion': None, 'degree_mae': None, 'clinical_accuracy': None, 'threshold_selected': False,
        'interpretation': 'external binary stress ranking; does not validate 0-100 degrees, Chinese transfer or longitudinal changes',
        'participant_independence_verified': False, 'base_pretraining_exposure': 'unknown'}


def ordinal_emotion_report(rows, predictions):
    # PSEUDOCODE: compare fixed shared emotion concepts on original ordinal labels; retain imbalance and scale differences.
    from scipy.stats import pearsonr, spearmanr, kendalltau
    from sklearn.metrics import roc_auc_score, average_precision_score
    if not rows or len(rows) != len(predictions) or len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('External evaluation requires unique identities and aligned predictions.')
    for row, prediction in zip(rows, predictions):
        if row.get('external_scale') != 'ordinal_0_3' or row['category'] != 'emotion' or set(row['external_reference']) != set(MAPPING):
            raise ValueError('External labels must retain the declared ordinal emotion contract.')
        if any(type(v) is not int or v not in (0, 1, 2, 3) for v in row['external_reference'].values()):
            raise ValueError('External emotion labels require integer levels 0 through 3.')
        if any(type(prediction['scores'][key]) not in (int, float) or not math.isfinite(prediction['scores'][key])
               or not 0 <= prediction['scores'][key] <= 100 for key in MAPPING.values()):
            raise ValueError('External predictions require bounded continuous project scores.')
    result = {}
    for label, key in MAPPING.items():
        truth = np.asarray([r['external_reference'][label] for r in rows])
        scores = np.asarray([p['scores'][key] for p in predictions])
        nonconstant = len(set(truth)) > 1 and len(set(scores)) > 1
        presence = truth > 0
        positive_nonconstant = len(set(truth[presence])) > 1 and len(set(scores[presence])) > 1
        levels = {}
        for level in range(4):
            selected = scores[truth == level]
            levels[str(level)] = {'n': len(selected), 'mean_prediction': float(selected.mean()) if len(selected) else None,
                'prediction_quartiles': np.quantile(selected, [.25, .5, .75]).tolist() if len(selected) else None}
        result[key] = {'n': len(truth), 'external_label': label, 'levels': levels,
            'spearman': float(spearmanr(truth, scores).statistic) if nonconstant else None,
            'pearson': float(pearsonr(truth, scores).statistic) if nonconstant else None,
            'kendall_tau_b': float(kendalltau(truth, scores).statistic) if nonconstant else None,
            'presence_roc_auc': float(roc_auc_score(presence, scores)) if len(set(presence)) == 2 else None,
            'presence_average_precision': float(average_precision_score(presence, scores)) if len(set(presence)) == 2 else None,
            'presence_prevalence': float(presence.mean()),
            'positive_intensity': {'n': int(presence.sum()),
                'spearman': float(spearmanr(truth[presence], scores[presence]).statistic) if positive_nonconstant else None,
                'pearson': float(pearsonr(truth[presence], scores[presence]).statistic) if positive_nonconstant else None}}
    return {'metrics': result, 'rows': len(rows), 'mapping': MAPPING, 'scale_conversion': None,
            'clinical_accuracy': None, 'evidence_calibration': None,
            'interpretation': 'external perceived-emotion ranking diagnostic; ordinal labels are not 0-100 personal clinical scores',
            'limitations': ['does not review our training labels', 'does not evaluate the other fifteen metrics',
                           'no independent participant or longitudinal metadata', 'base pretraining exposure may be unknown']}
