"""Per-metric regression error, category macro averages and an honest mean baseline."""

from collections import defaultdict
import math
import numpy as np

from .schema import CATEGORIES,METRICS,FORMAL_CATEGORIES


def average_ranks(values):
    # PSEUDOCODE: stable-sort values -> give tied observations their average rank.
    order=np.argsort(values,kind='stable')
    ranks=np.empty(len(order),dtype=float)
    start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[end]]==values[order[start]]:
            end+=1
        ranks[order[start:end]]=(start+end-1)/2
        start=end
    return ranks


def errors(truth,predicted):
    # PSEUDOCODE: validate paired scores -> calculate absolute/squared errors and tied-rank correlation.
    truth=np.asarray(truth,dtype=float)
    predicted=np.asarray(predicted,dtype=float)
    if not len(truth) or truth.shape!=predicted.shape or not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError('Evaluation needs matching finite nonempty values.')
    delta=np.abs(truth-predicted)
    a,b=average_ranks(truth),average_ranks(predicted)
    rho=float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
    return {'n':len(truth),'mae':float(delta.mean()),'rmse':float(np.sqrt(np.mean(delta**2))),
            'within_5':float(np.mean(delta<=5)),'within_10':float(np.mean(delta<=10)),
            'spearman':rho}


def mean_baseline(train_rows):
    # PSEUDOCODE: fit one constant mean per score using training labels only.
    values=defaultdict(list)
    for row in train_rows:
        for key,value in row['scores'].items():
            values[key].append(value)
    return {key:float(np.mean(values[key])) for key,*_ in METRICS}


def median_baseline(train_rows):
    # PSEUDOCODE: fit the absolute-error-optimal constant on training labels only.
    values = defaultdict(list)
    for row in train_rows:
        for key, value in row['scores'].items():
            values[key].append(value)
    return {key: float(np.median(values[key])) for key, *_ in METRICS}


def report(rows,predictions,baseline,median_reference=None):
    # PSEUDOCODE: check every prediction -> aggregate per-score/category errors -> compare training-only constants.
    if len(rows)!=len(predictions) or not rows:
        raise ValueError('Every evaluation row must have one prediction.')
    truth,pred=defaultdict(list),defaultdict(list)
    for row,values in zip(rows,predictions):
        keys=CATEGORIES[row['category']][1]
        if set(values)!=set(keys) or any(not math.isfinite(v) or not 0<=v<=100 for v in values.values()):
            raise ValueError('Malformed or out-of-range prediction.')
        for key in keys:
            truth[key].append(row['scores'][key])
            pred[key].append(values[key])
    per_metric={key:errors(truth[key],pred[key]) for key in truth}
    per_category={c:{'n':sum(r['category']==c for r in rows),
                      'mae':float(np.mean([per_metric[k]['mae'] for k in keys if k in per_metric]))}
                  for c,(_,keys) in CATEGORIES.items() if any(k in per_metric for k in keys)}
    formal=[v['mae'] for c,v in per_category.items() if c in FORMAL_CATEGORIES]
    return {'records':len(rows),'per_metric':per_metric,'per_category':per_category,
            'macro_category_mae':float(np.mean([v['mae'] for v in per_category.values()])),
            'formal_category_mae':float(np.mean(formal)) if formal else None,
            'mean_baseline':{k:errors(truth[k],[baseline[k]]*len(truth[k])) for k in truth},
            'median_baseline': {k: errors(truth[k], [median_reference[k]] * len(truth[k])) for k in truth} if median_reference is not None else None,
            'baseline_interpretation': 'training median minimizes constant MAE; training mean minimizes constant squared error',
            'complete_output_rate':1.0,'in_range_rate':1.0,
            'reference':'semantic_reference_scores_not_clinical_measurements',
            'independent_human_gold':False}
