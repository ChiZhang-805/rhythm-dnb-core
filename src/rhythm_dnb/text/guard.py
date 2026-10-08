"""Fit a separate evidence classifier on frozen text features and keep experimental rejection uncertified."""

import json
import math
from pathlib import Path
import warnings
import numpy as np
from ..provenance import canonical_json, fingerprint, file_hash
from .schema import CATEGORIES, METRICS
from .labels import evidence_target
from .features import load_features


def binary_metrics(labels, probabilities):
    # PSEUDOCODE: report proper scoring losses and ranking without treating an arbitrary 0.5 as an acceptance policy.
    from scipy.special import xlogy
    from sklearn.metrics import roc_auc_score
    labels, probabilities = np.asarray(labels, dtype=float), np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or not len(labels) or labels.shape != probabilities.shape or not np.isin(labels, [0., 1.]).all() or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Evidence evaluation needs nonempty binary labels and bounded probabilities.')
    bounded = probabilities.clip(np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {'n': len(labels), 'positive': int(labels.sum()), 'negative': int(len(labels) - labels.sum()),
        'brier': float(np.mean((labels - probabilities) ** 2)),
        'log_loss': float(-np.mean(xlogy(labels, bounded) + xlogy(1-labels, 1-bounded))),
        'roc_auc': float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None}


def choose_empirical_threshold(labels, probabilities, target_precision, *, boundary='observed'):
    # PSEUDOCODE: meet validation precision -> maximize supported recall -> reject extra false acceptances that add no supported cases.
    if type(target_precision) not in (float, int) or not 0 < target_precision < 1:
        raise ValueError('Declare the experimental validation precision target explicitly.')
    if boundary not in ('observed', 'validation_gap_midpoint'):
        raise ValueError('Unknown evidence threshold boundary rule.')
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    binary_metrics(labels, probabilities)
    selected, best = None, None
    for threshold in sorted(set(probabilities.tolist()), reverse=True):
        accepted = probabilities >= threshold
        if labels[accepted].mean() >= target_precision:
            correct = int(labels[accepted].sum())
            objective = (correct, -int(accepted.sum() - correct))
            if best is None or objective > best:
                selected, best = float(threshold), objective
    if selected is not None and boundary == 'validation_gap_midpoint':
        rejected = probabilities[probabilities < selected]
        if not len(rejected):
            return 0.
        lower = float(rejected.max())
        # Every cutoff in this gap gives the same validation decisions; its midpoint maximizes the minimum margin.
        middle = lower + (selected-lower)/2
        return middle if middle > lower else selected
    return selected


def selective_metrics(labels, probabilities, threshold):
    # PSEUDOCODE: apply the already selected threshold and expose every false acceptance and rejection.
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    binary_metrics(labels, probabilities)
    if threshold is not None and (type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Evidence threshold must be null or a finite probability.')
    accepted = probabilities >= threshold if threshold is not None else np.zeros(len(labels), dtype=bool)
    return {'accepted': int(accepted.sum()), 'coverage': float(accepted.mean()),
        'accepted_supported': int(labels[accepted].sum()), 'false_acceptances': int((labels[accepted] == 0).sum()),
        'missed_supported': int(((labels == 1) & ~accepted).sum()),
        'precision': float(labels[accepted].mean()) if accepted.any() else None,
        'supported_recall': float(accepted[labels == 1].mean()) if (labels == 1).any() else None,
        'certified': False}


def _targets(rows, metric, role):
    # PSEUDOCODE: select only explicit reviewed evidence states of the requested metric and partition.
    indices, targets = [], []
    for index, row in enumerate(rows):
        if row['split'] == role and metric in row['scores']:
            target = evidence_target(row, metric)
            if target is not None:
                indices.append(index); targets.append(target)
    return np.asarray(indices, dtype=int), np.asarray(targets)


def fit_guard(cache, output_dir, *, regularization, target_precision):
    # PSEUDOCODE: fit scaling and independent logistic heads on train -> choose regularization/threshold on validation -> seal without test.
    from scipy.special import expit
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from collections import Counter
    rows, vectors, _, _, source = load_features(cache)
    vectors = np.asarray(vectors, dtype=np.float64)
    if {r['split'] for r in rows} != {'train', 'validation'}:
        raise ValueError('Guard fitting accepts train and validation only; sealed test must remain separate.')
    if not regularization or len(set(regularization)) != len(regularization) or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in regularization):
        raise ValueError('Prespecify distinct positive logistic inverse-regularization candidates.')
    if type(target_precision) not in (int, float) or not 0 < target_precision < 1:
        raise ValueError('Declare the experimental validation precision target explicitly.')
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    protocol = {'regularization': regularization, 'selection_metric': 'validation_log_loss_per_metric',
        'target_precision': target_precision, 'checkpoint_id': source['checkpoint_id'], 'corpus_id': source['rows_id'],
        'test_loaded': False, 'intensity_weights_changed': False, 'independent_human_gold': False,
        'sampling': 'equal total weight per declared training family, normalized to sample count'}
    (output/'protocol.json').write_text(canonical_json(protocol), encoding='utf-8')
    means, scales, coefficients, intercepts, reports = [], [], [], [], {}
    for metric, *_ in METRICS:
        train, labels = _targets(rows, metric, 'train')
        validation, references = _targets(rows, metric, 'validation')
        if set(labels) != {0., 1.} or set(references) != {0., 1.}:
            raise ValueError(metric + ': both evidence classes are required in train and validation.')
        families = [rows[i].get('family_id', rows[i]['group_id']) for i in train]
        counts = Counter(families)
        weights = np.asarray([len(train) / (len(counts) * counts[f]) for f in families])
        scaler = StandardScaler().fit(vectors[train], sample_weight=weights)
        x, xv = scaler.transform(vectors[train]), scaler.transform(vectors[validation])
        candidates = []
        for inverse_penalty in sorted(regularization):
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                fitted = LogisticRegression(C=inverse_penalty, solver='lbfgs', max_iter=2000, tol=1e-7)
                fitted.fit(x, labels, sample_weight=weights)
            probability = fitted.predict_proba(xv)[:, 1]
            diagnostic = binary_metrics(references, probability)
            candidates.append((diagnostic['log_loss'], inverse_penalty, fitted, diagnostic, probability))
        _, penalty, fitted, diagnostic, probability = min(candidates, key=lambda r: r[:2])
        threshold = choose_empirical_threshold(references, probability, target_precision)
        prevalence = float(np.average(labels, weights=weights))
        reports[metric] = {'regularization': penalty, 'train_rows': len(train), 'train_families': len(counts),
            'validation': diagnostic, 'threshold': threshold, 'validation_selection': selective_metrics(references, probability, threshold),
            'constant_probability': prevalence, 'constant_validation': binary_metrics(references, np.full(len(references), prevalence)),
            'candidates': [{'regularization': c[1], **c[3]} for c in candidates]}
        means.append(scaler.mean_); scales.append(scaler.scale_); coefficients.append(fitted.coef_[0]); intercepts.append(fitted.intercept_[0])
        # Check serialization math before saving; sklearn objects and pickle are not part of the artifact.
        restored = expit(((vectors[validation]-scaler.mean_)/scaler.scale_) @ fitted.coef_[0] + fitted.intercept_[0])
        if not np.allclose(restored, probability, atol=1e-12, rtol=1e-12):
            raise ValueError('Evidence-head serialization changed predictions.')
    np.savez_compressed(output/'weights.npz', means=means, scales=scales, coefficients=coefficients, intercepts=intercepts)
    exposure = {key: sorted({r[key] for r in rows if r.get(key)}) for key in ('example_id','group_id','participant_id','family_id')}
    manifest = {**protocol, 'format': 'frozen_feature_evidence_guard', 'heads': reports, 'exposure': exposure,
        'exposure_texts': sorted({fingerprint([r['category'], ''.join(r['text'].split())]) for r in rows}),
        'feature_width': vectors.shape[1], 'weights_sha256': file_hash(output/'weights.npz'),
        'evidence_calibrated': False, 'status': 'experimental_only', 'eligible_for_primary_dnb': False}
    (output/'manifest.json').write_text(canonical_json(manifest), encoding='utf-8')
    return manifest


class EvidenceGuard:
    def __init__(self, directory, checkpoint_id):
        # PSEUDOCODE: bind each separate evidence head to the exact encoder checkpoint and reject changed parameter files.
        directory = Path(directory)
        self.manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['format'] != 'frozen_feature_evidence_guard' or self.manifest['checkpoint_id'] != checkpoint_id or self.manifest['evidence_calibrated']:
            raise ValueError('Evidence guard does not belong to this exact experimental checkpoint.')
        if file_hash(directory/'weights.npz') != self.manifest['weights_sha256']:
            raise ValueError('Evidence-guard weights changed.')
        with np.load(directory/'weights.npz', allow_pickle=False) as arrays:
            self.means, self.scales, self.coefficients, self.intercepts = (arrays[k].copy() for k in ('means','scales','coefficients','intercepts'))
        expected = (len(METRICS), self.manifest['feature_width'])
        if any(a.shape != expected or not np.isfinite(a).all() for a in (self.means, self.scales, self.coefficients)) or self.intercepts.shape != (len(METRICS),) or not np.isfinite(self.intercepts).all() or np.any(self.scales <= 0):
            raise ValueError('Malformed evidence guard parameters.')
        if set(self.manifest['heads']) != {m[0] for m in METRICS}:
            raise ValueError('Evidence guard must define every metric.')
        for head in self.manifest['heads'].values():
            selective_metrics([0, 1], [0., 1.], head['threshold'])

    def probabilities(self, vectors, category):
        # PSEUDOCODE: transform caller-supplied frozen representations with train-only scaling and fitted linear evidence heads.
        from scipy.special import expit
        vectors = np.asarray(vectors, dtype=np.float64)
        if category not in CATEGORIES:
            raise ValueError('Unsupported evidence category.')
        if vectors.ndim != 2 or vectors.shape[1] != self.manifest['feature_width'] or not np.isfinite(vectors).all():
            raise ValueError('Unexpected frozen feature shape or values.')
        return {metric: expit(((vectors-self.means[i])/self.scales[i]) @ self.coefficients[i] + self.intercepts[i])
                for i, (metric, domain, *_) in enumerate(METRICS) if domain == category}


def evaluate_guard(cache, guard_dir, output_dir, *, regression_only=False):
    # PSEUDOCODE: reject development exposure -> evaluate every frozen per-head policy once -> preserve errors and limitations.
    rows, vectors, _, _, source = load_features(cache)
    if {r['split'] for r in rows} != {'test'}:
        raise ValueError('Frozen guard evaluation accepts a separately sealed test cache only.')
    guard = EvidenceGuard(guard_dir, source['checkpoint_id'])
    manifest = guard.manifest
    for row in rows:
        if any(row.get(key) in values for key, values in manifest['exposure'].items()) or fingerprint([row['category'], ''.join(row['text'].split())]) in manifest['exposure_texts']:
            raise ValueError('Guard test overlaps development exposure.')
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    reports, predictions = {}, []
    for metric, category, *_ in METRICS:
        indices, labels = _targets(rows, metric, 'test')
        if set(labels) != {0., 1.}:
            raise ValueError(metric + ': both test evidence classes must be represented.')
        probability = guard.probabilities(vectors[indices], category)[metric]
        head = manifest['heads'][metric]
        reports[metric] = {**binary_metrics(labels, probability), 'threshold': head['threshold'],
            'selective': selective_metrics(labels, probability, head['threshold']),
            'constant': binary_metrics(labels, np.full(len(labels), head['constant_probability']))}
        families = np.asarray([rows[i].get('family_id', rows[i]['group_id']) for i in indices])
        reports[metric]['by_family'] = {family: {**binary_metrics(labels[families == family], probability[families == family]),
            'selective': selective_metrics(labels[families == family], probability[families == family], head['threshold'])}
            for family in sorted(set(families))}
        predictions.extend({'example_id': rows[i]['example_id'], 'metric': metric, 'reference': float(y), 'probability': float(p)}
                           for i, y, p in zip(indices, labels, probability))
    result = {'per_metric': reports, 'macro_brier': float(np.mean([r['brier'] for r in reports.values()])),
        'macro_log_loss': float(np.mean([r['log_loss'] for r in reports.values()])),
        'macro_constant_brier': float(np.mean([r['constant']['brier'] for r in reports.values()])),
        'macro_constant_log_loss': float(np.mean([r['constant']['log_loss'] for r in reports.values()])),
        'nominal_precision_goal': manifest['target_precision'],
        'heads_below_nominal_goal': [key for key, report in reports.items()
            if report['selective']['precision'] is None or report['selective']['precision'] < manifest['target_precision']],
        'promotion_to_primary_dnb': False,
        'reference': 'explicit_authored_evidence_labels_not_independent_human_gold', 'evidence_calibrated': False,
        'evaluation_role': 'development_regression' if regression_only else 'held_out_construction_test',
        'test_used_for_selection': bool(regression_only), 'checkpoint_id': source['checkpoint_id'], 'test_id': source['rows_id'],
        'guard_id': file_hash(Path(guard_dir)/'manifest.json'), 'intensity_weights_changed': False,
        'interpretation': 'empirical performance on authored held-out scenarios; not a population precision guarantee'}
    (output/'report.json').write_text(canonical_json(result), encoding='utf-8')
    (output/'predictions.json').write_text(canonical_json(predictions), encoding='utf-8')
    return result
