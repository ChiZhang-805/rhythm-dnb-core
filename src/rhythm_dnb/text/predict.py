"""Continuous semantic regression with explicit pinned checkpoints and no training import."""

import math
import threading
from .schema import CATEGORIES, validate_input
from .dataset import encode
from .checkpoint import inspect_checkpoint, load_checkpoint, device_for
from .evidence import qualified_scores


class TextPredictor:
    def __init__(self, checkpoint, *, device='auto', base_path=None, allow_experimental=False, guard_dir=None, evidence_checkpoint=None, precision='auto'):
        # PSEUDOCODE: inspect a caller-selected local checkpoint; defer accelerator allocation until inference.
        self.checkpoint = checkpoint
        self.manifest, self.model_identity = inspect_checkpoint(checkpoint, allow_experimental=allow_experimental)
        if precision not in ('auto', 'fp32', 'bf16', 'fp16'):
            raise ValueError('Unknown inference precision.')
        if precision != 'auto' and (not allow_experimental or self.manifest['purpose'] != 'experimental_semantic_regression'):
            raise ValueError('Changing inference precision requires an explicitly experimental degree model; formal calibration is unchanged.')
        if precision != 'auto' and guard_dir is not None:
            raise ValueError('The frozen-feature guard has not been validated with changed inference precision.')
        self.precision = precision
        self.allow_experimental = allow_experimental
        self.device_name = device
        self.base_path = base_path
        self._loaded = False
        self._lock = threading.Lock()
        self.guard = None
        self.evidence_adapter = None
        if guard_dir is not None and evidence_checkpoint is not None:
            raise ValueError('Select one experimental evidence method at a time.')
        if evidence_checkpoint is not None:
            from .evidence_adapter import EvidenceAdapter
            if not allow_experimental or self.manifest['purpose'] != 'experimental_semantic_regression':
                raise ValueError('An experimental adapter cannot qualify formal DNB inputs.')
            self.evidence_adapter = EvidenceAdapter(evidence_checkpoint, self.model_identity, base_path, device)
        if guard_dir is not None:
            from .guard import EvidenceGuard
            if not allow_experimental or self.manifest['purpose'] != 'experimental_semantic_regression':
                raise ValueError('An experimental guard cannot qualify formal DNB inputs.')
            self.guard = EvidenceGuard(guard_dir, self.model_identity)

    def predict(self, category, text):
        # PSEUDOCODE: validate category/text -> encode without truncation -> return unrounded bounded intensities.
        import torch
        from .inference import resolve_precision, inference_profile
        category, text = validate_input(category, text)
        with self._lock:
            if not self._loaded:
                self.device = device_for(self.device_name)
                dtype = resolve_precision(self.precision, self.device, self.manifest['storage'])
                with torch.autocast(self.device.type, enabled=False):
                    model, tokenizer, manifest, identity = load_checkpoint(self.checkpoint, base_path=self.base_path, device=self.device,
                                                                           dtype=dtype, allow_experimental=self.allow_experimental)
                if identity != self.model_identity:
                    raise ValueError('Checkpoint changed after predictor initialization.')
                self.model, self.tokenizer = model.eval(), tokenizer
                self.inference_dtype = dtype
                self._loaded = True
            encoded = encode(self.tokenizer, category, text, self.manifest['config']['max_length'])
            profile = inference_profile(self.model_identity, self.manifest, self.model, self.device, self.inference_dtype)
            with torch.inference_mode(), torch.autocast(self.device.type, enabled=False):
                outputs = self.model(**{k: torch.tensor([v], device=self.device) for k, v in encoded.items()},
                                     return_representation=self.guard is not None)
                values = (outputs[category][0].float().cpu() * 100).tolist()
                evidence = dict(zip(CATEGORIES[category][1], torch.sigmoid(outputs['_evidence'][category][0].float()).cpu().tolist()))
            if any(not math.isfinite(v) or not 0 <= v <= 100 for v in values):
                raise ValueError('Nonfinite/out-of-range semantic prediction.')
            estimates = dict(zip(CATEGORIES[category][1], values))
            if self.manifest['purpose'] == 'experimental_semantic_regression':
                diagnostic = {}
                if self.guard is not None:
                    probabilities = self.guard.probabilities(outputs['_representation'].float().cpu().numpy(), category)
                    diagnostic = {'evidence_estimates': {k: float(v[0]) for k, v in probabilities.items()},
                        'evidence_guard_kind': 'experimental_frozen_feature_classifier',
                        'experimental_acceptance': {k: self.guard.manifest['heads'][k]['threshold'] is not None and
                            bool(v[0] >= self.guard.manifest['heads'][k]['threshold']) for k, v in probabilities.items()},
                        'acceptance_certified': False}
                elif self.evidence_adapter is not None:
                    diagnostic = self.evidence_adapter.predict(category, text)
                return {**diagnostic, 'category': category, 'scores': {k: None for k in estimates}, 'normalized': {k: None for k in estimates},
                        'estimates': estimates, 'reasons': {k: 'uncalibrated_experimental_text_evidence' for k in estimates},
                        'model_identity': self.model_identity, 'run_id': self.manifest.get('run_id'),
                        'inference_profile': profile,
                        'score_kind': 'experimental_semantic_reference_estimate', 'evidence_calibrated': False,
                        'evidence_trained': self.guard is not None or self.evidence_adapter is not None or self.manifest.get('evidence_trained', False),
                        'independent_human_gold': False, 'eligible_for_primary_dnb': False}
            scores, reasons = qualified_scores(estimates, evidence, self.manifest.get('evidence_calibration'))
            return {'category': category, 'scores': scores, 'normalized': {k: v / 100 if v is not None else None for k, v in scores.items()},
                    'estimates': estimates, 'evidence': evidence, 'reasons': reasons,
                    'model_identity': self.model_identity, 'run_id': self.manifest.get('run_id'),
                    'inference_profile': profile,
                    'score_kind': 'text_semantic_estimate_not_probability'}
