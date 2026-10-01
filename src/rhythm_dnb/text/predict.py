"""Continuous semantic regression with explicit pinned checkpoints and no training import."""

import math
import threading
from .schema import CATEGORIES, validate_input
from .dataset import encode
from .checkpoint import inspect_checkpoint, load_checkpoint, device_for


class TextPredictor:
    def __init__(self, checkpoint, *, device='cpu'):
        # PSEUDOCODE: inspect a caller-selected local checkpoint; defer accelerator allocation until inference.
        self.checkpoint = checkpoint
        self.manifest, self.model_identity = inspect_checkpoint(checkpoint)
        self.device_name = device
        self._loaded = False
        self._lock = threading.Lock()

    def predict(self, category, text):
        # PSEUDOCODE: validate category/text -> encode without truncation -> return unrounded bounded intensities.
        import torch
        category, text = validate_input(category, text)
        with self._lock:
            if not self._loaded:
                model, tokenizer, manifest, identity = load_checkpoint(self.checkpoint)
                if identity != self.model_identity:
                    raise ValueError('Checkpoint changed after predictor initialization.')
                self.device = device_for(self.device_name)
                self.model, self.tokenizer = model.to(self.device).eval(), tokenizer
                self._loaded = True
            encoded = encode(self.tokenizer, category, text, self.manifest['config']['max_length'])
            with torch.inference_mode():
                outputs = self.model(**{k: torch.tensor([v], device=self.device) for k, v in encoded.items()})
                values = (outputs[category][0].float().cpu() * 100).tolist()
            if any(not math.isfinite(v) or not 0 <= v <= 100 for v in values):
                raise ValueError('Nonfinite/out-of-range semantic prediction.')
            scores = dict(zip(CATEGORIES[category][1], values))
            return {'category': category, 'scores': scores, 'normalized': {k: v / 100 for k, v in scores.items()},
                    'model_identity': self.model_identity, 'run_id': self.manifest.get('run_id'),
                    'score_kind': 'text_semantic_estimate_not_probability'}
