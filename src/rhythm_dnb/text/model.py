"""Qwen text representations with bounded intensity and evidence heads; MacBERT is a comparator."""

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from .schema import CATEGORIES, METRICS


class ScoringModel(nn.Module):
    def __init__(self, encoder, dropout=0.1, *, scope_supervision=False):
        # PSEUDOCODE: attach independent score/evidence heads -> freeze any unused encoder pooler.
        super().__init__()
        self.encoder = encoder
        if getattr(encoder, 'pooler', None) is not None:
            encoder.pooler.requires_grad_(False)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({c: nn.Linear(encoder.config.hidden_size, len(keys)) for c, (_, keys) in CATEGORIES.items()})
        self.evidence_heads = nn.ModuleDict({c: nn.Linear(encoder.config.hidden_size, len(keys)) for c, (_, keys) in CATEGORIES.items()})
        self.scope_head = nn.Linear(3 * encoder.config.hidden_size, len(METRICS)) if scope_supervision else None

    @classmethod
    def pretrained(cls, path, dropout=0.1, *, config=None, device=None, dtype=None, training=True):
        # PSEUDOCODE: load local backbone -> optionally quantize frozen Qwen -> attach trainable LoRA and heads.
        options = {'local_files_only': True, 'trust_remote_code': False}
        config = config or {}
        device = torch.device(device or 'cpu')
        architecture = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        qwen = architecture.model_type == 'qwen3'
        quantized = qwen and config.get('quantization') == 'nf4'
        if qwen:
            options.update(torch_dtype=dtype or torch.float32, attn_implementation='sdpa')
        if quantized:
            if device.type != 'cuda':
                raise ValueError('NF4 requires CUDA; select quantization=none explicitly for CPU experiments.')
            from transformers import BitsAndBytesConfig
            options.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype or torch.float32),
                           device_map={'': device.index if device.index is not None else torch.cuda.current_device()})
        encoder = AutoModel.from_pretrained(path, **options)
        if qwen:
            encoder.config.use_cache = False
            if quantized:
                from peft import prepare_model_for_kbit_training
                # Keep embedding/residual precision identical during training and checkpoint inference.
                encoder = prepare_model_for_kbit_training(encoder, use_gradient_checkpointing=False)
            if training:
                from peft import LoraConfig, get_peft_model
                encoder = get_peft_model(encoder, LoraConfig(r=config['lora_rank'], lora_alpha=config['lora_alpha'],
                    lora_dropout=config['dropout'], target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'], bias='none'))
        model = cls(encoder, dropout, scope_supervision=config.get('scope_loss_weight', 0) > 0)
        model.heads.to(device); model.evidence_heads.to(device)
        if model.scope_head is not None:
            model.scope_head.to(device)
        if not quantized:
            model.to(device)
        return model

    @classmethod
    def from_config(cls, path, dropout=0.1, *, scope_supervision=False):
        # PSEUDOCODE: construct a local unquantized architecture before restoring full comparator weights.
        return cls(AutoModel.from_config(AutoConfig.from_pretrained(path, local_files_only=True), trust_remote_code=False), dropout, scope_supervision=scope_supervision)

    def representation(self, input_ids, attention_mask, token_type_ids=None):
        # PSEUDOCODE: encode valid tokens -> return full text and token representations without changing scoring weights.
        qwen = self.encoder.config.model_type == 'qwen3'
        options = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if qwen:
            options.update(position_ids=(attention_mask.long().cumsum(-1) - 1).clamp_min(0), use_cache=False)
        elif token_type_ids is not None:
            options['token_type_ids'] = token_type_ids
        hidden = self.encoder(**options).last_hidden_state
        if qwen:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
            last = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
            if (last < 0).any():
                raise ValueError('A text sequence cannot consist entirely of padding.')
            pooled = hidden[torch.arange(len(hidden), device=hidden.device), last]
        else:
            pooled = hidden[:, 0]
        return pooled.float(), hidden

    def forward(self, input_ids, attention_mask, token_type_ids=None, *, return_representation=False):
        # PSEUDOCODE: reuse the exact frozen representation -> score independent intensity, evidence and optional scope heads.
        full_context, hidden = self.representation(input_ids, attention_mask, token_type_ids)
        pooled = self.dropout(full_context)
        result = {name: torch.sigmoid(head(pooled)) for name, head in self.heads.items()}
        result['_evidence'] = {name: head(pooled) for name, head in self.evidence_heads.items()}
        if return_representation:
            result['_representation'] = full_context
        if self.scope_head is not None:
            # A causal token alone cannot see a later attribution or reversal; condition it on the complete text.
            tokens = nn.functional.layer_norm(hidden.float(), (hidden.shape[-1],))
            context = nn.functional.layer_norm(full_context, (hidden.shape[-1],)).unsqueeze(1).expand_as(tokens)
            result['_scope'] = self.scope_head(torch.cat((tokens, context, tokens * context), dim=-1))
        return result


def regression_loss(outputs, labels, categories, delta=0.1, *, reduction='mean', evidence_weight=0.1, evidence_labels=None,
                    scope_labels=None, scope_weight=0.):
    # PSEUDOCODE: mask unknown intensities -> learn evidence separately -> average per sample with all DDP heads linked.
    losses = [None] * len(categories)
    for category, (_, keys) in CATEGORIES.items():
        indices = [i for i, c in enumerate(categories) if c == category]
        if indices:
            columns = [next(i for i, m in enumerate(METRICS) if m[0] == key) for key in keys]
            target = labels[indices][:, columns]
            known = torch.isfinite(target)
            errors = nn.functional.huber_loss(outputs[category][indices].float(), target.nan_to_num(), delta=delta, reduction='none')
            values = (errors * known).sum(dim=1) / known.sum(dim=1).clamp_min(1)
            evidence = (torch.where(known, 1., float('nan')) if evidence_labels is None else evidence_labels[indices][:, columns])
            reviewed = torch.isfinite(evidence)
            evidence_errors = nn.functional.binary_cross_entropy_with_logits(
                outputs['_evidence'][category][indices].float(), evidence.nan_to_num(), reduction='none')
            values += evidence_weight * (evidence_errors * reviewed).sum(dim=1) / reviewed.sum(dim=1).clamp_min(1)
            for index, value in zip(indices, values):
                losses[index] = value
    if any(x is None for x in losses) or not categories or reduction not in ('mean', 'none'):
        raise ValueError('Every training sample needs exactly one supported category.')
    linked = list(outputs['_evidence'].values()) + [outputs[c] for c in CATEGORIES]
    result = torch.stack(losses) + sum(value.float().sum() * 0 for value in linked)
    if '_scope' in outputs:
        result = result + outputs['_scope'].sum(dim=(1, 2)) * 0
        if scope_weight and scope_labels is not None:
            known_scope = torch.isfinite(scope_labels)
            scope_errors = nn.functional.binary_cross_entropy_with_logits(outputs['_scope'].float(), scope_labels.nan_to_num(), reduction='none')
            result += scope_weight * (scope_errors * known_scope).sum(dim=(1, 2)) / known_scope.sum(dim=(1, 2)).clamp_min(1)
    elif scope_weight:
        raise ValueError('Scope supervision was requested without a scope head.')
    return result.mean() if reduction == 'mean' else result
