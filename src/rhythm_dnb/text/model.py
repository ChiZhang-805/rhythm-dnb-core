"""Chinese encoder with fixed category heads and independently bounded scores."""

import torch
from torch import nn
from transformers import BertConfig,BertModel

from .schema import CATEGORIES,METRICS


class ScoringModel(nn.Module):
    def __init__(self,encoder,dropout=0.1):
        # PSEUDOCODE: attach encoder -> freeze unused pooler -> create one regression head per category.
        super().__init__()
        self.encoder=encoder
        if encoder.pooler is not None:
            encoder.pooler.requires_grad_(False)  # The predictor uses CLS hidden state, not the pooler.
        self.dropout=nn.Dropout(dropout)
        self.heads=nn.ModuleDict({category:nn.Linear(encoder.config.hidden_size,len(keys))
                                  for category,(_,keys) in CATEGORIES.items()})

    @classmethod
    def pretrained(cls,path,dropout=0.1):
        # PSEUDOCODE: load pinned local encoder weights -> attach trainable category heads.
        return cls(BertModel.from_pretrained(path,local_files_only=True),dropout)

    @classmethod
    def from_config(cls,path,dropout=0.1):
        # PSEUDOCODE: rebuild the recorded local encoder architecture before checkpoint restoration.
        return cls(BertModel(BertConfig.from_pretrained(path,local_files_only=True)),dropout)

    def forward(self,input_ids,attention_mask,token_type_ids=None):
        # PSEUDOCODE: read CLS representation -> apply dropout -> bound every category score with sigmoid.
        hidden=self.encoder(input_ids=input_ids,attention_mask=attention_mask,
                            token_type_ids=token_type_ids).last_hidden_state[:,0]
        hidden=self.dropout(hidden)
        return {name:torch.sigmoid(head(hidden)) for name,head in self.heads.items()}


def regression_loss(outputs,labels,categories,delta=0.1,*,reduction='mean'):
    # Average within each sample first so five-output tasks do not dominate one-output tasks.
    # PSEUDOCODE: compute category-specific Huber errors -> average heads within each sample -> preserve DDP graph links.
    losses=[None] * len(categories)
    for category,(_,keys) in CATEGORIES.items():
        indices=[i for i,c in enumerate(categories) if c==category]
        if indices:
            columns=[next(i for i,m in enumerate(METRICS) if m[0]==key) for key in keys]
            target=labels[indices][:,columns]
            values=nn.functional.huber_loss(outputs[category][indices].float(),target.float(),
                                           delta=delta,reduction='none').mean(dim=1)
            for index,value in zip(indices,values):
                losses[index]=value
    if any(x is None for x in losses) or not categories or reduction not in ('mean','none'):
        raise ValueError('Every training sample needs exactly one supported category.')
    # Every rank keeps all head parameters in the graph, including categories absent from its shard.
    result=torch.stack(losses) + sum(value.float().sum() * 0 for value in outputs.values())
    return result.mean() if reduction == 'mean' else result
