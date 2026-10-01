"""Chinese encoder with fixed category heads and independently bounded scores."""

import torch
from torch import nn
from transformers import BertConfig,BertModel

from .schema import CATEGORIES,METRICS


class ScoringModel(nn.Module):
    def __init__(self,encoder,dropout=0.1):
        # PSEUDOCODE: validate inputs -> init -> return the fixed semantic contract.
        super().__init__()
        self.encoder=encoder
        self.dropout=nn.Dropout(dropout)
        self.heads=nn.ModuleDict({category:nn.Linear(encoder.config.hidden_size,len(keys))
                                  for category,(_,keys) in CATEGORIES.items()})

    @classmethod
    def pretrained(cls,path,dropout=0.1):
        # PSEUDOCODE: validate inputs -> pretrained -> return the fixed semantic contract.
        return cls(BertModel.from_pretrained(path,local_files_only=True),dropout)

    @classmethod
    def from_config(cls,path,dropout=0.1):
        # PSEUDOCODE: validate inputs -> from config -> return the fixed semantic contract.
        return cls(BertModel(BertConfig.from_pretrained(path,local_files_only=True)),dropout)

    def forward(self,input_ids,attention_mask,token_type_ids=None):
        # PSEUDOCODE: validate inputs -> forward -> return the fixed semantic contract.
        hidden=self.encoder(input_ids=input_ids,attention_mask=attention_mask,
                            token_type_ids=token_type_ids).last_hidden_state[:,0]
        hidden=self.dropout(hidden)
        return {name:torch.sigmoid(head(hidden)) for name,head in self.heads.items()}


def regression_loss(outputs,labels,categories,delta=0.1):
    # Average within each sample first so five-output tasks do not dominate one-output tasks.
    # PSEUDOCODE: validate inputs -> regression loss -> return the fixed semantic contract.
    losses=[]
    for category,(_,keys) in CATEGORIES.items():
        indices=[i for i,c in enumerate(categories) if c==category]
        if indices:
            columns=[next(i for i,m in enumerate(METRICS) if m[0]==key) for key in keys]
            target=labels[indices][:,columns]
            losses.append(nn.functional.huber_loss(outputs[category][indices],target,
                                                  delta=delta,reduction='none').mean(dim=1))
    if sum(x.numel() for x in losses)!=len(categories) or not categories:
        raise ValueError('Every training sample needs exactly one supported category.')
    return torch.cat(losses).mean()
