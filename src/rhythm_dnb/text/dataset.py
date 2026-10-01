"""Category-aware tokenization and padded regression batches; no corpus or paths."""
from .schema import CATEGORIES, METRICS

def encode(tokenizer,category,text,max_length):
    # PSEUDOCODE: tokenize category and text together -> reject overlength inputs without truncation.
    result=tokenizer(CATEGORIES[category][0],text,add_special_tokens=True,truncation=False)
    if len(result['input_ids'])>max_length:
        raise ValueError(f'文本与类别共{len(result["input_ids"])}个token，超过{max_length}；不会静默截断，请缩短文本。')
    return result

class ScoreDataset:
    def __init__(self,rows,tokenizer,max_length):
        # PSEUDOCODE: retain reviewed rows -> tokenize every row using the fixed token limit.
        self.rows=rows
        self.features=[encode(tokenizer,r['category'],r['text'],max_length) for r in rows]

    def __len__(self):
        # PSEUDOCODE: return the number of original corpus rows.
        return len(self.rows)

    def __getitem__(self,index):
        # PSEUDOCODE: preserve sampler weights so empty distributed shards contribute no duplicate loss.
        weight = 1.
        if isinstance(index, tuple):
            index, weight = index
        row=self.rows[index]
        return self.features[index],row['category'],row['scores'],weight

class Collator:
    def __init__(self,tokenizer):
        # PSEUDOCODE: retain the tokenizer used to pad each batch consistently.
        self.tokenizer=tokenizer

    def __call__(self,items):
        # PSEUDOCODE: pad token sequences -> align category labels -> preserve per-sample loss weights.
        import torch
        features,categories,scores,weights=zip(*items)
        batch=dict(self.tokenizer.pad(list(features),padding=True,return_tensors='pt'))
        batch['categories']=list(categories)
        batch['labels']=torch.tensor([[score.get(m[0],0)/100 for m in METRICS] for score in scores],dtype=torch.float32)
        batch['sample_weights']=torch.tensor(weights,dtype=torch.float32)
        return batch
