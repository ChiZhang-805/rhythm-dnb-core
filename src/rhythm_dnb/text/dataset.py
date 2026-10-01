"""Category-aware tokenization and padded regression batches; no corpus or paths."""
from .schema import CATEGORIES, METRICS

def encode(tokenizer,category,text,max_length):
    # PSEUDOCODE: validate inputs -> encode -> return the fixed semantic contract.
    result=tokenizer(CATEGORIES[category][0],text,add_special_tokens=True,truncation=False)
    if len(result['input_ids'])>max_length:
        raise ValueError(f'文本与类别共{len(result["input_ids"])}个token，超过{max_length}；不会静默截断，请缩短文本。')
    return result

class ScoreDataset:
    def __init__(self,rows,tokenizer,max_length):
        # PSEUDOCODE: validate inputs -> init -> return the fixed semantic contract.
        self.rows=rows
        self.features=[encode(tokenizer,r['category'],r['text'],max_length) for r in rows]

    def __len__(self):
        # PSEUDOCODE: validate inputs -> len -> return the fixed semantic contract.
        return len(self.rows)

    def __getitem__(self,index):
        # PSEUDOCODE: validate inputs -> getitem -> return the fixed semantic contract.
        row=self.rows[index]
        return self.features[index],row['category'],row['scores']

class Collator:
    def __init__(self,tokenizer):
        # PSEUDOCODE: validate inputs -> init -> return the fixed semantic contract.
        self.tokenizer=tokenizer

    def __call__(self,items):
        # PSEUDOCODE: validate inputs -> call -> return the fixed semantic contract.
        import torch
        features,categories,scores=zip(*items)
        batch=dict(self.tokenizer.pad(list(features),padding=True,return_tensors='pt'))
        batch['categories']=list(categories)
        batch['labels']=torch.tensor([[score.get(m[0],0)/100 for m in METRICS] for score in scores],dtype=torch.float32)
        return batch
