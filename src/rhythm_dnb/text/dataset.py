"""Category-aware tokenization and padded regression batches; no corpus or paths."""
from .schema import CATEGORIES, METRICS

def encode(tokenizer,category,text,max_length, *, offsets=False):
    # PSEUDOCODE: tokenize category and text together -> reject overlength inputs without truncation.
    if 'token_type_ids' in tokenizer.model_input_names:
        result=tokenizer(CATEGORIES[category][0],text,add_special_tokens=True,truncation=False, return_offsets_mapping=offsets)
        source_start = 0
    else:
        names='、'.join(next(m[2] for m in METRICS if m[0]==key) for key in CATEGORIES[category][1])
        prompt=f'阅读真实的{CATEGORIES[category][0]}记录，评估{names}。信息不足的指标应保留未知。\n记录：{text}\n指标评分：'
        result=tokenizer(prompt,add_special_tokens=True,truncation=False, return_offsets_mapping=offsets)
        source_start = len(prompt) - len(text) - len('\n指标评分：')
    if len(result['input_ids'])>max_length:
        raise ValueError(f'文本与类别共{len(result["input_ids"])}个token，超过{max_length}；不会静默截断，请缩短文本。')
    if offsets:
        mapping = result.pop('offset_mapping')
        sequences = result.sequence_ids()
        result['_text_offsets'] = [(a - source_start, b - source_start) if b > a
            and (sequences[i] == 1 if 'token_type_ids' in tokenizer.model_input_names else source_start <= a < b <= source_start + len(text))
            else (0, 0) for i, (a, b) in enumerate(mapping)]
    return result

class ScoreDataset:
    def __init__(self,rows,tokenizer,max_length, *, explicit_evidence_only=False):
        # PSEUDOCODE: retain reviewed rows -> tokenize every row using the fixed token limit.
        self.rows=rows
        self.explicit_evidence_only = explicit_evidence_only
        self.features=[]
        for row in rows:
            try:
                feature = encode(tokenizer,row['category'],row['text'],max_length, offsets=bool(row.get('scope_targets')))
                from .annotations import scope_token_targets
                feature['_scope_labels'] = scope_token_targets(row, feature.pop('_text_offsets', []))
                self.features.append(feature)
            except ValueError as error:
                raise ValueError(f"example_id={row.get('example_id', '<unknown>')}: {error}") from error

    def __len__(self):
        # PSEUDOCODE: return the number of original corpus rows.
        return len(self.rows)

    def __getitem__(self,index):
        # PSEUDOCODE: preserve sampler weights so empty distributed shards contribute no duplicate loss.
        weight = 1.
        if isinstance(index, tuple):
            index, weight = index
        row=self.rows[index]
        from .labels import evidence_target
        return self.features[index],row['category'],row['scores'],weight, {
            k: None if self.explicit_evidence_only and 'label_states' not in row else evidence_target(row, k) for k in row['scores']}

class Collator:
    def __init__(self,tokenizer):
        # PSEUDOCODE: retain the tokenizer used to pad each batch consistently.
        self.tokenizer=tokenizer

    def __call__(self,items):
        # PSEUDOCODE: pad token sequences -> align category labels -> preserve per-sample loss weights.
        import torch
        features,categories,scores,weights,evidence=zip(*items)
        batch=dict(self.tokenizer.pad([{k:v for k,v in f.items() if k != '_scope_labels'} for f in features],padding=True,return_tensors='pt'))
        batch['categories']=list(categories)
        batch['labels']=torch.tensor([[float('nan') if score.get(m[0]) is None else score[m[0]]/100 for m in METRICS] for score in scores],dtype=torch.float32)
        batch['sample_weights']=torch.tensor(weights,dtype=torch.float32)
        batch['evidence_labels']=torch.tensor([[float('nan') if target.get(m[0]) is None else target[m[0]] for m in METRICS] for target in evidence],dtype=torch.float32)
        scope = torch.full((len(items), batch['input_ids'].shape[1], len(METRICS)), float('nan'))
        for i, feature in enumerate(features):
            labels = feature['_scope_labels']
            if labels:
                offset = scope.shape[1] - len(labels) if self.tokenizer.padding_side == 'left' else 0
                scope[i, offset:offset + len(labels)] = torch.tensor(labels)
        batch['scope_labels'] = scope
        return batch
