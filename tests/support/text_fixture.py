"""Small artificial model and labeled records for execution tests, never study training."""

import json
from pathlib import Path
from rhythm_dnb.provenance import file_hash
from rhythm_dnb.text.schema import CATEGORIES


def create_text_fixture(root):
    # PSEUDOCODE: build a deterministic local encoder and disjoint artificial corpus without downloads.
    import torch
    from transformers import BertConfig, BertModel, BertTokenizer
    root = Path(root)
    base = root / 'base'
    base.mkdir()
    torch.manual_seed(8)
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + list(dict.fromkeys('今天的心情描述压力睡眠饮食社交绪'))
    (base / 'vocab.txt').write_text('\n'.join(vocab), encoding='utf-8')
    tokenizer = BertTokenizer(vocab_file=str(base / 'vocab.txt'))
    tokenizer.save_pretrained(base)
    model = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32, max_position_embeddings=128,
                                hidden_dropout_prob=0., attention_probs_dropout_prob=0.))
    model.save_pretrained(base)
    config = {'model_id': 'hfl/chinese-macbert-base', 'revision': 'a' * 40, 'seed': 37,
              'max_length': 64, 'batch_size': 2, 'gradient_accumulation': 4, 'epochs': 1,
              'patience': 2, 'encoder_lr': .001, 'head_lr': .001, 'weight_decay': .01,
              'warmup_ratio': 0., 'huber_delta': .1, 'dropout': 0., 'max_grad_norm': 1.,
              'threads': 1, 'device': 'cpu', 'precision': 'fp32', 'gradient_checkpointing': True, 'num_workers': 0}
    receipt = {'model_id': config['model_id'], 'revision': config['revision'],
               'files': {p.name: file_hash(p) for p in base.iterdir() if p.is_file()}}
    (base / 'download.json').write_text(json.dumps(receipt), encoding='utf-8')
    rows = []
    for split in ('train', 'validation', 'test'):
        for index, (category, (_, keys)) in enumerate(CATEGORIES.items()):
            identity = split + category
            rows.append({'example_id': identity, 'category': category, 'text': '今天的心情描述' + identity,
                         'scores': {key: float(20 + index * 13) for key in keys}, 'split': split, 'group_id': identity,
                         'participant_id': identity, 'origin': 'observed', 'review_status': 'accepted',
                         'annotation_evidence_id': 'artificial-unit-test-only'})
    return base, config, rows
