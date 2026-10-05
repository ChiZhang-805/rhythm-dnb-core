"""A new training stage keeps weights, never silently reuses an old optimizer or holdout."""

from .checkpoint import inspect_checkpoint
from ..provenance import fingerprint


def validate_initialization(folder, partitions, corpus, base_id, config, experimental):
    # PSEUDOCODE: bind the prior model -> preserve architecture -> reject previously seen validation people/texts.
    if not experimental:
        raise ValueError('Continuation currently requires the explicit experimental workflow.')
    previous, identity = inspect_checkpoint(folder, allow_experimental=True)
    if previous['purpose'] != 'experimental_semantic_regression' or previous['base_id'] != base_id:
        raise ValueError('Initial checkpoint purpose or base differs.')
    for key in ('model_id', 'revision', 'quantization', 'lora_rank', 'lora_alpha', 'dropout'):
        if previous['config'][key] != config[key]:
            raise ValueError('Continuation architecture mismatch: ' + key)
    expected = corpus.get('initial_checkpoint_id')
    if expected != identity:
        raise ValueError('Expanded corpus must be sealed against the exact initial checkpoint.')
    seen = previous['dataset']
    ancestors = previous.get('initialization') or {}
    exposure = {key: sorted(set(values) | set(ancestors.get('exposure_groups', {}).get(key, [])))
                for key, values in seen['development_groups'].items()}
    exposed_texts = sorted(set(seen['development_text_hashes']) | set(ancestors.get('exposure_text_hashes', [])))
    for key, values in exposure.items():
        if set(values) & {r[key] for r in partitions['validation'] if r.get(key)}:
            raise ValueError('Continuation validation overlaps previous development: ' + key)
    hashes = {fingerprint(''.join(r['text'].split())) for r in partitions['validation']}
    if hashes & set(exposed_texts):
        raise ValueError('Continuation validation text was previously seen.')
    return {'checkpoint_id': identity, 'previous_corpus_id': seen['id'],
            'previous_epoch': previous['epoch'], 'optimizer_reset': True,
            'exposure_groups': exposure, 'exposure_text_hashes': exposed_texts,
            'ancestor_checkpoint_ids': [*ancestors.get('ancestor_checkpoint_ids', []), identity]}
