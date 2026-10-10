"""Resume compact, disjoint GPU scoring shards for a frozen text inventory."""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import time

from rhythm_dnb.provenance import canonical_json, fingerprint, file_hash
from rhythm_dnb.research.experiment_data import save_json
from rhythm_dnb.text.predict import TextPredictor
from rhythm_dnb.text.dataset import encode
from rhythm_dnb.text.schema import CATEGORIES


def predict_many(predictor, tasks):
    # PSEUDOCODE: use the same encoder, masks and scoring heads; batch padding never becomes an input token.
    import torch
    if not predictor._loaded:
        predictor.predict(**tasks[0])
    tokens = [encode(predictor.tokenizer, t['category'], t['text'], predictor.manifest['config']['max_length']) for t in tasks]
    batch = predictor.tokenizer.pad(tokens, padding=True, return_tensors='pt')
    with torch.inference_mode(), torch.autocast(predictor.device.type, enabled=False):
        output = predictor.model(**{k: v.to(predictor.device) for k, v in batch.items()})
    results = []
    for i, task in enumerate(tasks):
        values = (output[task['category']][i].float().cpu()*100).tolist()
        if any(not math.isfinite(v) or not 0 <= v <= 100 for v in values):
            raise ValueError('Invalid batched score.')
        results.append(dict(zip(CATEGORIES[task['category']][1], values)))
    return results


def check_batch(predictor, tasks, size):
    # PSEUDOCODE: test short/medium/long texts in every category against singleton inference before accelerating.
    selected = []
    for category in sorted({t['category'] for _, t in tasks}):
        ordered = sorted([(k, t) for k, t in tasks if t['category'] == category], key=lambda item: (len(item[1]['text']), item[0]))
        selected.extend(ordered[i] for i in sorted({0, len(ordered)//3, 2*len(ordered)//3, len(ordered)-1}))
    singles = [predictor.predict(**task) for _, task in selected]
    start = time.monotonic(); batched = []
    for i in range(0, len(selected), size):
        batched.extend(predict_many(predictor, [t for _, t in selected[i:i+size]]))
    error = max(abs(v - batched[i][k]) for i, result in enumerate(singles) for k, v in result['estimates'].items())
    if error > .001:
        raise ValueError('Batch equivalence exceeds 0.001 points: ' + str(error))
    return {'batch_size': size, 'checked_task_ids': [k for k, _ in selected], 'maximum_absolute_difference': error,
            'tolerance_points': .001, 'tolerance_basis': 'One hundred-thousandth of the 0-100 score range; numerical equivalence only.',
            'batch_seconds': time.monotonic()-start, 'singleton_profile': singles[0]['inference_profile']}


def score(tasks_path, checkpoint, base, model_id, output, shard, shards, batch_size=1, skip_cache=None, verify_only=False):
    # PSEUDOCODE: lock one SQLite cache -> verify model and task identities -> resume only unfinished texts.
    import torch
    torch.set_num_threads(2)
    if not 0 <= shard < shards or batch_size < 1:
        raise ValueError('Invalid shard assignment.')
    inventory = json.loads(Path(tasks_path).read_text(encoding='utf-8'))
    if fingerprint(inventory['tasks']) != inventory['tasks_id']:
        raise ValueError('Text inventory changed.')
    tasks = [(key, task) for i, (key, task) in enumerate(sorted(inventory['tasks'].items())) if i % shards == shard]
    predictor = TextPredictor(checkpoint, device='cuda', base_path=base,
                              allow_experimental=True, precision='fp32')
    if predictor.model_identity != model_id:
        raise ValueError('Wrong selected checkpoint.')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if verify_only:
        receipt = check_batch(predictor, tasks, batch_size)
        save_json(output / f'shard-{shard}-batch-check.json', receipt)
        print(canonical_json(receipt), flush=True)
        return
    skipped, skipped_hash = set(), None
    if skip_cache is not None:
        old = Path(skip_cache) / f'shard-{shard}.sqlite'
        with closing(sqlite3.connect(old.resolve().as_uri()+'?mode=ro', uri=True)) as saved:
            metadata = dict(saved.execute('SELECT key,value FROM metadata'))
            previous = json.loads(metadata['binding'])
            if any(previous[k] != v for k, v in {'tasks_id': inventory['tasks_id'], 'model_id': model_id,
                    'shard': shard, 'shards': shards, 'precision': 'fp32'}.items()):
                raise ValueError('Skipped cache belongs to another task/model.')
            skipped = {r[0] for r in saved.execute('SELECT task_id FROM predictions')}
        skipped_hash = file_hash(old)
        if not skipped <= {k for k, _ in tasks}:
            raise ValueError('Skipped cache has extra tasks.')
        tasks = [(k, t) for k, t in tasks if k not in skipped]
    binding = {'tasks_id': inventory['tasks_id'], 'model_id': model_id, 'shard': shard,
               'shards': shards, 'precision': 'fp32', 'batch_size': batch_size}
    if skipped_hash:
        binding.update(skipped_cache_sha256=skipped_hash, skipped_tasks=len(skipped))
    path = output / f'shard-{shard}.sqlite'
    with closing(sqlite3.connect(path, timeout=0)) as db, db:
        db.execute('PRAGMA locking_mode=EXCLUSIVE')
        db.execute('BEGIN EXCLUSIVE')
        db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS predictions (task_id TEXT PRIMARY KEY, estimates TEXT NOT NULL)')
        existing = dict(db.execute('SELECT key,value FROM metadata'))
        if existing and json.loads(existing['binding']) != binding:
            raise ValueError('Cache belongs to another inference plan.')
        if not existing:
            db.execute('INSERT INTO metadata VALUES (?,?)', ('binding', canonical_json(binding)))
        db.commit()
        completed = {r[0] for r in db.execute('SELECT task_id FROM predictions')}
        if not completed <= {key for key, _ in tasks}:
            raise ValueError('Unexpected cached task.')
        profile = json.loads(existing['profile']) if 'profile' in existing else None
        if tasks and batch_size > 1:
            audit = check_batch(predictor, tasks, batch_size)
            audit_file = output/f'shard-{shard}-batch-check.json'
            if not audit_file.exists():
                save_json(audit_file, audit)
        current = predictor.predict(**tasks[0][1])['inference_profile'] if tasks else profile
        if tasks and batch_size > 1:
            current = {k: v for k, v in current.items() if k != 'id'}
            current.update(batch_size=batch_size, padding='right_padding_to_longest_in_batch')
            current['id'] = fingerprint(current)
        if tasks:
            if profile is not None and profile != current:
                raise ValueError('Inference runtime changed; preserve the old shard.')
            if profile is None:
                profile = current
                db.execute('INSERT INTO metadata VALUES (?,?)', ('profile', canonical_json(profile)))
        pending = sorted([(k, t) for k, t in tasks if k not in completed], key=lambda item: (item[1]['category'], len(item[1]['text']), item[0]))
        started = time.monotonic(); initial = len(completed)
        for offset in range(0, len(pending), batch_size):
            chunk = pending[offset:offset+batch_size]
            if any(fingerprint([t['category'], t['text']]) != k for k, t in chunk):
                raise ValueError('Text/task mismatch.')
            values = predict_many(predictor, [t for _, t in chunk])
            for (key, _), estimates in zip(chunk, values):
                db.execute('INSERT INTO predictions VALUES (?,?)', (key, canonical_json(estimates)))
                completed.add(key)
            if (offset//batch_size) % 5 == 0 or len(completed) == len(tasks):
                db.commit()
                elapsed = time.monotonic() - started
                rate = (len(completed) - initial) / elapsed
                progress = {'status': 'complete' if len(completed) == len(tasks) else 'running',
                            'shard': shard, 'completed': len(completed), 'total': len(tasks),
                            'texts_per_second': rate, 'updated_at': datetime.now(timezone.utc).isoformat(),
                            'remaining_seconds': (len(tasks)-len(completed))/rate if rate else None}
                partial = output / f'shard-{shard}-progress.partial'
                partial.write_text(canonical_json(progress), encoding='utf-8')
                partial.replace(output / f'shard-{shard}-progress.json')
                if (offset//batch_size) % 25 == 0 or len(completed) == len(tasks):
                    print(canonical_json(progress), flush=True)
        db.commit()
        if len(completed) != len(tasks):
            raise ValueError('Incomplete shard.')
    receipt = output / f'shard-{shard}-complete.json'
    if not receipt.exists():
        save_json(receipt, {**binding, 'count': len(completed), 'profile': profile})


def main():
    # PSEUDOCODE: choose a physical GPU through CUDA_VISIBLE_DEVICES outside the worker.
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('tasks', 'checkpoint', 'base', 'model-id', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--skip-cache')
    parser.add_argument('--verify-batch', action='store_true')
    args = parser.parse_args()
    score(args.tasks, args.checkpoint, args.base, args.model_id, args.output, args.shard, args.shards,
          args.batch_size, args.skip_cache, args.verify_batch)


if __name__ == '__main__':
    main()
