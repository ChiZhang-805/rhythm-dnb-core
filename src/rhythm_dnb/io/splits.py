"""Frozen role assignments; participant and text-template groups never cross roles."""

import numpy as np
from ..provenance import fingerprint


def make_splits(participant_ids, *, fractions=(.25, .35, .2, .2), seed=20261001):
    # PSEUDOCODE: sort unique people -> seeded shuffle -> assign reference/development/calibration/test.
    ids = sorted(participant_ids)
    if len(ids) != len(set(ids)) or not ids or len(fractions) != 4 or any(f <= 0 for f in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError('Invalid participant list or split fractions.')
    np.random.default_rng(seed).shuffle(ids)
    boundaries = np.r_[0, np.floor(np.cumsum(fractions) * len(ids)).astype(int)]
    boundaries[-1] = len(ids)
    roles = ('reference', 'development', 'calibration', 'test')
    partitions = {role: sorted(ids[boundaries[i]:boundaries[i + 1]]) for i, role in enumerate(roles)}
    if any(not values for values in partitions.values()):
        raise ValueError('Too few people for nonempty partitions.')
    payload = {'seed': seed, 'partitions': partitions}
    return {**payload, 'id': fingerprint(payload)}


def validate_splits(rows, *, split_key='split', group_keys=('participant_id', 'group_id')):
    # PSEUDOCODE: track the assigned split for each leakage group and reject conflicting ownership.
    ownership = {}
    for row in rows:
        if not row.get(split_key):
            raise ValueError('Missing split assignment.')
        for key in group_keys:
            value = row.get(key)
            if value:
                identity = (key, value)
                if identity in ownership and ownership[identity] != row[split_key]:
                    raise ValueError('Cross-split leakage: ' + key)
                ownership[identity] = row[split_key]
    return True
