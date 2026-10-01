"""Fixed-universe Pearson DNB; Chen 2012 / Liu 2017, DOI 10.1371/journal.pcbi.1005633.

Rows are observations; ddof=1; missing rows are jointly excluded.
Ported from the existing audited project kernel without changing pair semantics.
"""

from collections.abc import Mapping
from itertools import combinations
import math
from .statistics import _names, _integer, _indices, _sequence


def candidate_modules(features, modules=None, method='fixed', sizes=(2, 3), maximum_candidates=10000):
    """Return candidate-name -> sorted feature list, with no fitting or ranking.

    'fixed' requires a nonempty mapping of names to modules; overlap is allowed.
    'bounded_exhaustive' counts combinations before enumeration, then names them
    module_1, module_2, ... in ascending size and lexicographic feature order.
    Sizes must leave an outside feature; use sizes=(2,) for three features.
    """
    # PSEUDOCODE: validate fixed dimensions -> compute candidate modules -> retain invalid-data reasons.
    names = tuple(sorted(_names(features, 'features')))
    if len(names) < 3:
        raise ValueError('At least three features are required.')
    budget = _integer(maximum_candidates, 'maximum_candidates')
    if method == 'fixed':
        if not isinstance(modules, Mapping) or not modules:
            raise ValueError('fixed requires a nonempty mapping of named modules.')
        if len(modules) > budget:
            raise ValueError(f'Search requires {len(modules)} candidates, exceeding budget {budget}.')
        if any(not isinstance(name, str) or not name.strip() for name in modules):
            raise ValueError('Candidate names must be nonempty strings.')
        return {name: [names[i] for i in _indices(modules[name], names)] for name in sorted(modules)}
    if method != 'bounded_exhaustive':
        raise ValueError("method must be 'fixed' or 'bounded_exhaustive'.")
    if modules is not None:
        raise ValueError('modules must be omitted for bounded_exhaustive.')
    sizes = tuple(_integer(k, 'module size', 2) for k in _sequence(sizes, 'sizes'))
    if not sizes or len(set(sizes)) != len(sizes) or any(k >= len(names) for k in sizes):
        raise ValueError('sizes must be unique and leave a nonempty outside complement.')
    count = sum(math.comb(len(names), size) for size in sizes)
    if count > budget:
        raise ValueError(f'Search requires {count} candidates, exceeding budget {budget}.')
    candidates = (list(group) for size in sorted(sizes) for group in combinations(names, size))
    return {f'module_{i}': group for i, group in enumerate(candidates, 1)}
