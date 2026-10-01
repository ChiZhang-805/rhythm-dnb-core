"""Score frozen modules with the fixed outside universe; no online module search."""

from ..contracts import DNBResult
from ..dnb.classic import dnb_components
from ..dnb.single_sample import sdnb_components
from ..measures.scaling import transform


def score_modules(values, reference, modules, *, kind='single_sample', epsilon=1e-8,
                  minimum=23, pair_convention='paper_k_squared'):
    # PSEUDOCODE: use reference scaling -> score each frozen module -> convert component names.
    transformed = transform(values, reference['scaler'])
    results = []
    for module in modules:
        if kind == 'single_sample':
            result = sdnb_components(transformed, reference['matrix'], reference['features'],
                                     module, epsilon=epsilon, pair_convention=pair_convention)
            keys = ('sed_in', 'spcc_in', 'spcc_out')
        elif kind == 'rolling':
            result = dnb_components(transformed, reference['features'], module,
                                    min_samples=minimum, epsilon=epsilon)
            keys = ('sd_in', 'pcc_in', 'pcc_out')
        else:
            raise ValueError('Unknown DNB method.')
        results.append(DNBResult(result['valid'], result['score'], *(result[k] for k in keys),
                                 tuple(result['module']), result['n_valid'], result['reason']))
    return tuple(results)


def aggregate_score(results):
    # PSEUDOCODE: require every frozen module to be valid, preserving the calibrated max strategy.
    if not results or any(not result.valid for result in results):
        return None
    return max(result.score for result in results)
