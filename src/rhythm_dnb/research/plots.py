"""Export measurement distributions and network diagnostics from caller-supplied data."""

from pathlib import Path
import numpy as np


def _subplots(*args, figsize=None, layout=None, **kwargs):
    # PSEUDOCODE: create an export-only figure without a desktop GUI manager or Tk event loop.
    from matplotlib.figure import Figure
    figure = Figure(figsize=figsize, layout=layout)
    return figure, figure.subplots(*args, **kwargs)


def plot_experiment_result(result, path):
    # PSEUDOCODE: show held-out accuracy with person intervals and coverage; unavailable DNB accuracy stays blank.
    methods = list(result['methods'])
    figure, axes = _subplots(1, 2, figsize=(12, 4), layout='constrained')
    labels = [name.replace('_', ' ') for name in methods]
    for i, name in enumerate(methods):
        record = result['methods'][name]
        value = record['test']['risk_balanced_accuracy']
        interval = record['intervals']['risk_balanced_accuracy']['percentile_95']
        if value is None:
            axes[0].text(i, .12, 'Unavailable\n(no qualified policy)', ha='center', fontsize=9)
        else:
            axes[0].bar(i, value, color='#376f88')
            if interval:
                axes[0].plot([i, i], interval, color='black', linewidth=2)
            axes[0].text(i, min(.96, value + .035), f'{value:.1%}', ha='center')
        axes[1].bar(i, record['test']['classification_coverage'], color='#64836f')
    for axis in axes:
        axis.set_xticks(range(len(methods)), labels, rotation=12, ha='right')
        axis.set_ylim(0, 1.03)
        axis.grid(axis='y', alpha=.15)
    axes[0].set_title('Risk balanced accuracy (95% person bootstrap)')
    axes[1].set_title('Classification coverage')
    figure.suptitle('Existing authored simulation: not clinical validation')
    return _save(figure, path)


def _save(figure, path):
    # PSEUDOCODE: create new PNG and SVG artifacts with matching stems; preserve prior evidence.
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for suffix in ('.png', '.svg'):
            output = path.with_suffix(suffix)
            with output.open('xb') as stream:
                figure.savefig(stream, format=suffix[1:], dpi=150, bbox_inches='tight')
    finally:
        figure.clear()
    return str(path.with_suffix('.png'))


def plot_source_coverage(audit, path):
    # PSEUDOCODE: divide traceable numeric cells by source rows -> show missing evidence explicitly.
    sources, fields = sorted(audit['sources']), audit['numeric_fields']
    if not sources or not fields:
        raise ValueError('Source coverage needs at least one source and one numeric field.')
    matrix = np.array([[audit['traceable_by_source'].get(s, {}).get(k, 0) / audit['sources'][s] for k in fields] for s in sources])
    figure, axis = _subplots(figsize=(19, 6), layout='constrained')
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    axis.set_xticks(range(len(fields)), fields, rotation=70, ha='right', fontsize=7)
    axis.set_yticks(range(len(sources)), [f'{s} (n={audit["sources"][s]})' for s in sources], fontsize=8)
    axis.set_title('Stored numeric fields with file/key evidence / all rows in each source\nRetrospective traceability only; constructed values excluded; not prospective eligibility')
    figure.colorbar(image, ax=axis, label='Field-level traceable fraction')
    return _save(figure, path)


def plot_metric_distributions(audit, directory):
    # PSEUDOCODE: paginate every numeric field -> retain source separation -> annotate unobserved fields.
    from matplotlib.backends.backend_pdf import PdfPages
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    fields, outputs = audit['numeric_fields'], []
    pdf_path = directory / 'all_metric_distributions.pdf'
    with pdf_path.open('xb') as stream, PdfPages(stream) as pdf:
        for page in range(0, len(fields), 12):
            figure, axes = _subplots(4, 3, figsize=(15, 13), layout='constrained')
            for axis, key in zip(axes.flat, fields[page:page + 12]):
                groups = audit['traceable_distributions'].get(key, {})
                for source, values in sorted(groups.items()):
                    array = np.asarray(values, dtype=float); array = array[np.isfinite(array)]
                    if len(array):
                        axis.hist(array, bins='fd', histtype='step', density=True, label=f'{source}: {len(array)}')
                axis.set_title(key, fontsize=10)
                axis.set_ylabel('Density')
                if groups:
                    axis.legend(fontsize=5)
                else:
                    axis.text(.5, .5, 'No traceable source values', ha='center', transform=axis.transAxes)
            for axis in list(axes.flat)[len(fields[page:page + 12]):]:
                axis.set_visible(False)
            figure.suptitle('Retrospective source-backed field distributions — no clinical thresholds\nEach source is separate; row-level density is descriptive, not independent-person inference', fontsize=13)
            pdf.savefig(figure, bbox_inches='tight')
            outputs.append(_save(figure, directory / f'metrics_{page // 12 + 1}'))
    return {'pages': outputs, 'pdf': str(pdf_path)}


def plot_network_diagnostics(matrices, feature_names, module, path, *, provenance_label):
    # PSEUDOCODE: display signed Pearson matrices and independently computed DNB components.
    from ..dnb.classic import dnb_components
    if not matrices:
        raise ValueError('At least one observed matrix is required.')
    figure, axes = _subplots(2, len(matrices), figsize=(5 * len(matrices), 8), layout='constrained', squeeze=False)
    evidence = {}
    for j, (name, values) in enumerate(matrices.items()):
        x = np.asarray(values, dtype=float)
        result = dnb_components(x, feature_names, module); evidence[name] = result
        complete = x[np.isfinite(x).all(axis=1)]
        correlation = np.full((len(feature_names), len(feature_names)), np.nan)
        if len(complete) >= 2:
            with np.errstate(invalid='ignore', divide='ignore'):
                correlation = np.corrcoef(complete, rowvar=False)
        image = axes[0, j].imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, j].set_xticks(range(len(feature_names)), feature_names, rotation=45)
        axes[0, j].set_yticks(range(len(feature_names)), feature_names)
        axes[0, j].set_title(name)
        for a in range(len(feature_names)):
            for b in range(len(feature_names)):
                axes[0, j].text(b, a, f'{correlation[a,b]:.2f}', ha='center', va='center', fontsize=7)
        axes[1, j].bar(['SD in', '|PCC| in', '|PCC| out'], [np.nan if result[k] is None else result[k] for k in ('sd_in', 'pcc_in', 'pcc_out')], color=['#4477aa', '#228833', '#cc6677'])
        axes[1, j].set_title(f'CI = {result["score"]:.3g}' if result['valid'] else result['reason'])
    figure.colorbar(image, ax=axes[0, :].tolist(), shrink=.7, label='Signed Pearson correlation')
    upper = max(axis.get_ylim()[1] for axis in axes[1, :])
    for axis in axes[1, :]:
        axis.set_ylim(0, upper)
        axis.set_ylabel('SD: latent units; PCC: unitless')
    figure.suptitle(provenance_label + '\nDNB uses absolute correlations; heatmaps retain their directions')
    return {'figure': _save(figure, path), 'components': evidence}


def plot_pair_scatter(values, feature_names, pairs, path, *, provenance_label):
    # PSEUDOCODE: select named pairs -> retain finite paired observations -> show direction and actual counts.
    from ..dnb.statistics import _numeric
    names = tuple(feature_names)
    x = _numeric(values, 2, 'scatter matrix')
    if len(set(names)) != len(names) or x.shape[1] != len(names) or not pairs:
        raise ValueError('Scatter plots need unique feature names and at least one pair.')
    if any(len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(names) for pair in pairs):
        raise ValueError('Unknown or repeated scatter feature.')
    figure, axes = _subplots(1, len(pairs), figsize=(5 * len(pairs), 4), layout='constrained', squeeze=False)
    evidence = []
    for axis, (first, second) in zip(axes.flat, pairs):
        pair = x[:, [names.index(first), names.index(second)]]
        pair = pair[np.isfinite(pair).all(axis=1)]
        correlation = float(np.corrcoef(pair, rowvar=False)[0, 1]) if len(pair) >= 2 and np.all(np.std(pair, axis=0) > 0) else None
        axis.scatter(pair[:, 0], pair[:, 1], s=16, alpha=.6)
        axis.set_xlabel(first); axis.set_ylabel(second)
        axis.set_title(f'n={len(pair)}, r={correlation:.3f}' if correlation is not None else f'n={len(pair)}, correlation unavailable')
        evidence.append({'pair': [first, second], 'n': len(pair), 'correlation': correlation})
    figure.suptitle(provenance_label + '\nPairwise finite observations; descriptive association only')
    return {'figure': _save(figure, path), 'pairs': evidence}
