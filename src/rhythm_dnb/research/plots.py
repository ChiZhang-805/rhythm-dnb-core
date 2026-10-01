"""Exportable diagnostic figures; never infer validation from a synthetic plot."""

from pathlib import Path
import numpy as np


def _save(figure, path):
    # PSEUDOCODE: create new PNG and SVG artifacts with matching stems; preserve prior evidence.
    import matplotlib.pyplot as plt
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.svg'):
        output = path.with_suffix(suffix)
        with output.open('xb') as stream:
            figure.savefig(stream, format=suffix[1:], dpi=150, bbox_inches='tight')
    plt.close(figure)
    return str(path.with_suffix('.png'))


def plot_source_coverage(audit, path):
    # PSEUDOCODE: divide traceable numeric cells by source rows -> show missing evidence explicitly.
    import matplotlib.pyplot as plt
    sources, fields = sorted(audit['sources']), audit['numeric_fields']
    matrix = np.array([[audit['traceable_by_source'].get(s, {}).get(k, 0) / audit['sources'][s] for k in fields] for s in sources])
    figure, axis = plt.subplots(figsize=(19, 6), layout='constrained')
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    axis.set_xticks(range(len(fields)), fields, rotation=70, ha='right', fontsize=7)
    axis.set_yticks(range(len(sources)), [f'{s} (n={audit["sources"][s]})' for s in sources], fontsize=8)
    axis.set_title('Stored numeric fields with file/key evidence / all rows in each source\nRetrospective traceability only; constructed values excluded; not prospective eligibility')
    figure.colorbar(image, ax=axis, label='Field-level traceable fraction')
    return _save(figure, path)


def plot_metric_distributions(audit, directory):
    # PSEUDOCODE: paginate every numeric field -> retain source separation -> annotate unobserved fields.
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    fields, outputs = audit['numeric_fields'], []
    pdf_path = directory / 'all_metric_distributions.pdf'
    with pdf_path.open('xb') as stream, PdfPages(stream) as pdf:
        for page in range(0, len(fields), 12):
            figure, axes = plt.subplots(4, 3, figsize=(15, 13), layout='constrained')
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
    import matplotlib.pyplot as plt
    from ..dnb.classic import dnb_components
    figure, axes = plt.subplots(2, len(matrices), figsize=(5 * len(matrices), 8), layout='constrained', squeeze=False)
    evidence = {}
    for j, (name, values) in enumerate(matrices.items()):
        x = np.asarray(values, dtype=float)
        result = dnb_components(x, feature_names, module); evidence[name] = result
        correlation = np.corrcoef(x[np.isfinite(x).all(axis=1)], rowvar=False)
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
