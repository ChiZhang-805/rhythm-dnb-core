"""Plot held-out text errors, evidence coverage, dispersion and paired reference/model scores."""

from pathlib import Path
import numpy as np
from ..research.plots import _save


def plot_evaluation(result, predictions, directory):
    # PSEUDOCODE: use saved held-out results only -> plot error/coverage/variation -> inspect residuals and paired scores.
    import matplotlib.pyplot as plt
    report = result['test']
    keys = list(report['per_metric'])
    if not keys or predictions['identity'] != result['identity']:
        raise ValueError('Plots require matching held-out results and checkpoint identity.')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    labels = [key.replace('_', ' ') for key in keys]
    figure, axes = plt.subplots(1, 3, figsize=(17, max(6, len(keys) * .4)), layout='constrained', sharey=True)
    experimental = result.get('purpose') == 'experimental_semantic_regression'
    middle = ('baseline_mae', 'Training-mean baseline error') if experimental else ('coverage', 'Evidence acceptance fraction')
    for axis, field, title in zip(axes, ('mae', middle[0], 'sd_ratio'), ('Absolute error (0–100 scale)', middle[1], 'Prediction SD / reference SD')):
        source = report['evidence'] if field == 'coverage' else report['mean_baseline'] if field == 'baseline_mae' else report['per_metric']
        values = [source[k]['mae' if field == 'baseline_mae' else field] for k in keys]
        axis.barh(labels, [np.nan if v is None else v for v in values])
        axis.set_title(title); axis.grid(axis='x', alpha=.2)
        if field == 'coverage':
            axis.set_xlim(0, 1)
        if field == 'sd_ratio':
            axis.axvline(1, color='black', linestyle='--', linewidth=1)
    axes[0].invert_yaxis()
    figure.suptitle('Held-out text measurement: low error alone does not establish preserved variation')
    outputs = [_save(figure, directory / 'metric-diagnostics')]
    matrix = np.full((len(keys), len(keys)), np.nan)
    for pair in report['residual_correlations']:
        if pair['left'] in keys and pair['right'] in keys and pair['correlation'] is not None:
            i, j = keys.index(pair['left']), keys.index(pair['right'])
            matrix[i, j] = matrix[j, i] = pair['correlation']
    figure, axis = plt.subplots(figsize=(12, 10), layout='constrained')
    color = plt.get_cmap('coolwarm').with_extremes(bad='#dddddd')
    image = axis.imshow(matrix, cmap=color, vmin=-1, vmax=1)
    axis.set_xticks(range(len(keys)), labels, rotation=75, ha='right', fontsize=8)
    axis.set_yticks(range(len(keys)), labels, fontsize=8)
    axis.set_title('Correlated text scoring errors can contaminate DNB correlations\nGray: unavailable; only jointly annotated metrics are estimable')
    figure.colorbar(image, ax=axis, label='Residual Pearson correlation')
    outputs.append(_save(figure, directory / 'residual-correlations'))
    for page in range(0, len(keys), 6):
        figure, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
        for axis, key in zip(axes.flat, keys[page:page + 6]):
            pairs = [(r['truth'][key], r['scores'][key]) for r in predictions['rows'] if r['truth'].get(key) is not None]
            if pairs:
                x, y = np.asarray(pairs, dtype=float).T
                axis.scatter(x, y, s=10, alpha=.4)
            axis.plot([0, 100], [0, 100], '--', color='black', linewidth=1)
            axis.set(xlim=(0, 100), ylim=(0, 100), xlabel='Reference score', ylabel='Model estimate', title=key.replace('_', ' '))
        for axis in list(axes.flat)[len(keys[page:page + 6]):]:
            axis.set_visible(False)
        figure.suptitle('Experimental held-out reference scores' if experimental else 'Known held-out labels; raw estimates shown before evidence rejection')
        outputs.append(_save(figure, directory / f'paired-scores-{page // 6 + 1}'))
    return {'figures': outputs}
