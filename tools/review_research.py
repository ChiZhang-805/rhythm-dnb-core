"""Reproducible read-only repository/data scan and scientific diagnostic evidence.

Run from an explicit checkout; output must be a new directory. External data paths are optional and explicit. No raw
database writes, re-labeling, training, downloads or model selection are performed.
"""

import argparse
import ast
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rhythm_dnb.provenance import canonical_json
from rhythm_dnb.config import StudyConfig
from rhythm_dnb.research.report import audit_legacy_store
from rhythm_dnb.research.simulate import simulate_mechanism, simulate_cohort
from rhythm_dnb.research.discover import discover
from rhythm_dnb.dnb.reference import fit_reference
from rhythm_dnb.measures.panel import get_panel
from rhythm_dnb.dnb.classic import dnb_components
from rhythm_dnb.dnb.single_sample import sdnb_components


def scan_files():
    # PSEUDOCODE: parse every maintained Python/JSON file -> record exact hashes, dependencies, constants and docstrings.
    paths = set()
    for folder in ('src', 'tools', 'examples', 'tests', 'configs'):
        paths.update(p for p in (ROOT / folder).rglob('*') if p.suffix in ('.py', '.json') and '__pycache__' not in p.parts)
    paths.update(ROOT.glob('*.py'))
    records = []
    for path in sorted(paths):
        source = path.read_text(encoding='utf-8-sig'); rel = path.relative_to(ROOT).as_posix()
        record = {'file': rel, 'sha256': sha256(path.read_bytes()).hexdigest(), 'lines': len(source.splitlines()),
                  'review_scope': 'static_full_file_scan; targeted semantic review and tests documented separately'}
        if path.suffix == '.py':
            tree = ast.parse(source)
            functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            record.update(purpose=ast.get_docstring(tree), functions=[n.name for n in functions],
                imports=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))],
                numeric_literals=[{'line': n.lineno, 'value': n.value} for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (float, int)],
                missing_pseudocode=[n.name for n in functions if '# PSEUDOCODE:' not in ast.get_source_segment(source, n)] if rel.startswith('src/') else [],
                placeholder_lines=[n.lineno for n in ast.walk(tree) if isinstance(n, ast.Raise) and 'NotImplemented' in ast.unparse(n)])
        else:
            payload = json.loads(source); record.update(purpose='Versioned configuration', keys=list(payload) if isinstance(payload, dict) else [])
        records.append(record)
    return records


def text_audit(path):
    # PSEUDOCODE: count stored semantic corpus origins and partitions without exporting personal text.
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('BEGIN')
        counts = [dict(zip(('origin', 'split', 'category', 'rows'), row)) for row in db.execute('SELECT origin,split,category,count(*) FROM chinese_examples GROUP BY origin,split,category')]
        return {'counts': counts, 'rows': sum(r['rows'] for r in counts),
                'interpretation': 'Origins describe corpus construction; stored score labels alone do not establish independent human agreement.'}


def sensitivity():
    # PSEUDOCODE: vary sample counts across fixed seeds; preserve raw replicates instead of choosing an optimum.
    result = {'domain': 'synthetic_only', 'seeds': list(range(30)), 'rolling_samples': {}, 'reference_people': {}}
    names = tuple('abcdef'); module = names[:3]
    for count in (14, 28, 42, 56):
        scores = []
        for seed in result['seeds']:
            rng = np.random.default_rng(seed)
            x = rng.normal(size=(count, 6)); latent = rng.normal(scale=3, size=count)
            x[:, :3] = latent[:, None] * [1, -1, 1] + rng.normal(scale=.2, size=(count, 3))
            scores.append(dnb_components(x, names, module)['score'])
        result['rolling_samples'][str(count)] = scores
    for count in (20, 40, 60, 100, 200):
        scores = []
        for seed in result['seeds']:
            reference = np.random.default_rng(seed).normal(size=(count, 6))
            scores.append(sdnb_components([4, -4, 4, .2, -.1, .3], reference, names, module, pair_convention='paper_k_squared')['score'])
        result['reference_people'][str(count)] = scores
    return result


def single_sample_identifiability():
    # PSEUDOCODE: give identical current vectors two different histories; demonstrate the information sDNB lacks.
    reference = np.random.default_rng(7).normal(size=(60, 6))
    target = [4., -4., 4., .2, -.1, .3]
    result = sdnb_components(target, reference, tuple('abcdef'), tuple('abc'), pair_convention='paper_k_squared')
    return {'domain': 'synthetic_counterexample', 'constant_personal_trait': result, 'new_pretransition_deviation': result,
            'interpretation': 'Identical present-day vectors have identical sDNB scores, even if prior histories differ. A population deviation alone cannot establish an approaching transition.'}


def run(output, database=None, text_database=None):
    # PSEUDOCODE: reserve evidence location -> scan all files/data -> render separately labeled real and synthetic diagnostics.
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('MPLCONFIGDIR', str(output / 'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from rhythm_dnb.research.plots import plot_source_coverage, plot_metric_distributions, plot_network_diagnostics, _save
    inventory = scan_files()
    audit = audit_legacy_store(database.expanduser().resolve()) if database else None
    corpus = text_audit(text_database.expanduser().resolve()) if text_database else None
    scans = {'files': len(inventory), 'lines': sum(r['lines'] for r in inventory), 'inventory': inventory}
    reports = [('file_scan', scans), ('study_defaults', asdict(StudyConfig())), ('mechanism', simulate_mechanism()), ('single_sample_counterexample', single_sample_identifiability())]
    if audit is not None:
        reports.append(('data_audit', audit))
    if corpus is not None:
        reports.append(('text_audit', corpus))
    for name, value in reports:
        (output / (name + '.json')).write_text(canonical_json(value), encoding='utf-8')
    if audit is not None:
        plot_source_coverage(audit, output / 'source_coverage')
        plot_metric_distributions(audit, output / 'metrics')
    rng = np.random.default_rng(20261001); n = 200
    stable = rng.normal(size=(n, 6)) + .5 * rng.normal(size=(n, 1))
    critical = rng.normal(size=(n, 6)); latent = rng.normal(scale=3, size=n)
    critical[:, :3] = latent[:, None] * [1, -1, 1] + rng.normal(scale=.15, size=(n, 3))
    artifact = plot_network_diagnostics({'Stable': stable, 'Critical-like': critical, 'Shared measurement error': stable + np.column_stack([latent, latent, latent, np.zeros((n, 3))])},
        tuple('abcdef'), tuple('abc'), output / 'network_diagnostics', provenance_label='SYNTHETIC mechanism controls — not patient validation')
    (output / 'network_components.json').write_text(canonical_json(artifact), encoding='utf-8')
    figure, axis = plt.subplots(figsize=(6, 5), layout='constrained')
    axis.scatter(critical[:, 0], critical[:, 1], alpha=.5, s=12)
    axis.set(xlabel='Feature a', ylabel='Feature b', title='SYNTHETIC opposite-direction coordination')
    _save(figure, output / 'coordination')
    if corpus is not None:
        counts = Counter()
        for row in corpus['counts']:
            counts[row['origin']] += row['rows']
        figure, axis = plt.subplots(figsize=(7, 5), layout='constrained')
        axis.bar(list(counts), list(counts.values()))
        axis.set_title('Stored text corpus origins; external database')
        axis.tick_params(axis='x', rotation=20)
        _save(figure, output / 'corpus_origins')
    varied = sensitivity(); (output / 'sensitivity.json').write_text(canonical_json(varied), encoding='utf-8')
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for axis, name in zip(axes, ('rolling_samples', 'reference_people')):
        axis.boxplot(list(varied[name].values()), tick_labels=list(varied[name])); axis.set(xlabel=name, ylabel='DNB index', title='SYNTHETIC only; 30 fixed seeds')
    _save(figure, output / 'parameter_sensitivity')
    discoveries = []
    for n in (40, 80, 200):
        c = simulate_cohort(development_people=n)
        r = fit_reference(c['reference'], get_panel(c['config'].panel_id), c['reference_cutoff'], minimum=60, simulation=True)
        d = discover(c['pairs'], c['config'], r, c['discovery_cutoff'])
        discoveries.append({'development_people': n, 'selected_modules': len(d['modules']), 'minimum_adjusted_p': min(x['max_stat_adjusted_p'] for x in d['candidates']), 'status': d['status']})
    (output / 'discovery_controls.json').write_text(canonical_json(discoveries), encoding='utf-8')
    print(canonical_json({'output': str(output), 'scanned_files': len(inventory), 'database_rows': audit['rows'] if audit else None, 'data_snapshot': audit['snapshot_content_id'] if audit else None, 'discovery_controls': discoveries}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--database', type=Path, help='Optional read-only legacy rhythm SQLite input.')
    parser.add_argument('--text-database', type=Path, help='Optional read-only legacy text SQLite input.')
    args = parser.parse_args()
    run(args.output_dir, args.database, args.text_database)
