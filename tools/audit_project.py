"""Inspect every maintained file; optional figures use only an explicitly supplied real database."""

import argparse
import ast
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
import tomllib


def scan(root):
    # PSEUDOCODE: enumerate maintained files -> parse structured inputs -> record hashes and actionable findings.
    files = [path for path in root.iterdir() if path.is_file()]
    for name in ('src', 'tests', 'tools', 'configs', 'docs', '.github'):
        files.extend(path for path in (root / name).rglob('*') if path.is_file()
                     and '__pycache__' not in path.parts and not any(part.endswith('.egg-info') for part in path.parts))
    inventory, findings = [], []
    for path in sorted(files):
        name = path.relative_to(root).as_posix()
        if path.suffix in ('.pyc', '.pyo'):
            continue
        content = path.read_bytes()
        item = {'path': name, 'bytes': len(content), 'sha256': sha256(content).hexdigest()}
        if re.search(r'(?i)(?:^|[_-])v\d+(?=[_.-]|$)', path.name):
            findings.append(name + ': explicit project release label in filename')
        try:
            text = content.decode('utf-8-sig')
            if path.suffix == '.py':
                tree = ast.parse(text, filename=name)
                functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
                item.update(lines=len(text.splitlines()), functions=len(functions))
                if name.startswith('src/'):
                    if not ast.get_docstring(tree):
                        findings.append(name + ': missing module purpose')
                    lines = text.splitlines()
                    for node in functions:
                        first = node.body[1] if ast.get_docstring(node) and len(node.body) > 1 else node.body[0]
                        if not any('# PSEUDOCODE:' in line for line in lines[node.lineno:first.lineno]):
                            findings.append(f'{name}:{node.lineno}: missing calculation outline')
            elif path.suffix == '.json':
                json.loads(text)
            elif path.suffix == '.toml':
                tomllib.loads(text)
            elif path.suffix == '.csv':
                rows = list(csv.reader(text.splitlines()))
                if rows and any(len(row) != len(rows[0]) for row in rows):
                    raise ValueError('CSV columns do not match header')
            elif path.suffix == '.md':
                for target in re.findall(r'\]\(([^)]+)\)', text):
                    if '://' not in target and not target.startswith('#') and not (path.parent / target.split('#')[0]).exists():
                        findings.append(name + ': broken relative link ' + target)
        except (ValueError, SyntaxError) as error:
            findings.append(name + ': ' + str(error))
        inventory.append(item)
    return inventory, findings


def check_configs(root):
    # PSEUDOCODE: compare configured panels/defaults against the actual contracts and the parameter table.
    from rhythm_dnb.config import StudyConfig, load_study
    from rhythm_dnb.measures.panel import get_panel, CLOCKS, UNITS
    from rhythm_dnb.text.config import validate_config
    findings = []
    for path in sorted((root / 'configs').glob('study*.json')):
        load_study(path)
    for path in sorted((root / 'configs/panels').glob('*.json')):
        panel = json.loads(path.read_text(encoding='utf-8'))
        expected = [{'name': key, 'unit': UNITS[key], 'circular': key in CLOCKS} for key in get_panel(panel['id'])]
        if panel['features'] != expected:
            findings.append(str(path.relative_to(root)) + ': panel differs from executable contract')
    for path in sorted((root / 'configs/text').glob('*.json')):
        validate_config(json.loads(path.read_text(encoding='utf-8')))
    training = validate_config(json.loads((root / 'configs/text/qwen.json').read_text(encoding='utf-8')))
    with (root / 'docs/parameters.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    parameters = {row['parameter']: row for row in rows}
    if len(parameters) != len(rows):
        findings.append('docs/parameters.csv: duplicate parameter')
    for name, value in {**asdict(StudyConfig()), **{('text_seed' if k == 'seed' else k): v for k, v in training.items()}}.items():
        actual = parameters.get(name, {}).get('default')
        if isinstance(value, (tuple, list)):
            matched = actual == ';'.join(map(str, value))
        elif type(value) in (int, float):
            matched = actual is not None and float(actual) == value
        else:
            matched = actual == str(value).lower() if isinstance(value, bool) else actual == value
        if not matched:
            findings.append('docs/parameters.csv: default mismatch for ' + name)
    return findings


def main():
    # PSEUDOCODE: save a fresh source-audit receipt -> optionally audit actual historical records and plot them.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--database', help='Read-only legacy database with observations and observation_provenance tables')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    inventory, findings = scan(root)
    findings.extend(check_configs(root))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    report = {'files': inventory, 'file_count': len(inventory), 'findings': findings,
              'scope': 'Static consistency checks; this is not scientific or human code-review certification.'}
    (output / 'source-audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if args.database:
        from rhythm_dnb.research.report import audit_legacy_store, save_report
        from rhythm_dnb.research.plots import plot_source_coverage, plot_metric_distributions
        audit = audit_legacy_store(args.database)
        save_report(audit, output / 'data-audit.json')
        if audit['sources'] and audit['numeric_fields']:
            plot_source_coverage(audit, output / 'source-coverage')
            plot_metric_distributions(audit, output / 'distributions')
    print(json.dumps({'files': len(inventory), 'findings': findings, 'output': str(output)}, ensure_ascii=False))
    return int(bool(findings))


if __name__ == '__main__':
    raise SystemExit(main())
