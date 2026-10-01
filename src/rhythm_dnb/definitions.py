"""Automatic measurement identity from calculation code, without manual release labels."""

import ast
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path


def normalized_syntax(source):
    # PSEUDOCODE: parse executable structure -> remove prose -> normalize optional interpreter fields.
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body.pop(0)

    def canonical(value):
        # PSEUDOCODE: serialize syntax without positions or empty fields introduced by newer interpreters.
        if isinstance(value, ast.AST):
            fields = {name: canonical(item) for name, item in ast.iter_fields(value)
                      if not (name in ('type_params', 'kind', 'type_comment') and not item)}
            return [type(value).__name__, fields]
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if isinstance(value, bytes):
            return {'bytes': value.hex()}
        if isinstance(value, complex):
            return {'complex': [value.real, value.imag]}
        if value is Ellipsis:
            return {'ellipsis': True}
        return value

    return json.dumps(canonical(tree), sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


@lru_cache(maxsize=1)
def measurement_identity():
    # PSEUDOCODE: hash normalized scientific dependencies without manual release labels or machine paths.
    root = Path(__file__).resolve().parent
    paths = [path for name in ('measures', 'dnb', 'outcomes', 'warning') for path in (root / name).glob('*.py')]
    paths += [root / name for name in ('definitions.py', 'timebase.py', 'provenance.py', 'config.py', 'contracts.py',
              'io/validation.py', 'workflows/prepare.py', 'workflows/endpoints.py', 'workflows/develop.py',
              'research/discover.py', 'research/calibrate.py', 'research/evaluate.py',
              'text/schema.py', 'text/model.py', 'text/dataset.py', 'text/predict.py', 'text/evidence.py')]
    digest = sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(normalized_syntax(path.read_text(encoding='utf-8-sig')))
    return digest.hexdigest()


MEASUREMENT_ID = measurement_identity()
