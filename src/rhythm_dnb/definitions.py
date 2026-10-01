"""Automatic measurement identity from calculation code, without manual release labels."""

import ast
from functools import lru_cache
from hashlib import sha256
import io
import json
from pathlib import Path
import tokenize


def normalized_syntax(source):
    # PSEUDOCODE: remove comments/docstrings -> normalize indentation -> retain executable token order.
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node):
            statement = node.body[0]
            docstrings.add((statement.lineno, statement.end_lineno))
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENDMARKER):
            continue
        if token.type in (tokenize.STRING, tokenize.NEWLINE) and any(start <= token.start[0] <= end for start, end in docstrings):
            continue
        value = '' if token.type in (tokenize.INDENT, tokenize.DEDENT) else token.string
        tokens.append((tokenize.tok_name[token.type], value))
    return json.dumps(tokens, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


@lru_cache(maxsize=1)
def measurement_identity():
    # PSEUDOCODE: hash scientific calculation dependencies with a Python-release-independent token format.
    root = Path(__file__).resolve().parent
    paths = [path for name in ('measures', 'dnb', 'outcomes', 'warning') for path in (root / name).glob('*.py')]
    paths += [root / name for name in ('definitions.py', 'timebase.py', 'provenance.py', 'config.py', 'contracts.py',
              'io/validation.py', 'workflows/prepare.py', 'workflows/endpoints.py', 'workflows/develop.py',
              'research/discover.py', 'research/calibrate.py', 'research/evaluate.py',
              'text/schema.py', 'text/model.py', 'text/dataset.py', 'text/predict.py')]
    digest = sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(normalized_syntax(path.read_text(encoding='utf-8-sig')))
    return digest.hexdigest()


MEASUREMENT_ID = measurement_identity()
