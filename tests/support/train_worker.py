"""Subprocess entry for the distributed integration test only."""

import json
from pathlib import Path
import sys
from rhythm_dnb.text.train import train

if __name__ == '__main__':
    root = Path(sys.argv[1])
    train(json.loads((root / 'corpus.json').read_text(encoding='utf-8')), root / 'base', root / 'distributed',
          json.loads((root / 'training.json').read_text(encoding='utf-8')))
