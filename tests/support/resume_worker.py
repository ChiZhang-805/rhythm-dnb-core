"""Isolated distributed continuation worker used only by the recovery integration test."""

import json
from pathlib import Path
import sys

from rhythm_dnb.text.experiment import train_experiment


if __name__ == '__main__':
    root, stage = Path(sys.argv[1]), sys.argv[2]
    config = json.loads((root / 'training.json').read_text(encoding='utf-8'))
    corpus = json.loads((root / 'corpus/development.json').read_text(encoding='utf-8'))
    train_experiment(corpus, root / 'base', root / stage, config, save_resume_state=True,
                     resume_state=root / 'original/resume/epoch-1.json' if stage == 'resumed' else None)
