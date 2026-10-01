"""Score a caller-provided request with a frozen bundle; this script never fits."""

import argparse
import json
from pathlib import Path
from rhythm_dnb.api import RhythmPredictor
from rhythm_dnb.contracts import parse_request
from rhythm_dnb.provenance import canonical_json


def main():
    # PSEUDOCODE: load explicit artifact/request -> call the stable API -> print continuous and binary results.
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle'); parser.add_argument('request')
    args = parser.parse_args()
    payload = json.loads(Path(args.request).read_text(encoding='utf-8'))
    result = RhythmPredictor.from_bundle(args.bundle).predict(parse_request(payload))
    print(canonical_json(result))


if __name__ == '__main__':
    main()
