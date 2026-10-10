"""Normalize hospital XLS inputs and compare fixed indicator panels on the CPU."""

import argparse
import json
from pathlib import Path

from rhythm_dnb.research.hospital_indicators import run_hospital_analysis


def main():
    # PSEUDOCODE: require explicit sources and a fresh destination; leave original workbooks unchanged.
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'description', 'plan', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    result = run_hospital_analysis(args.input, args.description, plan, args.output, plots=args.plots)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
