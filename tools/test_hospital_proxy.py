"""Run a separate prospective-direction pilot against later recorded timing changes."""

import argparse
import json
from pathlib import Path
from rhythm_dnb.research.hospital_proxy import run_proxy


def main():
    # PSEUDOCODE: use explicit audited data and a frozen plan; create a new output directory.
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('input', 'plan', 'output'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    print(json.dumps(run_proxy(args.input, plan, args.output, plots=args.plots), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
