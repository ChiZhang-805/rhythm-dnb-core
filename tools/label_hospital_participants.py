"""Annotate available later timing records and compare one first-record prediction per person."""

import argparse
import json

from rhythm_dnb.research.hospital_participants import run_participants


def main():
    # PSEUDOCODE: require explicit audited inputs, protocol and a new private output directory.
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('input', 'plan', 'output'):
        parser.add_argument('--' + field, required=True)
    args = parser.parse_args()
    print(json.dumps(run_participants(args.input, args.plan, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
