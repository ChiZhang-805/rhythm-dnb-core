"""Collect verified expanded scores and append outputs beside preserved source tables."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from rhythm_dnb.provenance import canonical_json
from rhythm_dnb.research.experiment_data import save_json
from rhythm_dnb.research.expanded_results import read_json, verify_archive, merge_scores, append_database

def main():
    # PSEUDOCODE: finalize a complete downloaded run, optionally install its outputs beside preserved original tables.
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('packet', 'downloaded', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--database', type=Path)
    args = parser.parse_args()
    verify_archive(args.downloaded)
    scores, bindings, plan = merge_scores(args.packet, args.downloaded)
    verify_archive(args.downloaded/'evaluation')
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'text-scores.json', scores)
    if args.database:
        append_database(args.database, args.output, scores, bindings, plan, args.downloaded/'evaluation',
                        read_json(args.packet/'measurements.json'))
    save_json(args.output/'completion.json', {'packet_id': plan['id'], 'text_result_id': scores['id'],
        'counts': scores['counts'], 'warning_archive': str((args.downloaded/'evaluation').resolve()),
        'completed_at': datetime.now(timezone.utc).isoformat()})
    print(canonical_json(scores['counts']))


if __name__ == '__main__':
    main()
