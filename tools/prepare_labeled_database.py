"""Check every labeled record, fill only supported values, and export Excel/SQLite."""

import argparse
import json

from rhythm_dnb.research.labeled_dataset import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', required=True)
    parser.add_argument('--database', required=True)
    parser.add_argument('--hospital-normalized', required=True)
    parser.add_argument('--hospital-people', required=True)
    parser.add_argument('--hospital-protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = prepare(**vars(args))
    print(json.dumps({k: result[k] for k in ('rows_checked', 'filled_cells', 'filled_by_column',
                    'objective_input_rows', 'text_input_rows', 'hospital_verified_numeric_rows', 'workbook', 'database')}, ensure_ascii=False))
    if result.get('workbook_locked'):
        print(json.dumps({'workbook_locked': True, 'checked_workbook': result['checked_workbook']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
