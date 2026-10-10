"""Check ranking identity and the R boundary without needing a local R installation."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.run_dnbr_warning import choose_modules, read_scores, audit_components
from rhythm_dnb.dnb.single_sample import sdnb_components


class DnbrWarningTests(unittest.TestCase):
    def test_selection_uses_only_development_stage_and_unique_eligible_sets(self):
        settings = {'selection_stage': 'pre_event', 'max_modules': 2}
        def row(genes, score, stage='pre_event', usable='TRUE'):
            return dict(genes=genes, SCORE=str(score), stage=stage, usable=usable, resource=genes)
        rows = [row('a,b', 5), row('b,a', 4), row('c,d', 3), row('b,c', 100, 'stable'),
                row('a,c', 200, usable='FALSE')]
        selected = choose_modules(rows, list('abcd'), settings)
        self.assertEqual([x['module'] for x in selected], [['a', 'b'], ['c', 'd']])
        self.assertEqual(choose_modules([], list('abcd'), settings), [])
        with self.assertRaises(ValueError):
            choose_modules([row('a,b,c,d', 1)], list('abcd'), settings)

    def test_missing_is_not_zero_and_incomplete_or_nonfinite_results_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scores.csv'
            path.write_text('record_id,score,status\na,,invalid_module\nb,0,ok\n')
            self.assertEqual(read_scores(path, ['a', 'b']), {'a': None, 'b': 0.})
            for text in ('a,NaN,ok\nb,1,ok\n', 'a,1,ok\na,2,ok\n', 'a,0,invalid_module\nb,1,ok\n'):
                path.write_text('record_id,score,status\n' + text)
                with self.assertRaises(ValueError):
                    read_scores(path, ['a', 'b'])

    def test_component_audit_detects_different_pair_convention_and_missing_rows(self):
        rng = np.random.default_rng(82)
        reference = rng.normal(size=(60, 4)); targets = rng.normal(size=(2, 4))
        rows = [{'record_id': 'first'}, {'record_id': 'second'}]
        modules = [{'module': ['a', 'b']}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'components.csv'
            def write(convention, count=2):
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['record_id', 'module_id', 'valid',
                        'sED_in', 'sPCC_in', 'sPCC_out', 'score'])
                    writer.writeheader()
                    for i in range(count):
                        part = sdnb_components(targets[i], reference, list('abcd'), ['a', 'b'],
                                               pair_convention=convention)
                        writer.writerow({k: part[k.lower()] for k in ('sED_in', 'sPCC_in', 'sPCC_out', 'score')} |
                                        {'record_id': rows[i]['record_id'], 'module_id': 1, 'valid': 'TRUE'})
            write('paper_k_squared')
            self.assertEqual(audit_components(path, rows, reference, targets, list('abcd'), modules, 1e-8)['checked'], 2)
            write('unique_pairs')
            with self.assertRaises(AssertionError):
                audit_components(path, rows, reference, targets, list('abcd'), modules, 1e-8)
            write('paper_k_squared', 1)
            with self.assertRaises(ValueError):
                audit_components(path, rows, reference, targets, list('abcd'), modules, 1e-8)


if __name__ == '__main__':
    unittest.main()
