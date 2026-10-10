"""Guard date ambiguity, per-sheet mapping and participant-level network comparisons."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rhythm_dnb.io.sources.hospital_workbook import (FIELDS, CLOCKS, clock_minutes,
    normalized_number, partial_date, read_monitoring_workbook)
from rhythm_dnb.research.hospital_indicators import (analyze_panels, correlation_summary,
    participant_cube, validate_plan)


class FakeSheet:
    def __init__(self, name, pid, headers):
        self.name, self.ncols, self.nrows = name, len(headers), 7
        self.rows = [headers, headers, ['unit'] * len(headers)]
        for visit in range(4):
            values = dict.fromkeys(FIELDS, 1.)
            values.update({f: .25 for f in CLOCKS})
            values.update(number=float(pid), age=30., gender='女', exercise_type='步行',
                          date=9.15 + visit / 100, water_ml=1700.+pid, noise_db=42.+pid)
            values.update(meal_interval_CV='', _sd_bpm='')
            self.rows.append([values[f] for f in headers])

    def cell_value(self, r, c):
        return self.rows[r][c]

    def cell(self, r, c):
        value = self.cell_value(r, c)
        return SimpleNamespace(value=value, ctype=2 if isinstance(value, (int, float)) else 1 if value else 0)


class HospitalIndicatorTests(unittest.TestCase):
    def test_ambiguous_dates_and_clock_day_components_are_not_guessed(self):
        self.assertEqual(partial_date(9.1, 2)['candidates'], ['09-01', '09-10'])
        self.assertEqual(partial_date(9.2, 2)['candidates'], ['09-02', '09-20'])
        self.assertEqual(partial_date(9.3, 2)['candidates'], ['09-03', '09-30'])
        self.assertEqual(partial_date('9.10', 1)['month_day'], '09-10')
        self.assertEqual(partial_date(9.5, 2)['month_day'], '09-05')
        self.assertIsNone(partial_date(8.31, 2)['calendar_date'])
        self.assertEqual(partial_date((9*60+13)/1440, 3)['status'], 'clock_in_date_cell')
        self.assertEqual(partial_date((9*60+13)/1440, 3)['candidates'], ['09-13'])
        self.assertIsNone(clock_minutes(11.0))
        self.assertIsNone(clock_minutes('24:10'))
        self.assertAlmostEqual(clock_minutes('23:59:30'), 1439.5)
        self.assertIsNone(normalized_number('self_rate_state', 6))
        self.assertIsNone(normalized_number('press_score', float('nan')))

    def test_headers_are_mapped_per_sheet_and_raw_values_survive(self):
        first = FakeSheet('Sheet1', 1, list(FIELDS))
        second = FakeSheet('Sheet2', 2, list(reversed(FIELDS)) + ['meal_interval_CV', '_sd_bpm'])
        workbook = SimpleNamespace(sheets=lambda: [first, second], datemode=0, release_resources=lambda: None)
        with patch('xlrd.open_workbook', return_value=workbook), patch('rhythm_dnb.io.sources.hospital_workbook.file_hash', return_value='source'):
            data = read_monitoring_workbook('fixture.xls')
        self.assertEqual(data['participant_count'], 2)
        self.assertEqual(data['record_count'], 8)
        self.assertEqual(data['rows'][4]['values']['water_ml'], 1702.)
        self.assertEqual(data['rows'][4]['values']['noise_db'], 44.)
        self.assertEqual(data['rows'][0]['values']['noise_db'], 43.)
        self.assertIsNone(data['rows'][0]['observed_at'])
        self.assertNotIn('outcome', data['rows'][0])
        self.assertEqual(data['inventory']['_sd_bpm'], {'not_collected': 4, 'missing': 4})

    def test_absent_time_is_not_applicable_only_when_parent_is_known_zero(self):
        sheet = FakeSheet('Sheet1', 1, list(FIELDS))
        c = {f: i for i, f in enumerate(FIELDS)}
        for r in (3, 4):
            sheet.rows[r][c['caffeine_last_time']] = '—'
        sheet.rows[3][c['caffeine_mg']] = 0.
        sheet.rows[4][c['caffeine_mg']] = '—'
        sheet.rows[3][c['self_rate_state']] = 7.
        workbook = SimpleNamespace(sheets=lambda: [sheet], datemode=0, release_resources=lambda: None)
        with patch('xlrd.open_workbook', return_value=workbook), patch('rhythm_dnb.io.sources.hospital_workbook.file_hash', return_value='source'):
            data = read_monitoring_workbook('fixture.xls')
            with self.assertRaises(ValueError):
                read_monitoring_workbook('fixture.xls', [{'sheet': 'absent', 'cell': 'A1', 'field': 'water_ml'}])
        self.assertEqual(data['rows'][0]['states']['caffeine_last_time'], 'not_applicable')
        self.assertEqual(data['rows'][1]['states']['caffeine_last_time'], 'missing')
        self.assertIsNone(data['rows'][0]['values']['self_rate_state'])
        self.assertEqual(data['rows'][0]['raw']['self_rate_state']['value'], 7.)

    def example(self):
        features = ['sleep_duration_h', 'resting_hr_bpm', 'social_interaction_min', 'skin_temp_c']
        random = np.random.default_rng(11)
        rows = []
        for p in range(36):
            for visit in range(1, 5):
                values = dict(zip(features, random.normal(size=4)))
                rows.append({'participant_id': f'p{p}', 'occasion': visit, 'values': values,
                             'states': dict.fromkeys(features, 'observed'),
                             'raw': {f: {'value': v} for f, v in values.items()}})
        plan = {'purpose': 'descriptive_hospital_indicator_audit_not_warning_validation',
                'panels': {'core': features[:2], 'expanded': features},
                'bootstrap_replicates': 100, 'seed': 92, 'interval_quantiles': [.025, .975]}
        return rows, plan

    def test_shared_people_draws_and_edges_do_not_change_by_adding_nodes(self):
        rows, plan = self.example()
        result = analyze_panels({'rows': rows}, plan)
        self.assertEqual(result['people'], 36)
        self.assertIsNone(result['warning_accuracy'])
        core = result['panels']['core']; expanded = result['panels']['expanded']
        for a, b in zip(core['occasions'], expanded['occasions']):
            self.assertEqual(a['people'], b['people'])
            self.assertEqual(a['bootstrap_valid_draws'], 100)
        for edge in core['edges']:
            match = next(e for e in expanded['edges'] if all(e[k] == edge[k] for k in ('occasion', 'feature_a', 'feature_b')))
            for key in ('pearson', 'low', 'high'):
                self.assertAlmostEqual(edge[key], match[key], places=12)
        self.assertEqual(result, analyze_panels({'rows': rows}, plan))

    def test_missing_one_visit_excludes_whole_person_and_not_only_one_row(self):
        rows, plan = self.example()
        rows[0]['values']['skin_temp_c'] = None
        rows[0]['states']['skin_temp_c'] = 'missing'
        cube, people, occasions, excluded = participant_cube(rows, plan['panels']['expanded'])
        self.assertEqual(cube.shape, (35, 4, 4))
        self.assertNotIn('p0', people)
        self.assertEqual(excluded[0]['participant_id'], 'p0')
        with self.assertRaises(ValueError):
            participant_cube(rows + [rows[0]], plan['panels']['expanded'])
        self.assertIsNone(correlation_summary(np.ones((60, 4))))
        # Perfect copies concentrate in one covariance direction, independent columns do not.
        identical = np.tile(np.arange(30)[:, None], (1, 4))
        self.assertAlmostEqual(correlation_summary(identical)['participation_ratio'], 1)

    def test_plan_rejects_clock_dimensions_and_malformed_settings(self):
        _, plan = self.example()
        validate_plan(plan)
        for change in ({'bootstrap_replicates': True}, {'seed': -1}, {'interval_quantiles': [.1, .9]},
                       {'panels': {'core': ['sleep_start_time', 'sleep_end_time'], 'expanded': ['sleep_start_time', 'sleep_end_time', 'resting_hr_bpm']}}):
            with self.assertRaises(ValueError):
                validate_plan({**plan, **change})


if __name__ == '__main__':
    unittest.main()
