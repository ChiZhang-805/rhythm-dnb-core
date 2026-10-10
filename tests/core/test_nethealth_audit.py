"""Check calendar eligibility and ambiguous source-day handling for expansion planning."""

from datetime import date, timedelta
import importlib.util
import unittest

PANDAS_AVAILABLE = importlib.util.find_spec('pandas') is not None
if PANDAS_AVAILABLE:
    import pandas as pd
    from tools.audit_nethealth import pair_days, windows


@unittest.skipUnless(PANDAS_AVAILABLE, 'NetHealth audit requires the optional io dependencies.')
class NetHealthAuditTests(unittest.TestCase):
    def test_window_includes_history_current_day_and_future(self):
        days = [date(2026, 1, 1) + timedelta(days=i) for i in range(38)]
        self.assertEqual(windows(days, 28, 9), 1)
        self.assertEqual(windows(days[:37], 28, 9), 0)
        self.assertEqual(windows(days[:18] + days[19:], 28, 9), 0)
        self.assertEqual(windows(days + [days[18]], 28, 9), 0)

    def test_conflicts_are_not_resolved_by_row_order(self):
        activity = pd.DataFrame([
            [1, 'a', 20], [1, 'a', 20], [1, 'b', 30], [1, 'b', 40], [1, 'c', 30]
        ], columns=['egoid', 'datadate', 'steps'])
        sleep = pd.DataFrame([
            [1, 'a', 60, 'nap'], [1, 'a', 400, 'main'], [1, 'b', 400, 'main'],
            [1, 'c', 400, 'first'], [1, 'c', 400, 'second']
        ], columns=['egoid', 'datadate', 'bedtimedur', 'episode'])
        paired, receipt = pair_days(activity, sleep)
        self.assertEqual(paired.datadate.tolist(), ['a'])
        self.assertEqual(paired.episode.tolist(), ['main'])
        self.assertEqual(receipt['activity_exact_duplicate_rows'], 1)
        self.assertEqual(receipt['conflicting_activity_days_excluded'], 1)
        self.assertEqual(receipt['tied_longest_sleep_days_excluded'], 1)
        self.assertEqual(len(activity), 5)
        self.assertEqual(len(sleep), 5)


if __name__ == '__main__':
    unittest.main()
