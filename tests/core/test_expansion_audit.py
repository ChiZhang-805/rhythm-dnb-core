"""Check expansion counts do not bridge missing or ambiguous calendar days."""

from datetime import date, timedelta
import unittest

from tools.audit_expansion import calendar_capacity


class ExpansionAuditTests(unittest.TestCase):
    def test_fourteen_days_supply_five_complete_future_windows(self):
        first = date(2026, 1, 1)
        result = calendar_capacity([first + timedelta(days=i) for i in range(14)])
        self.assertEqual(result['calendar_window_upper_bound'], 5)

    def test_gaps_and_duplicates_break_windows(self):
        first = date(2026, 1, 1)
        dates = [first + timedelta(days=i) for i in range(14)]
        for values in (dates[:6] + dates[7:], dates + [dates[6]]):
            result = calendar_capacity(values)
            self.assertEqual(result['calendar_window_upper_bound'], 0)
            self.assertEqual(result['longest_streak'], 7)
        self.assertEqual(calendar_capacity([])['calendar_window_upper_bound'], 0)


if __name__ == '__main__':
    unittest.main()
