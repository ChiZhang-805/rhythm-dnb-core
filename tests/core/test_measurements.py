"""Measurement edge cases, daylight-saving time and missing wear."""

from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo
import unittest
import numpy as np
from rhythm_dnb.timebase import instant, research_day, forecast_bounds, local_boundary, available_as_of
from rhythm_dnb.measures.sleep import daily_sleep, sleep_regularity, sleep_regularity_index
from rhythm_dnb.measures.eating import daily_eating, eating_regularity
from rhythm_dnb.measures.activity import daily_activity, activity_regularity, intradaily_variability, hourly_profile
from rhythm_dnb.measures.physiology import daily_physiology
from rhythm_dnb.contracts import Observation, Provenance

UTC = timezone.utc


class MeasurementTests(unittest.TestCase):
    def test_dst_elapsed_sleep(self):
        zone = ZoneInfo('America/Chicago')
        result = daily_sleep([(datetime(2025, 3, 8, 23, tzinfo=zone), datetime(2025, 3, 9, 7, tzinfo=zone))], 'America/Chicago')
        self.assertEqual(result['sleep_duration_h'], 7)
        self.assertEqual(result['sleep_midpoint_h'], 3.5)

    def test_naive_and_nonexistent_time_rejected(self):
        for value in [datetime(2025, 1, 1), datetime(2025, 3, 9, 2, 30, tzinfo=ZoneInfo('America/Chicago'))]:
            with self.assertRaises(ValueError):
                instant(value)

    def test_boundary_and_forecast_noon(self):
        at = datetime(2025, 1, 2, 3, 59, tzinfo=UTC)
        self.assertEqual(research_day(at, 'UTC'), date(2025, 1, 1))
        self.assertEqual(research_day(at.replace(hour=4), 'UTC'), date(2025, 1, 2))
        self.assertEqual(forecast_bounds(at.replace(hour=12, minute=0), 'UTC')[0], date(2025, 1, 1))
        with self.assertRaises(ValueError):
            forecast_bounds(at, 'UTC')

    def test_sleep_overlap_rejected_and_circular_s1(self):
        start = datetime(2025, 1, 1, tzinfo=UTC)
        with self.assertRaises(ValueError):
            daily_sleep([(start, start + timedelta(hours=4)), (start + timedelta(hours=3), start + timedelta(hours=7))], 'UTC')
        self.assertLess(sleep_regularity([23.8, .1, 0, .2, 23.9, 0]), .2)
        self.assertIsNone(sleep_regularity([1, 2, 3]))

    def test_fragmented_sleep_midpoint_needs_main_period(self):
        start = datetime(2025, 1, 1, 22, tzinfo=UTC)
        intervals = [(start, start + timedelta(hours=3)), (start + timedelta(hours=4), start + timedelta(hours=8))]
        self.assertIsNone(daily_sleep(intervals, 'UTC')['sleep_midpoint_h'])
        result = daily_sleep(intervals, 'UTC', main_period=(start, start + timedelta(hours=8)))
        self.assertEqual(result['sleep_midpoint_h'], 2)
        self.assertEqual(result['sleep_duration_h'], 7)

    def test_sri_exact_lag_and_missing(self):
        day = [1] * 8 + [0] * 16
        self.assertEqual(sleep_regularity_index(day * 3, 24), 100)
        self.assertEqual(sleep_regularity_index(day + [1 - x for x in day], 24), -100)
        self.assertIsNone(sleep_regularity_index(day + [np.nan] * 24, 24))

    def test_caloric_order_crosses_midnight(self):
        first = datetime(2025, 1, 1, 22, tzinfo=UTC)
        events = [(first, 10), (first + timedelta(hours=3), 40), (first + timedelta(hours=4), 0)]
        self.assertEqual(daily_eating(events, 'UTC', complete=True), {'first_caloric_h': 22, 'last_caloric_h': 1})
        self.assertIsNone(daily_eating(events, 'UTC', complete=False)['first_caloric_h'])
        self.assertIsNone(eating_regularity([1] * 5, [2] * 5))

    def test_m10_l5_and_zero_activity(self):
        x = np.zeros(24); x[8:18] = 10
        result = daily_activity(x, [60] * 24)
        self.assertEqual(result['activity_m10_start_h'], 8)
        self.assertEqual(result['activity_ra'], 1)
        self.assertEqual(result['activity_total'], 100)
        self.assertIsNone(daily_activity(np.zeros(24))['activity_ra'])

    def test_nonwear_is_not_zero_and_dst_not_24_bins(self):
        x = np.arange(24, dtype=float); x[0] = np.nan
        self.assertIsNone(daily_activity(x)['activity_total'])
        wear = [60] * 24; wear[0] = 44
        self.assertIsNone(daily_activity(np.arange(24), wear)['activity_ra'])
        with self.assertRaises(ValueError):
            daily_activity(np.arange(23))

    def test_hourly_counts_normalize_actual_wear_exposure(self):
        wear = [60.] * 24; wear[0] = 45.
        counts = [60.] * 24; counts[0] = 45.
        np.testing.assert_allclose(hourly_profile(counts, wear), 60.)
        self.assertEqual(daily_activity(counts, wear)['activity_total'], 1440.)

    def test_is_repeated_vs_shifted_profile_and_iv(self):
        day = np.r_[np.zeros(8), np.ones(10), np.zeros(6)]
        self.assertAlmostEqual(activity_regularity(np.tile(day, (7, 1))), 0)
        self.assertGreater(activity_regularity([np.roll(day, shift) for shift in range(7)]), .1)
        self.assertIsNone(activity_regularity(np.ones((7, 24))))
        self.assertAlmostEqual(intradaily_variability([0, 1, 0, 1]), 4)

    def test_resting_hr_is_not_arbitrary_minimum(self):
        self.assertIsNone(daily_physiology([50, 60, 80])['resting_hr_bpm'])
        self.assertEqual(daily_physiology([50, 60, 80], resting_mask=[True, True, False])['resting_hr_bpm'], 55)

    def test_event_end_at_cutoff_and_late_arrival(self):
        cutoff = datetime(2025, 1, 2, 4, tzinfo=UTC)
        row = Observation('a', 'p', 'activity_hour', 12., 'count', cutoff - timedelta(hours=1), cutoff,
                          cutoff + timedelta(hours=1), 'UTC', Provenance('observed', 'a', 'abc'))
        self.assertTrue(available_as_of(row, cutoff, cutoff + timedelta(hours=8)))
        self.assertFalse(available_as_of(row, cutoff, cutoff))
