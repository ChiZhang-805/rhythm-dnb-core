"""Causal daily feature assembly; no outcomes or research modules are imported."""

from datetime import timedelta
import math
import numpy as np
from ..timebase import instant, local_boundary
from ..provenance import eligible_measurement
from ..measures.panel import UNITS, CLOCKS


def panel_vector(panel, features, as_of, *, simulation=False):
    # PSEUDOCODE: enforce the fixed universe, units, provenance and arrival time before extraction.
    mapping = {value.name: value for value in panel.features}
    if len(mapping) != len(panel.features):
        raise ValueError('Duplicate feature name in a person-day.')
    vector, reasons = [], []
    unfinished = instant(local_boundary(panel.day + timedelta(days=1), panel.timezone)) > instant(as_of)
    for name in features:
        f = mapping.get(name)
        reason = None
        if unfinished:
            reason = 'unfinished_research_day'
        elif f is None or f.value is None:
            reason = 'missing'
        elif type(f.value) not in (int, float) or not math.isfinite(f.value):
            reason = 'invalid_value'
        elif f.unit != UNITS.get(name, f.unit):
            reason = 'unit_mismatch'
        elif f.version != '2':
            reason = 'measurement_version_mismatch'
        elif instant(f.available_at) > instant(as_of):
            reason = 'not_available'
        elif f.measured_until is None or instant(f.measured_until) < instant(local_boundary(panel.day, panel.timezone)) or instant(f.measured_until) > instant(local_boundary(panel.day + timedelta(days=1), panel.timezone)) or instant(f.measured_until) > instant(f.available_at):
            reason = 'unverified_measurement_cutoff'
        elif not eligible_measurement(f.provenance, simulation=simulation):
            reason = 'ineligible_provenance'
        elif f.reason is not None or type(f.coverage) not in (int, float) or not 0 < f.coverage <= 1:
            reason = f.reason or 'no_coverage'
        elif name in CLOCKS and not 0 <= f.value < 24:
            reason = 'clock_out_of_range'
        elif name.startswith('text_') and not 0 <= f.value <= 100:
            reason = 'text_score_out_of_range'
        elif name == 'activity_ra' and not 0 <= f.value <= 1:
            reason = 'ratio_out_of_range'
        elif name == 'sleep_duration_h' and f.value > 24:
            reason = 'duration_out_of_range'
        elif name == 'resting_hr_bpm' and not 20 <= f.value <= 250:
            reason = 'physiology_out_of_range'
        elif name not in CLOCKS and f.value < 0:
            reason = 'negative_measurement'
        vector.append(np.nan if reason else f.value)
        if reason:
            reasons.append(name + ':' + reason)
    return np.asarray(vector, dtype=float), tuple(reasons)


def select_window(history, participant_id, zone, last_day, features, as_of, *, days=28,
                  minimum=23, max_missing_run=2, simulation=False):
    # PSEUDOCODE: align to every calendar day -> retain explicit gaps -> gate complete-day coverage.
    if any(type(n) is not int for n in (days, minimum, max_missing_run)) or not 1 <= minimum <= days or max_missing_run < 0:
        raise ValueError('Invalid rolling window coverage policy.')
    mapping = {}
    for panel in history:
        if panel.participant_id != participant_id or panel.timezone != zone:
            raise ValueError('Mixed participant or timezone in inference history.')
        if panel.day in mapping:
            raise ValueError('Duplicate day in inference history.')
        mapping[panel.day] = panel
    rows, current_run, longest = [], 0, 0
    for offset in range(days - 1, -1, -1):
        day = last_day - timedelta(days=offset)
        row = panel_vector(mapping[day], features, as_of, simulation=simulation)[0] if day in mapping else np.full(len(features), np.nan)
        missing = not np.isfinite(row).all()
        current_run = current_run + 1 if missing else 0
        longest = max(longest, current_run)
        rows.append(row)
    matrix = np.asarray(rows)
    complete = int(np.isfinite(matrix).all(axis=1).sum())
    reason = 'insufficient_complete_days' if complete < minimum else 'missing_run_too_long' if longest > max_missing_run else None
    if not np.isfinite(matrix[-1]).all():
        reason = 'missing_target_day'
    return matrix, reason
