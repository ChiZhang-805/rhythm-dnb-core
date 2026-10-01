"""Frozen measurement universes. Missing nodes never change the complement."""

from ..text.schema import METRICS

OBJECTIVE8 = ('sleep_midpoint_h', 'sleep_duration_h', 'first_caloric_h', 'last_caloric_h',
              'activity_m10_start_h', 'activity_ra', 'activity_total', 'resting_hr_bpm')
JOINT12 = OBJECTIVE8 + ('text_anxiety_intensity', 'text_sadness_intensity',
                       'text_post_sleep_fatigue', 'text_stress_intensity')
CLOCKS = ('sleep_midpoint_h', 'first_caloric_h', 'last_caloric_h', 'activity_m10_start_h')
ALL_MEASURES = OBJECTIVE8 + tuple('text_' + key for key, *_ in METRICS)
UNITS = {name: ('hour' if name in CLOCKS or name == 'sleep_duration_h' else
                'bpm' if name == 'resting_hr_bpm' else 'count' if name == 'activity_total' else
                'score_0_100' if name.startswith('text_') else 'ratio') for name in ALL_MEASURES}


def get_panel(panel_id):
    # PSEUDOCODE: resolve a named fixed universe; reject implicit patient-specific panels.
    panels = {'joint12': JOINT12, 'objective8': OBJECTIVE8}
    if panel_id not in panels:
        raise ValueError('Unknown panel: ' + str(panel_id))
    return panels[panel_id]
