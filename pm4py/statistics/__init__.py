from pm4py.util import constants as pm4_constants

if pm4_constants.ENABLE_INTERNAL_IMPORTS:
    from pm4py.statistics import traces, attributes, variants, start_activities, end_activities, \
        service_time, concurrent_activities, eventually_follows, rework
