from pm4py.util import constants as pm4_constants

if pm4_constants.ENABLE_INTERNAL_IMPORTS:
    from pm4py.objects.stochastic_petri import tangible_reachability, utils

    import warnings

    if pm4_constants.SHOW_INTERNAL_WARNINGS:
        warnings.warn("The stochastic_petri package will be removed in a future release.")
