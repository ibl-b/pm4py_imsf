from pm4py.util import constants as pm4_constants

if pm4_constants.ENABLE_INTERNAL_IMPORTS:
    from pm4py.algo.decision_mining import algorithm

    import warnings

    if pm4_constants.SHOW_INTERNAL_WARNINGS:
        warnings.warn("The decision_mining package will be removed in a future release.")
