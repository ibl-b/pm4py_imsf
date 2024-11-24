from pm4py.util import constants as pm4_constants

if pm4_constants.ENABLE_INTERNAL_IMPORTS:
    from pm4py.objects.random_variables import constant0, normal, uniform, exponential, random_variable, lognormal, gamma

    import warnings

    if pm4_constants.SHOW_INTERNAL_WARNINGS:
        warnings.warn("The random_variables package will be removed in a future release.")
