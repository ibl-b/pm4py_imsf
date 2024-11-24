from pm4py.util import constants as pm4_constants

if pm4_constants.ENABLE_INTERNAL_IMPORTS:
    import importlib.util

    if importlib.util.find_spec("matplotlib"):
        from pm4py.algo.comparison import petrinet
