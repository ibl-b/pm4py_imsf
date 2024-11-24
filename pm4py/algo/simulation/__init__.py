from pm4py.util import constants as pm4_constants

if pm4_constants.ENABLE_INTERNAL_IMPORTS:
    from pm4py.algo.simulation import playout

    import importlib.util
    if importlib.util.find_spec("tree_generator"):
        from pm4py.algo.simulation import tree_generator
