from pm4py.util import constants as pm4_constants

if pm4_constants.ENABLE_INTERNAL_IMPORTS:
    from pm4py.objects.process_tree import obj, semantics, state, utils
    import importlib.util

    if importlib.util.find_spec("lxml"):
        from pm4py.objects.process_tree import importer, exporter
