from pm4py.util import constants as pm4_constants

if pm4_constants.ENABLE_INTERNAL_IMPORTS:
    from pm4py.objects.bpmn import obj, exporter, layout, semantics, util
    import importlib.util

    if importlib.util.find_spec("lxml"):
        from pm4py.objects.bpmn import importer
