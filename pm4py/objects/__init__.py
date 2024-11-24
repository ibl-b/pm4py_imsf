from pm4py.util import constants as pm4_constants

if pm4_constants.ENABLE_INTERNAL_IMPORTS:
    from pm4py.objects import log, petri_net, transition_system, conversion, process_tree, \
        dfg, trie, org
    import importlib.util

    if importlib.util.find_spec("networkx"):
        from pm4py.objects import bpmn
