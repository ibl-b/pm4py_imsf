from pm4py.objects.bpmn.obj import BPMN
from typing import Optional, Dict, Any
from copy import deepcopy


def apply(bpmn_graph: BPMN, string_dictio: Dict[str, str], parameters: Optional[Dict[Any, Any]] = None) -> BPMN:
    if parameters is None:
        parameters = {}

    bpmn_graph = deepcopy(bpmn_graph)

    for node in bpmn_graph.get_nodes():
        if isinstance(node, BPMN.Task):
            name = node.get_name()
            if name in string_dictio:
                node.set_name(string_dictio[name])

    return bpmn_graph
