from pm4py.objects.process_tree.obj import ProcessTree
from typing import Optional, Dict, Any
from copy import deepcopy


def apply(process_tree: ProcessTree, string_dictio: Dict[str, str], rec_depth=0, parameters: Optional[Dict[Any, Any]] = None) -> ProcessTree:
    """
    Replaces the labels in the given process tree using the provided dictionary.

    Parameters
    -----------------
    process_tree
        Process tree
    string_dictio
        Correspondence dictionary (old labels -> new labels)

    Returns
    -----------------
    revised_tree
        Revised process tree
    """
    if parameters is None:
        parameters = {}

    if rec_depth == 0:
        process_tree = deepcopy(process_tree)

    if process_tree.label is not None and process_tree.label in string_dictio:
        process_tree.label = string_dictio[process_tree.label]

    for child in process_tree.children:
        apply(child, string_dictio, rec_depth=rec_depth+1, parameters=parameters)

    return process_tree
