from pm4py.objects.petri_net.obj import PetriNet, Marking
from typing import Optional, Dict, Any, Tuple
from copy import deepcopy


def apply(net: PetriNet, initial_marking: Marking, final_marking: Marking, string_dictio: Dict[str, str], parameters: Optional[Dict[Any, Any]] = None) -> Tuple[PetriNet, Marking, Marking]:
    """
    Replaces the labels in the provided accepting Petri net using the provided correspondence dictionary.

    Parameters
    ----------------
    net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    string_dictio
        Correspondence dictionary (old labels -> new labels)

    Returns
    ----------------
    net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    """
    if parameters is None:
        parameters = {}

    net, initial_marking, final_marking = deepcopy([net, initial_marking, final_marking])

    for trans in net.transitions:
        if trans.label is not None and trans.label in string_dictio:
            trans.label = string_dictio[trans.label]

    return net, initial_marking, final_marking
