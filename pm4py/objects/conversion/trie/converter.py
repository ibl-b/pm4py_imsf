from pm4py.objects.conversion.trie.variants import to_petri_net
from enum import Enum
from pm4py.util import exec_utils
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.trie.obj import Trie
from typing import Optional, Dict, Any, Tuple


class Variants(Enum):
    TO_PETRI_NET = to_petri_net


def apply(prefix_tree: Trie, variant=Variants.TO_PETRI_NET, parameters: Optional[Dict[Any, Any]] = None) -> Tuple[
    PetriNet, Marking, Marking]:
    """
    Converts the prefix tree objects using the specified variant

    Parameters
    ----------------
    prefix_tree
        Prefix tree
    variant
        Variant of the conversion:
        - Variants.TO_PETRI_NET => converts the prefix tree object to a Petri net
    parameters
        Optional parameters of the method.

    Returns
    ----------------
    obj
        Converted object
    """
    return exec_utils.get_variant(variant).apply(prefix_tree, parameters)
