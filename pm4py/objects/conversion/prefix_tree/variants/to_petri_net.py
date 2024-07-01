import uuid
from pm4py.objects.petri_net.utils import petri_utils
from typing import Optional, Dict, Any, Tuple
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.trie.obj import Trie


def __add_prefix_tree_node_to_petri(node: Trie, net: PetriNet, prev: PetriNet.Place, sink: PetriNet.Place):
    """
    Internal method to add a prefix tree node to a Petri net.
    """
    trans = PetriNet.Transition(str(uuid.uuid4()), label=node.label)
    net.transitions.add(trans)
    petri_utils.add_arc_from_to(prev, trans, net)
    if node.children:
        succ_place = PetriNet.Place(str(uuid.uuid4()))
        net.places.add(succ_place)
        petri_utils.add_arc_from_to(trans, succ_place, net)
        for child in node.children:
            __add_prefix_tree_node_to_petri(child, net, succ_place, sink)
        if node.final:
            new_trans = PetriNet.Transition(str(uuid.uuid4()))
            net.transitions.add(new_trans)
            petri_utils.add_arc_from_to(succ_place, new_trans, net)
            petri_utils.add_arc_from_to(new_trans, sink, net)
    else:
        petri_utils.add_arc_from_to(trans, sink, net)


def apply(prefix_tree: Trie, parameters: Optional[Dict[Any, Any]] = None) -> Tuple[PetriNet, Marking, Marking]:
    """
    Transforms a prefix tree to an accepting Petri net.

    Parameters
    -----------------
    prefix_tree
        Prefix_tree

    Returns
    -----------------
    net
        Petri net
    im
        Initial marking
    fm
        Final marking
    """
    if parameters is None:
        parameters = {}

    # create a empty Petri net
    net = PetriNet("petri")
    im = Marking()
    fm = Marking()
    source = PetriNet.Place(str(uuid.uuid4()))
    net.places.add(source)
    sink = PetriNet.Place(str(uuid.uuid4()))
    net.places.add(sink)
    im[source] = 1
    fm[sink] = 1

    __add_prefix_tree_node_to_petri(list(prefix_tree.children)[0], net, source, sink)

    return net, im, fm
