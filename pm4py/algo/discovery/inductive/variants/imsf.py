"""
    PM4Py – A Process Mining Library for Python
Copyright (C) 2024 Process Intelligence Solutions UG (haftungsbeschränkt)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see this software project's root or
visit <https://www.gnu.org/licenses/>.

Website: https://processintelligence.solutions
Contact: info@processintelligence.solutions
"""

from typing import TypeVar, Generic, Dict, Any, Optional, Tuple

from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructureUVCL,
    IMDataStructureLog,
)

from pm4py.algo.discovery.inductive.variants.abc import InductiveMinerFramework
from pm4py.algo.discovery.inductive.variants.instances import IMInstance
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.algo.discovery.inductive.fall_through.empty_traces import EmptyTracesUVCL
from pm4py.objects.conversion.process_tree import converter as process_tree_converter
from pm4py.objects.process_tree.utils import generic as pt_utils
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.utils import final_marking as fm
from copy import copy


T = TypeVar("T", bound=IMDataStructureLog)

"""
Variant of the inductive Miner using synthesis for creating a workflow net as 
fall through.
"""

class IMSF(Generic[T], InductiveMinerFramework[T]):

    def instance(self) -> IMInstance:
        return IMInstance.IMsf
    
    
    """
    Converts a process tree to a petri net. Searches for placeholders in the tree, that 
    store the synthesized nets of the synthesis fall through. 

    :param tree: The process tree to convert  
    :returns a Tuple containing the resulting petri net, an initial marking and a final marking.
    """
    
    def convert_to_petri_net(tree: ProcessTree) -> Tuple[PetriNet, Marking, Marking]:
        net, initial_marking, final_marking = process_tree_converter.apply(tree)
        leaves = pt_utils.get_leaves(tree)
        synth_nets = {}
        for leave in leaves:
            if leave.label and leave.label.startswith("synth_placeholder_"):
                placeholder_id = leave.label.split("_")[-1]
                synth_nets[placeholder_id] = (leave.petri_net, leave.im)
                
        if not synth_nets:
            return net, initial_marking, final_marking

        placeholders = [t for t in net.transitions if t.label and t.label.startswith("synth_placeholder_")]

        for placeholder in placeholders:
            placeholder_id = placeholder.label.split("_")[-1]
            if placeholder_id in synth_nets:
                synth_net, synth_im = synth_nets[placeholder_id]
                net, initial_marking = IMSFUVCL._replace_placeholder_with_synth_net(net, placeholder, synth_net, synth_im, initial_marking)

        net = petri_utils.remove_unconnected_components(net)
        updated_fm = fm.discover_final_marking(net)
        return net, initial_marking, updated_fm   


    """
    Replaces a placeholder transition in the petri net with the synthesized net of the corresponding
    synthesized net

    :param net: the net to insert the synthesized net
    :param net_placeholder: the placeholder transition to replace with the synthesized net
    :param synth_net: the synthesized net
    :param synth_im: the initial marking of the synthesized net
    :param initial_marking: the initial marking of the petri net
    :returns a Tuple containing the resulting petri net and an initial marking
    """
    def _replace_placeholder_with_synth_net(net:PetriNet, net_placeholder: PetriNet.Transition, synth_net: PetriNet, synth_im: Marking, initial_marking: Marking) -> Tuple[PetriNet, Marking]:
        synth_start = None
        synth_stop = None
        for t in synth_net.transitions:
            if t.name == "Start":
                synth_start = t
            if t.name == "Stop":
                synth_stop = t

        if not synth_start or not synth_stop:
            return net  

        net.transitions.update(synth_net.transitions)
        net.places.update(synth_net.places)
        net.arcs.update(synth_net.arcs)

        synth_source_arcs = copy(synth_start.in_arcs)
        synth_sink_arcs = copy(synth_stop.out_arcs)
        for arc in synth_source_arcs:
            place = arc.source
            if place in synth_im:
                del synth_im[place]
            net = petri_utils.remove_arc(net, arc)
            net = petri_utils.remove_place(net, place)

        for arc in synth_sink_arcs:
            place = arc.target
            net = petri_utils.remove_arc(net, arc)
            net = petri_utils.remove_place(net, place)

        net = IMSFUVCL._connect_nets(net, net_placeholder, synth_start, synth_stop)

        for place, count in synth_im.items():
            initial_marking[place] = count

        return net, initial_marking

    """
    Connects the synthesized net with the petri net.

    :param net: the petri net
    :param net_placeholder: the placeholder transition to replace with the synthesized net
    :param synth_start: the starting silent transition of the synthesized net
    :param synth_im: the silent transition marking the end of the synthesized net
    :returns the resulting petri net
    """
    def _connect_nets(net: PetriNet, net_placeholder: PetriNet.Transition, synth_start: PetriNet.Transition, synth_stop: PetriNet.Transition) -> PetriNet:
        
        arcs_to_remove = []

        # connect net with start of the synthesized net
        for arc in net_placeholder.in_arcs:
            source = arc.source
            arcs_to_remove.append(arc)
            petri_utils.add_arc_from_to(fr=source, to=synth_start, net=net)
        
        # connect end of the synthesized net with net
        for arc in net_placeholder.out_arcs:
            target = arc.target
            arcs_to_remove.append(arc)
            petri_utils.add_arc_from_to(fr=synth_stop, to=target, net=net)
        
        for arc in arcs_to_remove:
            petri_utils.remove_arc(net, arc)
        
        return net


class IMSFUVCL(IMSF[IMDataStructureUVCL]):

    def apply(
        self,
        obj: IMDataStructureUVCL,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ProcessTree:

        empty_traces = EmptyTracesUVCL.apply(obj, parameters)
        if empty_traces is not None:
            return self._recurse(empty_traces[0], empty_traces[1], parameters)
        return super().apply(obj, parameters)
            
        

