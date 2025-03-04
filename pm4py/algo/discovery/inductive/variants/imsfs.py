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

from typing import TypeVar, Generic, Dict, Any, Optional, Counter, Tuple

from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructureUVCL,
    IMDataStructureLog,
)
from pm4py.algo.discovery.inductive.fall_through.synthesis import SynthesisUVCL
from pm4py.algo.discovery.inductive.variants.abc import InductiveMinerFramework
from pm4py.algo.discovery.inductive.variants.instances import IMInstance
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.algo.discovery.inductive.fall_through.empty_traces import EmptyTracesUVCL
from pm4py.objects.conversion.process_tree import converter as process_tree_converter
from pm4py.objects.process_tree.utils import generic as pt_utils
from pm4py.objects.petri_net.utils import petri_utils

from pm4py import vis
from copy import copy


T = TypeVar("T", bound=IMDataStructureLog)


class IMSFS(Generic[T], InductiveMinerFramework[T]):

    def instance(self) -> IMInstance:
        return IMInstance.IMsfs


class IMSFSUVCL(IMSFS[IMDataStructureUVCL]):

    # TODO später erst prüfen, ob es schon start und Ende hat
    def _preprocess_log(obj: IMDataStructureUVCL) -> IMDataStructureUVCL:
        traces = obj.data_structure
        first_element = next(iter(traces.keys()))[0]
        if first_element == "Start":
            return obj
        new_traces = Counter()
        for trace, count in traces.items():
            new_trace = ("Start",) + trace + ("Stop",)
            new_traces[new_trace] = count

        return IMDataStructureUVCL(new_traces)

    def apply(
        self,
        obj: IMDataStructureUVCL,
        parameters: Optional[Dict[str, Any]] = None,
        second_iteration: bool = False,
    ) -> ProcessTree:

        empty_traces = EmptyTracesUVCL.apply(obj, parameters)
        if empty_traces is not None:
            return self._recurse(empty_traces[0], empty_traces[1], parameters)
        tree = self.apply_base_cases(obj, parameters)
        if tree is None:
            cut = self.find_cut(obj, parameters)
            if cut is not None:
                tree = self._recurse(cut[0], cut[1], parameters=parameters)
        if tree is None:
            ft = self.fall_through(obj, parameters)
            if isinstance(ft, ProcessTree):
                return ft
            tree = self._recurse(ft[0], ft[1], parameters=parameters)
            
        return tree
    
    def convert_process_tree(tree: ProcessTree) -> Tuple[PetriNet, Marking, Marking]:
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
                net = IMSFSUVCL._replace_placeholder_with_synth_net(net, placeholder, synth_net, synth_im, initial_marking)

        net = petri_utils.remove_unconnected_components(net)
        vis.view_petri_net(net, initial_marking, final_marking, format="svg")  
        return net, initial_marking, final_marking   

    def _replace_placeholder_with_synth_net(net:PetriNet, net_placeholder: PetriNet.Transition, synth_net: PetriNet, synth_im: Marking, initial_marking: Marking) -> PetriNet:
        synth_start = None
        synth_stop = None
        for t in synth_net.transitions:
            if t.label == "Start":
                synth_start = t
            if t.label == "Stop":
                synth_stop = t

        if not synth_start or not synth_stop:
            return net  # Falls kein gültiges Synthesenetz, breche ab

        # Füge das Synthesenetz in das Hauptnetz ein
        net.transitions.update(synth_net.transitions)
        net.places.update(synth_net.places)
        net.arcs.update(synth_net.arcs)

        vis.view_petri_net(net, initial_marking, Marking(), format="svg")

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

        vis.view_petri_net(net, initial_marking, Marking(), format="svg")

        net = IMSFSUVCL._connect_nets(net, net_placeholder, synth_start, synth_stop)

        # Übernehme initiale Markierung
        for place, count in synth_im.items():
            initial_marking[place] = count

        vis.view_petri_net(net, initial_marking, Marking(), format="svg")

        return net




    def _connect_nets(net: PetriNet, net_placeholder: PetriNet.Transition, synth_start: PetriNet.Transition, synth_stop: PetriNet.Transition) -> PetriNet:
        
        arcs_to_remove = []
        silent_synth_start = petri_utils.add_transition(net, "silent_synth_start", None)
        silent_synth_stop = petri_utils.add_transition(net, "silent_synth_stop", None)

        # connect net with start of the synthesized net
        for arc in net_placeholder.in_arcs:
            source = arc.source
            arcs_to_remove.append(arc)
            petri_utils.add_arc_from_to(fr=source, to=silent_synth_start, net=net)
        for arc in synth_start.out_arcs:
            target = arc.target
            arcs_to_remove.append(arc)
            petri_utils.add_arc_from_to(fr=silent_synth_start, to=target, net=net)
        
        # connect end of the synthesized net with net
        for arc in net_placeholder.out_arcs:
            target = arc.target
            arcs_to_remove.append(arc)
            petri_utils.add_arc_from_to(fr=silent_synth_stop, to=target, net=net)
        for arc in synth_stop.in_arcs:
            source = arc.source
            arcs_to_remove.append(arc)
            petri_utils.add_arc_from_to(fr=source, to=silent_synth_stop, net=net)

        for arc in arcs_to_remove:
            petri_utils.remove_arc(net, arc)
        
        return net


