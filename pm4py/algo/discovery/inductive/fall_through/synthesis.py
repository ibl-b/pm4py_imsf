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

from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructureUVCL,
)
from pm4py.algo.discovery.inductive.fall_through.abc import FallThrough
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from copy import deepcopy
from typing import Optional, Tuple, List, Dict, Any, Counter
from collections import defaultdict
from itertools import combinations
from pulp import LpProblem, LpMinimize, LpVariable, lpSum
import numpy as np
import uuid 
import pandas as pd
from datetime import datetime, timedelta

import pm4py


class SynthesisUVCL(FallThrough[IMDataStructureUVCL]):

    """
    Class for creating a synthesized net based on the token-based algorithm described in:
    Synthesizing Petri Nets from Labelled Petri Nets. Robin Bergenthum and Jakub Kovář, 2025.
    
    This class is used as a fallthrough in the inductive miner variant 
    "Inductive Miner with Synthesized Fallthrough Step" (IMSF).
    
    The inductive miner variant is described in:
    Ein Inductive Miner mit synthesebasiertem Fall-Through-Schritt. Lisa Berger, 2025
    """

    def _filter_traces(
        obj: IMDataStructureUVCL, threshold: float
    ) -> IMDataStructureUVCL:
        """
        Filters traces based on the given threshold. Ensures that no more than 1% of the unique traces are removed.
        
        This internal method is used for testing and optimizing the synthesized net.
        The threshold can be set in the class method `apply()`.

        :param obj: UVCL datastructure representing the given log.
        :param threshold: The threshhold value for filtering traces. (0 <= threshold <= 1)
        :return: The filtered UVCL-log.
        """
        if threshold == 0:
            return obj
        traces = obj.data_structure
        total_traces = sum(traces.values())
        trace_threshold = total_traces * threshold
        sorted_counts = sorted(traces.values())  
        cumulative_counts = np.cumsum(sorted_counts)
        
        # Remove max 0.5% of unique traces
        max_removable_threshold = sorted_counts[np.searchsorted(cumulative_counts, total_traces / 200)]
        final_threshold = min(trace_threshold, max_removable_threshold)
        filtered_traces = deepcopy(traces)
        for trace in traces:
            if filtered_traces[trace] < final_threshold:
                del filtered_traces[trace]

        return IMDataStructureUVCL(filtered_traces)

    
    def _set_start_and_end(obj: IMDataStructureUVCL) -> IMDataStructureUVCL:
        """
        Sets an artificial start and end activity for each trace.

        :param obj: UVCL datastructure representing the given log.
        :return: UVCL datastrucutre of the log with artificial start and end activities.
        """
        traces = obj.data_structure
        new_traces = Counter()
        for trace, count in traces.items():
            new_trace = ("Start",) + trace + ("Stop",)
            new_traces[new_trace] = count

        return IMDataStructureUVCL(new_traces)

    def _init_linear_nets(
        obj: IMDataStructureUVCL
    ) -> Tuple[List[Dict[str, Any]], List[int], List[int], int]:
        """
        Calculates the linear net for each trace in the given log as base for calculating valid token trails.
        This function creates a single list of entries, where each entry corresponds to an activity 
        and contains the following information:
            - the `activity` being executed,
            - the `token_places` before and after the activity.

        
        :param obj: UVCL datastructure representing the given log.
        :return: A list of dictionaries (List[Dict[str, any]]), containing every activity in each trace of the log with the corresponding token place numbers before and after the activity. 
        :return: A list of place numbers (List[int]) representing the initial place for each token trail.
        :return: A list of place numbers (List[int]) representing the final place for each token trail.
        :return: The total number of token places.
        """
        linear_nets = []
        initial_token_places = []
        final_token_places = []

        current_place = 0
        log = getattr(obj, "data_structure", [])

        for trace in log:
            
            start_place = current_place
            initial_token_places.append(start_place)

            for activity in trace:
                current_place += 1
                linear_nets.append(
                    {
                        "token_places": [start_place, current_place],
                        "activity": activity,
                    }
                )
                start_place = current_place

            final_token_places.append(start_place)
            current_place += 1

        return (
            linear_nets,
            initial_token_places,
            final_token_places,
            current_place,
        )

    def _get_rise_places(
        linear_nets: List[Dict[str, Any]]
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Calculates the rise places as a basis for the rise equations.
        To ensure equal rise for every activity, the entries in the token trails list are grouped by activity,
        mapping each activity to a list of (start_place, end_place) tuples.
        
        :param linear_nets: A list of dictionaries, each containing an "activity" and its corresponding 
                     "token_places" (start and end place).  
        :return: A dictionary mapping each activity to a list of tuples (start_place, end_place)
        """
        rise_places = defaultdict(list)

        for net_part in linear_nets:
            activity = net_part["activity"]
            start_place, end_place = net_part["token_places"]
            rise_places[activity].append((start_place, end_place))

        return rise_places


    def _init_net(
        obj: IMDataStructureUVCL,
    ) -> Tuple[PetriNet, Marking, Marking]:
        """
        Inits the PetriNet by creating a transition for every activity in the log.
        :param obj: UVCL datastructure representing the given log.  
        :return: A Petri Net with the transitions of the log
        :return: An empty initial marking
        :return: An empty final marking
        """
        net = PetriNet("imsfs")
        im = Marking()
        fm = Marking()
        trans_map = {}

        activities = sorted(
            list(set(activity for trace in obj.data_structure for activity in trace))
        )
        for act in activities:
            label = act
            trans_map[act] = PetriNet.Transition(act, label)
            net.transitions.add(trans_map[act])

        return net, im, fm


    def _solve_ilp_problem(
        linear_nets: List[Dict[str, Any]],
        initial_places: List[int],
        final_places: List[int],
        num_places: int,
        create_wf: bool
    ) -> defaultdict:
        """
        Defines a minimization problem with the following constraints:
        - prohibit the empty token trail
        - ensure empty final places
        - equal rise for same activities
        - limit token production/consumption to at most one token per place
        - ensure equal initial marking

        When a solution is found, an additional constraint is added to prohibit a solution that is bigger than the one just found.

        The constraints are based on [source]. Additional constraints are added to ensure the resulting net is a 
        workflow net.
        
        :param linear_nets: A list of dictionaries (List[Dict[str, any]]), containing every activity in each trace of the log with the corresponding token place numbers before and after the activity 
        :param initial_places: A list of place numbers representing the initial place for each token trail.
        :param final_places: A list of place numbers representing the final place for each token trail.
        :param num_places: The total number of token places.
        :param create_wf: Include constraints for workflow net (default True) 
        """

        problem = LpProblem("Minimize_Marking", LpMinimize)
        place_vars = [
            LpVariable(f"p_{i}", lowBound=0, cat="Integer")
            for i in range(0, num_places)
        ]

        problem += lpSum(place_vars), "Total Marking of the Token Trails"

        SynthesisUVCL._add_initial_constraints(
            problem, place_vars, initial_places, final_places, create_wf
        )

        rise_places = SynthesisUVCL._get_rise_places(linear_nets)
        SynthesisUVCL._add_rise_constraints(problem, place_vars, rise_places, create_wf)

        solutions = defaultdict(list)
        solution_no = 0
        problem.solve()

        while problem.status == 1:
            
            # when not creating a workflow net, the algorithm takes a long time to stop when facing huge data
            # solution limit can be modified here
            if create_wf or solution_no < 40:
                SynthesisUVCL._update_solutions(place_vars, solutions, solution_no, initial_places)
                SynthesisUVCL._add_new_constraint(problem, solution_no)
                solution_no += 1
                problem.solve()
            else:
                break

        net_places = SynthesisUVCL._get_net_places_from_ilp_solution(
            solutions, rise_places
        )

        return net_places


    def _add_initial_constraints(
        problem: LpProblem,
        vars: List[LpVariable],
        initial_places: List[int],
        final_places: List[int],
        create_wf: bool
    ):
        """
        Adds the basic constraints to the LP Problem: 
        - prohibit the empty token trail
        - equal initial marking for all token trails
        - empty final places (optional).

        The first two constraints are based on [source].
        The third constraint is added to ensure the resulting net is a workflow net.

        :param problem: The Lp Problem  
        :param vars: The list of variables of the problem
        :param initial_places: A list of place numbers of the initial places of the token trails 
        :param final_places: A list of place numbers of the final places of the token trails
        :param create_wf: Optional include constraints to create a workflow net (empty final places) 
        """
        problem += lpSum(vars) >= 1, "Prohibit the empty trail"
 
        if len(initial_places) > 1:
            first_place = vars[initial_places[0]]
            for i in range(1, len(initial_places)):
                problem += (
                    first_place == vars[initial_places[i]],
                    f"Equal initial marking iteration {i}",
                )

        if create_wf and len(final_places) > 1:
            problem += (
                lpSum(vars[i] for i in final_places) == 0,
                "Empty final places",
        )


    def _add_rise_constraints(
        problem: LpProblem,
        vars: List[LpVariable],
        rise_places: Dict[str, List[Tuple[int, int]]],
        create_wf: bool
    ):
        """
        Adds the rise constraints to the LP Problem: 
        - equal rise for same activities
        - producing / consuming maximum 1 token (optional)

        The first constraint is based on [source].
        The second constraint is added to ensure the resulting net is a workflow net with at most one marking per net place.

        :param problem: The LP Problem  
        :param vars: The list of variables of the problem
        :param rise_places: A dictionary mapping each activity to a list of tuples (start_place, end_place) 
        :param create_wf: Optional include constraints to create a workflow net (limit edge weight) 
        """
        for activity, places in rise_places.items():
            if len(places) > 1:
                first_diff = vars[places[0][1]] - vars[places[0][0]]
                
                if create_wf:
                    problem += (first_diff >= -1, f"Activity {activity}: rise min constraint")
                    problem += (first_diff <= 1, f"Activity {activity}: rise max constraint")

                for i in range(1, len(places)):
                    current_diff = vars[places[i][1]] - vars[places[i][0]]
                    problem += (
                        first_diff == current_diff,
                        f"Activity {activity}: equal rise iteration {i}",
                    )
            else:
                if len(places) == 1 and create_wf:
                    rise = vars[places[0][1]] - vars[places[0][0]]
                    problem += (rise >= -1, f"Activity {activity}: rise min constraint")
                    problem += (rise <= 1, f"Activity {activity}: rise max constraint")


    def _update_solutions(
        place_vars: List[LpVariable], solutions: defaultdict, solution_no: int, initial_places: List[int]
    ):
        """
        Adds the current LP solution to the dictionary of stored solutions. 
        Only places that are either initial or have a marking != 0 are included in the solution.

        :param place_vars: List of LP variables representing places in the net. 
        :param solutions: A dictionary that stores multiple LP solutions.
        :param solution_no: The index of the currently found solution
        :param initial_places: A list of initial places of the token trails
        """
        previous_value = 0
        for var in place_vars:
            place_no = int(var.name.split("_")[1])
            current_value = var.varValue
            attribute = SynthesisUVCL._get_attribute(
                place_no, current_value, previous_value, initial_places
            )

            if attribute is not None:
                solutions[solution_no].append(
                    {
                        "place_no": place_no,
                        "rise": current_value - previous_value,
                        "attribute": attribute,
                    }
                )
            previous_value = current_value

    def _get_attribute(
        place_no: int, current_value: float, previous_value: float, initial_places: List[int]
    ) -> Optional[str]:
        """
        Returns an attribute describing a token place in a solution of a LP problem:
        - initial: if it's an initial place of one of the token trails
        - added: if the activity before this place added a token (rise = 1)
        - removed: if the activity before this place removed a token (rise = -1)
        - potential loop: if a self loop is possible (removing and adding a token) 

        :param place_no: The number of the place to check  
        :param current_value: The number of tokens in the place to check
        :param previous_value: The number of tokens in the place before 
        :param initial_places: A list containing the initial place numbers of the token trails 
        :returns An attribute describing the rise condition of this place or None
        """
        if place_no == 0:
            return "initial"
        if current_value - previous_value > 0:
            return "added"
        if current_value - previous_value < 0 and place_no not in initial_places:
            return "removed"
        if current_value - previous_value == 0 and current_value > 0:
            return "potential_loop"
        return None


    def _add_new_constraint(problem: LpProblem, solution_no: int):
        """
        Adds a new constraint to the LP problem to prohibit a solution with the same places and greater values.

        :param problem: The LP Problem   
        :param solution_no: The index of the current solution.  
        """
        new_vars = 0
        k = 2

        p_vars = []
        for var in problem.variables():
            if not var.name.startswith("p"):
                break
            if var.varValue != 0:
                p_vars.append(var)

        for var in p_vars:
            x = LpVariable(f"x_{solution_no}_{var}", lowBound=0, cat="Binary")
            deviation = var - var.varValue
            problem += (deviation + k * x >= 0, f"Solution Nr {solution_no}, {x} constraint greater zero",)
            problem += (deviation + k * x <= k - 1, f"Solution Nr {solution_no}, {x} constraint less than {k - 1}",)
            new_vars += x

        problem += (new_vars >= 1, f"Exclude_Solution_{solution_no}",)


    def _get_net_places_from_ilp_solution(
        solutions: defaultdict, rise_places: Dict[str, List[Tuple[int, int]]]
    ) -> defaultdict:
        """
        Extracts the petri net place information for each ILP solution.
        This function identifies the corresponding activities in the rise places, depending on the attributes of the token places.

        For each net place, the following information is stored:
        - 'marking': Initial marking of this place. 
        - 'edges_in': List of activities, that add a token to this place 
        - 'edges_out': List of activities, that consume a token from this place
        -'potential_loop': List of activities, that potentially both add and consume a token 

        For token places with the attribute 'potential loop' the function searches for repeated occurrences of the same activity 
        at consecutive place numbers to detect loops. 

        :param solutions: A dictionary with the solutions of the LP problem
        :param rise_places: A dictionary mapping each activity to a list of tuples (start_place, end_place)   
        :return: A defaultdict mapping each solution to a dict representing the corresponding Petri net place.  
        """
        net_places = defaultdict(
            lambda: {"edges_in": [], "edges_out": [], "potential_loop": [], "marking": 0}
        )
        for solution, token_places in solutions.items():
            for token_place in token_places:
                place_no = token_place["place_no"]
                if place_no == 0:
                    net_places[solution]["marking"] = token_place["rise"]
                else:
                    for activity, rise_tuples in rise_places.items():
                        for tup in rise_tuples:

                            if place_no == tup[1]:

                                if token_place["attribute"] == "added":
                                    key = "edges_in"
                                elif token_place["attribute"] == "removed":
                                    key = "edges_out"
                                elif token_place["attribute"] == "potential_loop":
                                    key = "potential_loop"
                                if key == "potential_loop":
                                    net_places[solution][key].append((activity, place_no))
                                else:
                                    existing_activities = [a for (a, _) in net_places[solution][key]]
                                    if activity not in existing_activities:
                                        net_places[solution][key].append((activity, token_place["rise"]))   

        # remove dead ends if there are other places with arcs in
        all_non_dead_end_activities = set()
        for solution, data in net_places.items():
            if data["edges_out"]:
                all_non_dead_end_activities.update([a for a, _ in data["edges_in"]])

        filtered_net_places = {}
        for solution, data in net_places.items():
            if data["edges_out"]:
                    filtered_net_places[solution] = data
            else:
                edges_in = [activity for activity, _ in data["edges_in"]]
                if "Stop" in edges_in:
                    filtered_net_places[solution] = data
                elif not any(act in all_non_dead_end_activities for act in edges_in):
                    filtered_net_places[solution] = data
        net_places = filtered_net_places

        # detect potential short loops
        inserted_short_loops = defaultdict(list)
        # activities already reachable in the net shall be ignored
        already_connected = set(activity for data in net_places.values() for activity, _ in data["edges_in"])
        for solution in net_places:
            previous_activity, previous_place = None, None
            in_sequence = False
            in_first_loop = False
            short_loop_start_candidate = None
            potential_loop = sorted(net_places[solution]["potential_loop"], key=lambda x: x[1])

            # searches for short loops that appear at the beginning of a sequence
            for activity, place in potential_loop:
                if previous_activity is not None and place - previous_place == 1:
                    if not in_sequence:
                        in_sequence = True
                        short_loop_start_candidate = previous_activity
                        in_first_loop = True
                    if activity == short_loop_start_candidate and in_first_loop:
                            existing_in = [a for a, _ in net_places[solution]["edges_in"]]
                            existing_out = [a for a, _ in net_places[solution]["edges_out"]]

                            if activity not in already_connected:
                                if activity not in existing_in:
                                    net_places[solution]["edges_in"].append((activity, 1))
                                    inserted_short_loops[activity].append(solution)
                                if activity not in existing_out:
                                    net_places[solution]["edges_out"].append((activity, 1))
                    else:
                        in_first_loop = False
                    
                    previous_activity, previous_place = activity, place
                        
                else:
                    in_sequence = False
                    previous_activity, previous_place = activity, place
                

        # check if there where double short loop insertions:
        for activity, solutions in inserted_short_loops.items():
            if len(solutions) > 1:
                to_remove = set()
                # remove the short loop if edges_in of one place is a subset of the other place 
                for sol1, sol2 in combinations(solutions, 2):
                    set1 = set(a for a, _ in net_places[sol1]["edges_in"])
                    set2 = set(a for a, _ in net_places[sol2]["edges_in"])

                    if set1 < set2: 
                        to_remove.add(sol1)
                    elif set2 < set1:
                        to_remove.add(sol2)
                # delete the rest
                for solution in to_remove:
                    net_places[solution]["edges_in"] = [
                        (a, w) for a, w in net_places[solution]["edges_in"] if a != activity]
                    net_places[solution]["edges_out"] = [
                        (a, w) for a, w in net_places[solution]["edges_out"]if a != activity]
        
        return net_places

    
    def _insert_net_places(
        net: PetriNet, im: Marking, fm: Marking, net_places: defaultdict
    ) -> Tuple[PetriNet, Marking, Marking]:
        """
        Inserts places into the given Petri net based on the information in net_places. 

        For each entry in `net_places`, a new place is added to the net and connected to the
        corresponding transitions. If the place has an initial marking, it is added to the initial marking.

        :param PetriNet: The Petri to which the places will be added
        :param im: The initial marking of the Petri net
        :param fm: The final marking of the Petri net (Remains unchanged. The final marking gets calculated in the IMSFS class when combining the nets.)
        :param net_places: A defaultdict mapping each ILP solution to a dict representing the corresponding Petri net place.
        :return: The updated Petri net 
        :return: The updated initial marking
        :return: The final marking  
        """
        for place_no, place in net_places.items():
            if not (place["edges_in"] or place["edges_out"]):
                raise ValueError(f"Invalid place without any edges.")
            else:
                net_place = petri_utils.add_place(net)

                if place["marking"] > 0:
                    im[net_place] = int(place["marking"])
                
                if place["edges_in"] and place["edges_out"]:
                    for (activity, rise) in place["edges_in"]:
                        transition = petri_utils.get_transition_by_name(net, activity)
                        petri_utils.add_arc_from_to(
                            fr=transition, to=net_place, net=net, weight=int(abs(rise))
                        )
                    for (activity, rise) in place["edges_out"]:
                        transition = petri_utils.get_transition_by_name(net, activity)
                        petri_utils.add_arc_from_to(
                            fr=net_place, to=transition, net=net, weight=int(abs(rise))
                        )
                
                # places only with acs in
                elif place["edges_in"]:
                    for (activity, rise) in place["edges_in"]:
                        transition = petri_utils.get_transition_by_name(net, activity)
                        petri_utils.add_arc_from_to(
                                fr=transition, to=net_place, net=net, weight=int(abs(rise))
                            )
                     
                elif place["edges_out"]:
                    for (activity, rise) in place["edges_out"]:
                        transition = petri_utils.get_transition_by_name(net, activity)
                        petri_utils.add_arc_from_to(fr=net_place, to=transition, net=net, weight=int(abs(rise)))
                   
                else:
                    petri_utils.remove_place(net, net_place)

        return net, im, fm

    def _create_wf_net(
        net: PetriNet, im: Marking, fm: Marking
    ) -> Tuple[PetriNet, Marking, Marking]:
        """
        Converts the given Petri net to a workflow net. 
        
        - Adds a sink place if missing
        - Adds a flower model for unconnected transitions
        
        :param PetriNet: The Petri to which the places will be added
        :param im: The initial marking of the Petri net
        :param fm: The final marking of the Petri net (Remains unchanged. The final marking gets calculated in the IMSFS class when combining the nets.)
        :param net_places: A defaultdict mapping each ILP solution to a dict representing the corresponding Petri net place.
        :return: The updated Petri net 
        :return: The updated initial marking
        :return: The final marking  
        """

        # check sink place / add if not existing
        stop = petri_utils.get_transition_by_name(net, "Stop")
        start = petri_utils.get_transition_by_name(net, "Start")
        if len(stop.out_arcs) == 0:
            sink = petri_utils.add_place(net, "sink")
            petri_utils.add_arc_from_to(fr=stop, to=sink, net=net)
        elif len(stop.out_arcs) == 1:
            sink = next(iter(stop.out_arcs)).target
        
        # remove the artifical start/stop transition
        stop.label=None
        start.label=None

        # check for unconnected transitions
        unconnected_transitions = []
        for transition in net.transitions:
            if not transition.in_arcs and not transition.out_arcs:
                unconnected_transitions.append(transition)
        
        # use flower model for unconnected transitions    
        if unconnected_transitions:
            flower = petri_utils.add_place(net, "flower")
            for transition in unconnected_transitions:
                petri_utils.add_arc_from_to(fr=transition, to=flower, net=net)
                petri_utils.add_arc_from_to(fr=flower, to=transition, net=net)
            petri_utils.add_arc_from_to(fr=petri_utils.get_transition_by_name(net, "Start"), to=flower, net=net)
            petri_utils.add_arc_from_to(fr=flower, to=stop, net=net)

        cleaned_net = petri_utils.remove_unconnected_components(net)

        from pm4py.objects.petri_net.utils import final_marking 
        
        fm = final_marking.discover_final_marking(cleaned_net)

        return cleaned_net, im, fm
    
    def uvcl_to_eventlog_from_sequences(uvcl: Counter[Tuple[Any]]):
        """
        Converts an UVCL datastructure to an event log. Used to apply conformance
        checking algorithms for the synthesized nets.

        :param uvcl: uvcl representation of the event log
        :return: event log
        """
        data = []
        base_time = datetime.now()

        case_counter = 0
        for variant, freq in uvcl.items():
            for _ in range(freq):
                case_id = f"case_{case_counter}"
                timestamp = base_time
                for idx, activity in enumerate(variant):
                    data.append({
                    "case:concept:name": case_id,
                    "concept:name": str(activity),
                    "time:timestamp": timestamp,
                    "lifecycle:transition": "complete"
                    })
                    timestamp += timedelta(seconds=1)
                case_counter += 1

        df = pd.DataFrame(data)
        
        pm4py.write_xes(df, 'new_log.xes')
    
    @classmethod
    def holds(cls, obj: IMDataStructureUVCL, parameters: Optional[Dict[str, Any]] = None) -> bool:
        return True
       
    @classmethod
    def apply(
        cls,
        obj: IMDataStructureUVCL,
        pool=None,
        manager=None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple [
        ProcessTree, List[IMDataStructureUVCL]]
    ]:  

        obj_with_ss = SynthesisUVCL._set_start_and_end(obj)
        
        # optional: filter traces if the resulting net is not satisfying. 
        filtered_obj = SynthesisUVCL._filter_traces(obj_with_ss, 0) 
        
        # set this to False to try creating a net without limited arc weights and empty final places constraints
        create_wf = False

        (
            linear_nets,
            initial_places,
            final_places,
            num_places,
        ) = SynthesisUVCL._init_linear_nets(filtered_obj)
 
        # if using filter, you could try initing the net with obj_with_ss, so every activity of the original object will 
        # be part of the resulting net (activities that are filtered will be as part of a flower model)
        net, im, fm = SynthesisUVCL._init_net(obj_with_ss)

        net_places = SynthesisUVCL._solve_ilp_problem(
            linear_nets, initial_places, final_places, num_places, create_wf
        )

        net, im, fm = SynthesisUVCL._insert_net_places(net, im, fm, net_places)
        workflow_net, im, fm = SynthesisUVCL._create_wf_net(net, im, fm)
        pt = PlaceholderTree(workflow_net, im, fm)
        
        return pt, []
    

class PlaceholderTree(ProcessTree):
    """
    Class for creating a process Tree and storing a synthesized net with a label and unique id.
    
    This class is used for returning a placeholder process tree to the inductive miner algorithm, 
    while storing the workflow net created by this fallthrough class.
    
    """

    def __init__(self, petri_net: PetriNet, im: Marking, fm: Marking):
        tree_id = str(uuid.uuid4())
        super().__init__(label=f"synth_placeholder_{tree_id}")
        self.petri_net = petri_net
        self.im = im
        self.fm = fm
