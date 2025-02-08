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
    IMDataStructureDFG,
)
from pm4py.algo.discovery.inductive.fall_through.abc import FallThrough
from typing import Optional, Tuple, List, Dict, Any
from collections import defaultdict
from pm4py.objects.process_tree.obj import ProcessTree, Operator
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus

from pm4py.objects.dfg.obj import DFG
from copy import deepcopy

# pulp, scipy möglich -> bei LP-Miner schauen, was von PM4Py verwendet wird -> nutzt solver und petrinetze
from pm4py.util.lp import solver as synth_solver

from pm4py.objects.petri_net.utils import petri_utils
from pm4py import vis


from pm4py.convert import convert_to_process_tree

# Ziel


# TODO KLassenmethode holds implementieren? Prüft, ob der Ansatz für die gegebene Datenstruktur (IMDataStructureUVCL) anwendbar ist
# TODO in der Factory Datei registrieren
# TODO wie berechne ich die finale Markierung??
class SynthesisUVCL(FallThrough[IMDataStructureUVCL]):

    def _filter_traces(
        obj: IMDataStructureUVCL, threshold: float
    ) -> IMDataStructureUVCL:
        # Filter infrequent traces to keep the synthesized net a bit simpler
        number_of_traces = obj.data_structure.total
        trace_treshhold = number_of_traces * threshold
        filtered_traces = deepcopy(obj.data_structure)
        for trace in filtered_traces:
            if trace.count < trace_treshhold:
                del filtered_traces[trace]
        # dfg sollte automatisch erstellt werden
        return IMDataStructureUVCL(obj.data_structure)

    def _get_token_trails(
        obj: IMDataStructureUVCL, parameters: Optional[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[int], int]:
        token_trails = []
        initial_places = []
        current_place = 0
        log = getattr(obj, "data_structure", [])

        # TODO prüfen, wo Fehler geworfen werden sollten, wenn Eingabe nicht wie erwartet (UVCL, Counter...)
        for trace in log:
            current_place += 1
            start_place = current_place
            initial_places.append(start_place)

            for activity in trace:
                current_place += 1
                token_trails.append(
                    {
                        "token_places": [start_place, current_place],
                        "activity": activity,
                    }
                )
                start_place = current_place

        return token_trails, initial_places, current_place

    def _get_rise_places(
        token_trails: List[Dict[str, Any]]
    ) -> Dict[str, List[Tuple[int, int]]]:

        rise_places = defaultdict(list)

        for trail in token_trails:
            activity = trail["activity"]
            start_place, end_place = trail["token_places"]
            rise_places[activity].append((start_place, end_place))

        return rise_places

    @classmethod
    def holds(
        cls, obj: IMDataStructureUVCL, parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        # TODO hier prüfen ob Teillog ohne taus ist und sound
        return True

    # Initiales Netz mit Transitionen erstellen
    def _init_net(
        obj: IMDataStructureUVCL,
    ) -> Tuple[PetriNet, List[str], Marking, Marking]:
        net = PetriNet("imsfs")
        im = Marking()
        fm = Marking()
        # source = PetriNet.Place("source")
        # sink = PetriNet.Place("sink")
        # net.places.add(source)
        # net.places.add(sink)
        # TODO initial/final Marking source und sink 1 oder 0?
        # im[source] = 1
        # fm[sink] = 1

        trans_map = {}

        # Transitionen hinzufügen
        activities = sorted(
            list(set(activity for trace in obj.data_structure for activity in trace))
        )
        for act in activities:
            label = act
            trans_map[act] = PetriNet.Transition(act, label)
            net.transitions.add(trans_map[act])

        return net, activities, im, fm

    def _add_initial_constraints(
        problem: LpProblem, vars: List[LpVariable], initial_places: List[int]
    ):
        # Nebenbedingung 1: leeren Token Trail verbieten
        problem += lpSum(vars) >= 1, "Prohibit the empty trail"

        # Nebenbedingungen: Initialmarkierung gleich
        if len(initial_places) > 1:
            first_place = vars[initial_places[0] - 1]
            for i in range(1, len(initial_places)):
                problem += (
                    first_place == vars[initial_places[i] - 1],
                    f"Equal initial marking iteration {i}",
                )

    def _add_rise_constraints(
        problem: LpProblem,
        vars: List[LpVariable],
        rise_places: Dict[str, List[Tuple[int, int]]],
    ):
        for activity, places in rise_places.items():
            if len(places) > 1:
                first_diff = vars[places[0][1] - 1] - vars[places[0][0] - 1]
                for i in range(1, len(places)):
                    current_diff = vars[places[i][1] - 1] - vars[places[i][0] - 1]
                    problem += (
                        first_diff == current_diff,
                        f"Activity {activity}: equal rise iteration {i}",
                    )

    def _get_attribute(
        place_no: int, current_value: float, previous_value: float
    ) -> Optional[str]:
        if place_no == 1:
            return "initial"
        if current_value - previous_value > 0:
            return "added"
        if current_value - previous_value < 0:
            return "removed"
        return None

    # TODO bei consume eventuell doch den Folgeplatz abspeichern um konsistent zu sein?
    def _update_solutions(
        place_vars: List[LpVariable], solutions: defaultdict, solution_no: int
    ):
        previous_value = -1
        for var in place_vars:
            place_no = int(var.name.split("_")[1])
            current_value = var.varValue
            attribute = SynthesisUVCL._get_attribute(
                place_no, current_value, previous_value
            )

            if attribute is not None:
                solutions[solution_no].append(
                    {
                        "place_no": place_no,
                        "marking": current_value,
                        "attribute": attribute,
                    }
                )
            previous_value = current_value

    # Verhindern, dass token trail gefunden wird, der größer als der aktuell gefundene ist
    def _add_new_constraint(problem: LpProblem, solution_no: int):
        non_zero_var_count = 0
        new_constraint = 0
        for var in problem.variables():
            if var.varValue != 0:
                new_constraint += var
                non_zero_var_count += 1
        problem += (
            new_constraint <= non_zero_var_count - 1,
            f"Prohibit token trail greater than solution {solution_no}",
        )

    def _get_net_places_from_ilp_solution(
        solutions: defaultdict, rise_places: Dict[str, List[Tuple[int, int]]]
    ) -> defaultdict:
        net_places = defaultdict(
            lambda: {"edges_in": [], "edges_out": [], "marking": int}
        )
        for solution, token_places in solutions.items():
            for token_place in token_places:
                place_no = token_place["place_no"]
                if place_no == 1:
                    net_places[solution]["marking"] = token_place["marking"]
                else:
                    for activity, rise_tuples in rise_places.items():
                        for tup in rise_tuples:

                            if place_no == tup[1]:

                                key = (
                                    "edges_in"
                                    if token_place["attribute"] == "added"
                                    else "edges_out"
                                )
                                if activity not in net_places[solution][key]:
                                    net_places[solution][key].append(activity)
        return net_places

    def _solve_ilp_problem(
        token_trails: List[Dict[str, Any]], initial_places: List[int], num_places: int
    ) -> defaultdict:

        # Problem und Variablen
        problem = LpProblem("Minimize_Marking", LpMinimize)
        place_vars = [
            LpVariable(f"p_{i}", lowBound=0, cat="Integer")
            for i in range(1, num_places + 1)
        ]

        # Zielfunktion
        problem += lpSum(place_vars), "Total Marking of the Token Trails"

        # Nebenbedingungen: Leerer Token Trail, Initialmarkierung
        SynthesisUVCL._add_initial_constraints(problem, place_vars, initial_places)

        # Nebenbedingung: gleicher Rise für gleiche Activities
        rise_places = SynthesisUVCL._get_rise_places(token_trails)
        SynthesisUVCL._add_rise_constraints(problem, place_vars, rise_places)

        solutions = defaultdict(list)
        solution_no = 0
        problem.solve()

        while problem.status == 1:
            SynthesisUVCL._update_solutions(place_vars, solutions, solution_no)
            SynthesisUVCL._add_new_constraint(problem, solution_no)
            solution_no += 1
            problem.solve()

        net_places = SynthesisUVCL._get_net_places_from_ilp_solution(
            solutions, rise_places
        )

        return net_places

    # wird angewandt, wenn holds true zurückgibt
    @classmethod
    # cls: die Klasse selbst (wenn es kein Objekt der Klasse gibt, ansonsten self), obj = event log, pool = multiprocessing pool (falls mehrere Prozesse parallel), manager = multiprocessing manager, optional zusätzliche Parameter??
    # -> Tuple[PetriNet, Marking, Marking]
    def apply(
        cls,
        obj: IMDataStructureUVCL,
        pool=None,
        manager=None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[
        ProcessTree
    ]:  # Optional[Tuple[ProcessTree, List[IMDataStructureUVCL]]]:

        filtered_obj = SynthesisUVCL._filter_traces(obj, 0.1)
        # Token Trails erstellen
        token_trails, initial_places, num_places = SynthesisUVCL._get_token_trails(
            filtered_obj, parameters
        )

        # Netz erstellen #TODO wird activities überhaupt benötigt?
        # TODO Funktion add_transition aus petri net utils verwenden?
        net, activities, im, fm = SynthesisUVCL._init_net(filtered_obj)

        net_places = SynthesisUVCL._solve_ilp_problem(
            token_trails, initial_places, num_places
        )

        for place_no, place in net_places.items():
            if not (place["edges_in"] or place["edges_out"]):
                raise ValueError(f"Invalid place without any edges.")
            elif place["edges_in"] and place["edges_out"]:
                net_place = petri_utils.add_place(net)
                for activity in place["edges_in"]:
                    transition = petri_utils.get_transition_by_name(net, activity)
                    petri_utils.add_arc_from_to(fr=transition, to=net_place, net=net)
                for activity in place["edges_out"]:
                    transition = petri_utils.get_transition_by_name(net, activity)
                    petri_utils.add_arc_from_to(fr=net_place, to=transition, net=net)
            elif place["edges_in"]:
                net_place = petri_utils.add_place(net)
                for activity in place["edges_in"]:
                    transition = petri_utils.get_transition_by_name(net, activity)
                    petri_utils.add_arc_from_to(fr=transition, to=net_place, net=net)
                # sink = PetriNet.Place("sink")
                # net.places.add(sink)
            elif place["edges_out"]:
                net_place = petri_utils.add_place(net)
                for activity in place["edges_out"]:
                    transition = petri_utils.get_transition_by_name(net, activity)
                    petri_utils.add_arc_from_to(fr=net_place, to=transition, net=net)

        vis.view_petri_net(net, im, fm, format="svg")

        pt = convert_to_process_tree(net, im, fm)

        return pt
