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
from typing import Optional, Tuple, List, Dict, Any, Counter
from collections import defaultdict
from pm4py.objects.process_tree.obj import ProcessTree, Operator
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus

from pm4py.objects.dfg.obj import DFG
from copy import deepcopy

import numpy as np

import uuid #TODO testen, ob es hier was pm4py internes gibt

# pulp, scipy möglich -> bei LP-Miner schauen, was von PM4Py verwendet wird -> nutzt solver und petrinetze
from pm4py.util.lp import solver as synth_solver

from pm4py.objects.petri_net.utils import petri_utils
from pm4py import vis

from pm4py.objects.dfg.obj import DFG


from pm4py.convert import convert_to_process_tree

# Ziel


# TODO KLassenmethode holds implementieren? Prüft, ob der Ansatz für die gegebene Datenstruktur (IMDataStructureUVCL) anwendbar ist
# TODO in der Factory Datei registrieren
# TODO wie berechne ich die finale Markierung? Was genau ist die final Marking?
# TODO prüfen: in manchen Funktionen wird als Marking immer 1 gewählt -> sind abweichende Zahlen möglich?
class SynthesisUVCL(FallThrough[IMDataStructureUVCL]):

    def _filter_traces(
        obj: IMDataStructureUVCL, threshold: float
    ) -> IMDataStructureUVCL:
        
        if threshold == 0:
            return obj
        traces = obj.data_structure
        total_traces = sum(traces.values())
        trace_threshold = total_traces * threshold
        sorted_counts = sorted(traces.values())  
        cumulative_counts = np.cumsum(sorted_counts)
        # max 2% entfernen
        quarter_threshold = sorted_counts[np.searchsorted(cumulative_counts, total_traces / 50)]
        final_threshold = min(trace_threshold, quarter_threshold)

        filtered_traces = deepcopy(traces)
        for trace in traces:
            if filtered_traces[trace] < final_threshold:
                del filtered_traces[trace]
        # dfg sollte automatisch erstellt werden
        return IMDataStructureUVCL(filtered_traces)

    # Start und Endaktivität einfügen
    def _set_start_and_end(obj: IMDataStructureUVCL) -> IMDataStructureUVCL:
        traces = obj.data_structure
        new_traces = Counter()
        for trace, count in traces.items():
            new_trace = ("Start",) + trace + ("Stop",)
            new_traces[new_trace] = count

        return IMDataStructureUVCL(new_traces)

    def _filter_short_loops(obj: IMDataStructureUVCL) -> IMDataStructureUVCL:
        traces = obj.data_structure
        new_traces = Counter()
        for trace, count in traces.items():
            new_trace = []
            prev_activity = None
            for activity in trace:
                if activity != prev_activity:
                    new_trace.append(activity)
                    prev_activity = activity
            new_traces[tuple(new_trace)] = count

        return IMDataStructureUVCL(new_traces)

    def _get_token_trails(
        obj: IMDataStructureUVCL, parameters: Optional[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[int], List[int], int]:
        token_trails = []
        initial_token_places = []
        final_token_places = []
        # initial_activities = []
        # final_activities = []

        current_place = 0
        log = getattr(obj, "data_structure", [])

        # TODO prüfen, wo Fehler geworfen werden sollten, wenn Eingabe nicht wie erwartet (UVCL, Counter...)
        for trace in log:
            # TODO brauche ich die initial activities tatsächlich?
            # initial_activity = trace[0]
            # final_activity = trace[-1]
            # if initial_activity not in initial_activities:
            # initial_activities.append(initial_activity)
            # if final_activity not in final_activities:
            # final_activities.append(final_activity)
            start_place = current_place
            initial_token_places.append(start_place)

            for activity in trace:
                current_place += 1
                token_trails.append(
                    {
                        "token_places": [start_place, current_place],
                        "activity": activity,
                    }
                )
                start_place = current_place

            final_token_places.append(start_place)
            current_place += 1

        return (
            token_trails,
            initial_token_places,
            final_token_places,
            current_place,
            # initial_activities,
            # final_activities,
        )

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
        problem: LpProblem,
        vars: List[LpVariable],
        initial_places: List[int],
        final_places: List[int],
    ):
        # Nebenbedingung 1: leeren Token Trail verbieten
        problem += lpSum(vars) >= 1, "Prohibit the empty trail"

        # Nebenbedingungen: Initialmarkierung gleich
        if len(initial_places) > 1:
            first_place = vars[initial_places[0]]
            for i in range(1, len(initial_places)):
                problem += (
                    first_place == vars[initial_places[i]],
                    f"Equal initial marking iteration {i}",
                )
        # Nebenbedingungen: Pätze müssen am nach Durchlauf aller Traces leer sein. Sink nach Stop wird dann aber nicht mehr erzeugt
        if len(final_places) > 1:
            problem += (
                lpSum(vars[i] for i in final_places) == 0,
                "Empty Places after each trace for wf-net",
        )

    def _add_rise_constraints(
        problem: LpProblem,
        vars: List[LpVariable],
        rise_places: Dict[str, List[Tuple[int, int]]],
    ):
        for activity, places in rise_places.items():
            if len(places) > 1:
                first_diff = vars[places[0][1]] - vars[places[0][0]]
                # Sicherstellen, das WF-Netz erzeugt wird mit maximal einer Marke pro Platz
                problem += (first_diff >= -1, f"Activity {activity}: rise min constraint")
                problem += (first_diff <= 1, f"Activity {activity}: rise max constraint")

                for i in range(1, len(places)):
                    current_diff = vars[places[i][1]] - vars[places[i][0]]
                    problem += (
                        first_diff == current_diff,
                        f"Activity {activity}: equal rise iteration {i}",
                    )
            else:
            # TODO: was wenn diese Aktivität nur einmal vorkommt?
                if len(places) == 1:
                    rise = vars[places[0][1]] - vars[places[0][0]]
                    problem += (rise >= -1, f"Activity {activity}: rise min constraint")
                    problem += (rise <= 1, f"Activity {activity}: rise max constraint")


    def _get_attribute(
        place_no: int, current_value: float, previous_value: float, initial_places: List[int]
    ) -> Optional[str]:
        if place_no == 0:
            return "initial"
        if current_value - previous_value > 0:
            return "added"
        if current_value - previous_value < 0 and place_no not in initial_places:
            return "removed"
        if current_value - previous_value == 0 and current_value == 1:
            return "potential_loop"
        return None

    # TODO bei consume eventuell doch den Folgeplatz abspeichern um konsistent zu sein?
    def _update_solutions(
        place_vars: List[LpVariable], solutions: defaultdict, solution_no: int, initial_places: List[int]
    ):
        previous_value = -1
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
            lambda: {"edges_in": [], "edges_out": [], "potential_loop": [], "marking": int}
        )
        for solution, token_places in solutions.items():
            for token_place in token_places:
                place_no = token_place["place_no"]
                if place_no == 0:
                    net_places[solution]["marking"] = token_place["marking"]
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
                                elif activity not in net_places[solution][key]:
                                    net_places[solution][key].append(activity) 
                                #if activity not in net_places[solution][key]:
                                    #net_places[solution][key].append(activity)
                                #elif key == "potential_loop":
                                    #net_places[solution][key].append(activity)
            # loops prüfen
            previous_activity, previous_place = None, None
            for activity, place in sorted(net_places[solution]["potential_loop"], key=lambda x: x[1]):
                if previous_activity == activity and place == previous_place + 1:
                    if activity not in net_places[solution]["edges_in"]:
                        net_places[solution]["edges_in"].append(activity)
                    if activity not in net_places[solution]["edges_out"]:
                        net_places[solution]["edges_out"].append(activity)
                    
                previous_activity, previous_place = activity, place
        return net_places

    def _solve_ilp_problem(
        token_trails: List[Dict[str, Any]],
        initial_places: List[int],
        final_places: List[int],
        num_places: int,
    ) -> defaultdict:

        # Problem und Variablen
        problem = LpProblem("Minimize_Marking", LpMinimize)
        place_vars = [
            LpVariable(f"p_{i}", lowBound=0, cat="Integer")
            for i in range(0, num_places)
        ]

        # Zielfunktion
        problem += lpSum(place_vars), "Total Marking of the Token Trails"

        # Nebenbedingungen: Leerer Token Trail, Initialmarkierung, Plätze am Ende leer
        SynthesisUVCL._add_initial_constraints(
            problem, place_vars, initial_places, final_places
        )

        # Nebenbedingung: gleicher Rise für gleiche Activities
        rise_places = SynthesisUVCL._get_rise_places(token_trails)
        SynthesisUVCL._add_rise_constraints(problem, place_vars, rise_places)

        solutions = defaultdict(list)
        solution_no = 0
        problem.solve()

        while problem.status == 1:
            SynthesisUVCL._update_solutions(place_vars, solutions, solution_no, initial_places)
            SynthesisUVCL._add_new_constraint(problem, solution_no)
            solution_no += 1
            problem.solve()

        net_places = SynthesisUVCL._get_net_places_from_ilp_solution(
            solutions, rise_places
        )

        return net_places

    def _insert_net_places(
        net: PetriNet, im: Marking, fm: Marking, net_places: defaultdict
    ) -> Tuple[PetriNet, Marking, Marking]:
        for place_no, place in net_places.items():
            if not (place["edges_in"] or place["edges_out"]):
                raise ValueError(f"Invalid place without any edges.")
            else:
                net_place = petri_utils.add_place(net)
                if place["marking"] > 0:
                    im[net_place] = int(place["marking"])
                # innerer Platz
                if place["edges_in"] and place["edges_out"]:
                    for activity in place["edges_in"]:
                        transition = petri_utils.get_transition_by_name(net, activity)
                        petri_utils.add_arc_from_to(
                            fr=transition, to=net_place, net=net
                        )
                    for activity in place["edges_out"]:
                        transition = petri_utils.get_transition_by_name(net, activity)
                        petri_utils.add_arc_from_to(
                            fr=net_place, to=transition, net=net
                        )
                # elif place.get("edges_in") == ["Stop"]:
                elif place["edges_in"]:
                    # transition = petri_utils.get_transition_by_name(net, "Stop")
                    # eigentlich bisher Plätze herausgefiltert, die nicht stop sind, um zusätzliche Sinks zu vermeiden
                    for activity in place["edges_in"]:
                        transition = petri_utils.get_transition_by_name(net, activity)
                        petri_utils.add_arc_from_to(
                            fr=transition, to=net_place, net=net
                        )
                    if place.get("edges_in") == ["Stop"]:
                        fm[net_place] = int(place["marking"])
                elif place.get("edges_out") == ["Start"]:
                    transition = petri_utils.get_transition_by_name(net, "Start")
                    petri_utils.add_arc_from_to(fr=net_place, to=transition, net=net)
                # kein valider Platz
                else:
                    petri_utils.remove_place(net, net_place)
        return net, im, fm

    def _create_wf_net(
        net: PetriNet, im: Marking, fm: Marking
    ) -> Tuple[PetriNet, Marking, Marking]:
        #source = []
        #sink = []
        #updated_im = Marking()
        #updated_fm = Marking()
        #is_loop_source = False

        #for place in net.places:
            #if len(place.out_arcs) > 0 and len(place.in_arcs) == 0:
                #source.append(place)
            #if len(place.in_arcs) > 0 and len(place.out_arcs) == 0:
                #sink.append(place)

        # source Plätze finden, bei denen ggf. Loop vorliegt
        #for place in im:
            #if place not in source:
                #for arc in place.out_arcs:
                    #if arc.target.name in initial_activities:
                        #source.append(place)
                        #is_loop_source = True
                        #break

        #if len(source) != 1 or is_loop_source:
            #new_source = petri_utils.add_place(net, "global_source")
            #for place in source:
                #silent_transition = petri_utils.add_transition(net, "t_source")
                #petri_utils.add_arc_from_to(
                    #fr=new_source, to=silent_transition, net=net
                #)
                #petri_utils.add_arc_from_to(fr=silent_transition, to=place, net=net)
            # Markierung updaten
            #source_marking = Marking()

            #for place in source:
                #source_marking[place] = im[place]
            #updated_im[new_source] = 0
            # Marken umverteilen von ursprünglichen Plätzen
            #while source_marking:
                #updated_im[new_source] += 1
                #remove_places = []
                #for place in list(source_marking):
                    #source_marking[place] -= 1
                    #if source_marking[place] == 0:
                        #remove_places.append(place)
                #for place in remove_places:
                    #del source_marking[place]
            # alte Plätze behalten, die nicht zu source gehören, aber markierung haben
            #for place in im:
                #if place not in source:
                    #updated_im[place] = im[place]

        # keine Anpassung notwendig
        #else:
            #updated_im = im

       # if len(sink) != 1:
            #new_sink = petri_utils.add_place(net, "global_sink")
            #for place in sink:
                #silent_transition = petri_utils.add_transition(
                    #net, f"t_sink_{place.name}"
                #)
                #petri_utils.add_arc_from_to(fr=silent_transition, to=new_sink, net=net)
                #petri_utils.add_arc_from_to(to=silent_transition, fr=place, net=net)

            # TODO wie mit markings != 1 umgehen?
            #updated_fm[new_sink] = 1

        #else:
            # TODO anders lösen / direkt über keys
            #for place in sink:
                #updated_fm[place] = 1
        # remove single transitions
        
        # check sink place / add if not existing
        # TODO kann aktuell auch entfernt werden, da sink beim einfügen in das IM-Netz entfernt wird
        stop = petri_utils.get_transition_by_name(net, "Stop")
        if len(stop.out_arcs) == 0:
            sink = petri_utils.add_place(net, "sink")
            fm[sink] = 1
            petri_utils.add_arc_from_to(fr=stop, to=sink, net=net)


        cleaned_net = petri_utils.remove_unconnected_components(net)
        # wenn transition entfernt wurde, kann jetzt ein leerer Platz übrig sein -> TODO kann er auch nur ausgehende kanten haben??
        #places = list(net.places)
        #for place in places:
            #if (
                #len(place.in_arcs) > 0
                #and len(place.out_arcs) == 0
                #and place not in fm
            #):
                #petri_utils.remove_place(cleaned_net, place)

        return cleaned_net, im, fm
    
    # wird angewandt, wenn holds true zurückgibt
    @classmethod
    # cls: die Klasse selbst (wenn es kein Objekt der Klasse gibt, ansonsten self), obj = event log, pool = multiprocessing pool (falls mehrere Prozesse parallel), manager = multiprocessing manager, optional zusätzliche Parameter??
    def apply(
        cls,
        obj: IMDataStructureUVCL,
        pool=None,
        manager=None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[
        ProcessTree
    ]:  

        # Optional: Traces filtern
        filtered_obj = SynthesisUVCL._filter_traces(obj, 0.1)
        filtered_obj = SynthesisUVCL._set_start_and_end(filtered_obj)

        # Token Trails erstellen
        (
            token_trails,
            initial_places,
            final_places,
            num_places,
            # initial_activities,
            # final_activities,
        ) = SynthesisUVCL._get_token_trails(filtered_obj, parameters)

        # Netz erstellen #TODO wird activities überhaupt benötigt?
        net, activities, im, fm = SynthesisUVCL._init_net(filtered_obj)

        net_places = SynthesisUVCL._solve_ilp_problem(
            token_trails, initial_places, final_places, num_places
        )

        net, im, fm = SynthesisUVCL._insert_net_places(net, im, fm, net_places)

        workflow_net, im, fm = SynthesisUVCL._create_wf_net(net, im, fm)

        vis.view_petri_net(workflow_net, im, fm, format="svg")

        #try:
           #pt = convert_to_process_tree(
            #workflow_net,
            #im,
            #fm,
        #) 
        #except:
        pt = PlaceholderTree(workflow_net, im, fm)

        return pt
    

class PlaceholderTree(ProcessTree):
    def __init__(self, petri_net: PetriNet, im: Marking, fm: Marking):
        tree_id = str(uuid.uuid4())
        super().__init__(label=f"synth_placeholder_{tree_id}")
        self.petri_net = petri_net
        self.im = im
        self.fm = fm
