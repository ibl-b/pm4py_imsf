from typing import Any, Dict, List, Union, Optional, Collection, Tuple, Set, FrozenSet
from pm4py.utils import project_on_event_attribute
import pandas as pd
from pm4py.util import exec_utils
from enum import Enum
from pm4py.util import constants, xes_constants
from pm4py.objects.log.obj import EventLog
import importlib.util
from pm4py.objects.process_tree.utils import generic
from pm4py.objects.process_tree.obj import ProcessTree, Operator
import time
import sys


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    SHOW_PROGRESS_BAR = "show_progress_bar"
    MAX_TOTAL_TIME = "max_total_time"


def align_trace_with_process_tree(trace, tree):
    """
    Aligns a trace with a process tree using dynamic programming.
    Returns both the cost and the actual alignment.

    Version based on:
    A Dynamic Programming Approach for Alignments on Process Trees
    http://processquerying.com/wp-content/uploads/2024/09/PQMI_2024_279_A_Dynamic_Programming_Approach_for_Alignments_on_Process_Trees.pdf
    """
    # Convert trace to tuple for consistent hashing
    trace_tuple = tuple(trace)

    # Use dictionaries for memoization with optimized keys
    cost_table = {}
    alignment_table = {}

    # Pre-compute all labels for each subtree and cache them
    # Use a more efficient immutable data structure (frozenset) for lookup
    labels_cache = {}

    def compute_labels(subtree):
        """Compute and cache labels for each subtree"""
        if subtree in labels_cache:
            return labels_cache[subtree]

        if subtree.operator is None:  # Leaf node
            labels = frozenset([subtree.label]) if subtree.label is not None else frozenset()
        else:
            child_labels = [compute_labels(child) for child in subtree.children]
            labels = frozenset().union(*child_labels)

        labels_cache[subtree] = labels
        return labels

    # Pre-compute all labels at once
    compute_labels(tree)

    # Optimize key creation for memoization
    def make_key(w, T):
        """Create an efficient key for memoization"""
        return (w, id(T))

    # Create specialized functions for leaf nodes to avoid repeated code
    def handle_leaf_node(w, T):
        """Handle alignment for leaf nodes"""
        if T.label is None:  # Silent transition (tau)
            # All moves are deletions (move-on-log)
            cost = len(w)
            alignment = [(a, '>>') for a in w]
            return cost, alignment

        # Leaf with a label
        w_len = len(w)
        if w_len == 0:
            # Move-on-model (insertion)
            return 1, [('>>', T.label)]
        elif w_len == 1:
            if w[0] == T.label:
                # Synchronous move
                return 0, [(w[0], w[0])]
            else:
                # Move-on-log and move-on-model
                return 2, [(w[0], '>>'), ('>>', T.label)]
        else:
            # Try to find the label in the trace
            try:
                idx = w.index(T.label)
                # Deletions before match, match, deletions after match
                cost = idx + (w_len - idx - 1)
                alignment = [(w[i], '>>') for i in range(idx)] + \
                            [(T.label, T.label)] + \
                            [(w[i], '>>') for i in range(idx + 1, w_len)]
                return cost, alignment
            except ValueError:
                # Label not in trace - all deletions plus move-on-model
                return w_len + 1, [(a, '>>') for a in w] + [('>>', T.label)]

    # Main cost function with alignment tracking and optimizations
    def cost(w, T):
        """Calculate the minimum cost alignment between trace w and process tree T"""
        key = make_key(w, T)

        # Check memoization cache first
        if key in cost_table:
            return cost_table[key], alignment_table[key]

        # Handle different node types
        if T.operator is None:  # Leaf node
            result = handle_leaf_node(w, T)

        elif T.operator == Operator.SEQUENCE:
            # Optimize sequence operator with dynamic programming
            w_len = len(w)
            min_cost = float('inf')
            best_alignment = []

            # Compute all costs for the first child with different splits
            first_child_costs = []
            first_child_alignments = []

            for i in range(w_len + 1):
                c1, a1 = cost(w[:i], T.children[0])
                first_child_costs.append(c1)
                first_child_alignments.append(a1)

            # Compute all costs for the second child with different splits
            for i in range(w_len + 1):
                c1 = first_child_costs[i]
                a1 = first_child_alignments[i]

                # Skip if we already exceed the minimum cost
                if c1 >= min_cost:
                    continue

                c2, a2 = cost(w[i:], T.children[1])
                total_cost = c1 + c2

                if total_cost < min_cost:
                    min_cost = total_cost
                    best_alignment = a1 + a2

            result = min_cost, best_alignment

        elif T.operator == Operator.XOR:
            # Minimum cost between alternatives
            min_cost = float('inf')
            best_alignment = []

            for child in T.children:
                c, a = cost(w, child)
                if c < min_cost:
                    min_cost = c
                    best_alignment = a

            result = min_cost, best_alignment

        elif T.operator == Operator.PARALLEL:
            # Optimize parallel operator by using pre-computed labels
            labels_T1 = labels_cache[T.children[0]]
            labels_T2 = labels_cache[T.children[1]]

            # Partition the trace based on labels
            w1 = tuple(a for a in w if a in labels_T1)
            w2 = tuple(a for a in w if a in labels_T2)
            w_rest = tuple(a for a in w if a not in labels_T1 and a not in labels_T2)

            # Handle matched activities
            c1, a1 = cost(w1, T.children[0])
            c2, a2 = cost(w2, T.children[1])

            # Calculate total cost
            total_cost = c1 + c2 + len(w_rest)

            # Reconstruct alignment efficiently
            alignment = []
            idx_map = {}

            # Create index maps for w1 and w2
            w1_idx = 0
            w2_idx = 0
            for i, a in enumerate(w):
                if a in labels_T1:
                    idx_map[(0, w1_idx)] = i
                    w1_idx += 1
                elif a in labels_T2:
                    idx_map[(1, w2_idx)] = i
                    w2_idx += 1

            # Initialize alignment with move-on-log for all positions
            alignment = [(a, '>>') for a in w]

            # Update alignment for w1
            for i, move in enumerate(a1):
                orig_idx = idx_map.get((0, i))
                if orig_idx is not None:
                    alignment[orig_idx] = move

            # Update alignment for w2
            for i, move in enumerate(a2):
                orig_idx = idx_map.get((1, i))
                if orig_idx is not None:
                    alignment[orig_idx] = move

            result = total_cost, alignment

        elif T.operator == Operator.LOOP:
            w_len = len(w)

            # Initialize DP tables
            distances = [float('inf')] * (w_len + 1)
            predecessors = [None] * (w_len + 1)
            distances[0] = 0

            # Pre-compute costs for common subtrees to avoid redundant calculations
            t1_costs = {}
            t1_alignments = {}

            # Optimize initial computation
            for pos in range(w_len + 1):
                if pos > 0 and distances[pos - 1] == float('inf'):
                    # Skip if previous position is unreachable
                    continue

                # Pre-compute T1 costs
                w_slice = w[:pos]
                if w_slice not in t1_costs:
                    c, a = cost(w_slice, T.children[0])
                    t1_costs[w_slice] = c
                    t1_alignments[w_slice] = a
                else:
                    c, a = t1_costs[w_slice], t1_alignments[w_slice]

                if distances[pos] > c:
                    distances[pos] = c
                    predecessors[pos] = ('T1', 0, pos, a)

            # More efficient loop expansion
            for i in range(w_len + 1):
                if distances[i] >= float('inf'):
                    continue

                # Try to exit the loop
                w_slice = w[i:]
                if w_slice not in t1_costs:
                    c_t1_exit, a_t1_exit = cost(w_slice, T.children[0])
                    t1_costs[w_slice] = c_t1_exit
                    t1_alignments[w_slice] = a_t1_exit
                else:
                    c_t1_exit, a_t1_exit = t1_costs[w_slice], t1_alignments[w_slice]

                total_cost_exit = distances[i] + c_t1_exit
                if distances[w_len] > total_cost_exit:
                    distances[w_len] = total_cost_exit
                    predecessors[w_len] = ('T1_exit', i, w_len, a_t1_exit)

                # Try to execute the loop body (T2 followed by T1)
                for j in range(i + 1, w_len + 1):
                    # Skip if we're going to exceed the current best
                    if distances[i] >= distances[w_len]:
                        break

                    w_slice_t2 = w[i:j]
                    c_t2, a_t2 = cost(w_slice_t2, T.children[1])

                    # Early termination if this partial path is already worse
                    total_cost_t2 = distances[i] + c_t2
                    if total_cost_t2 >= distances[w_len]:
                        continue

                    # After T2, we must match T1 again
                    for k in range(j, w_len + 1):
                        w_slice_t1 = w[j:k]
                        if w_slice_t1 not in t1_costs:
                            c_t1_again, a_t1_again = cost(w_slice_t1, T.children[0])
                            t1_costs[w_slice_t1] = c_t1_again
                            t1_alignments[w_slice_t1] = a_t1_again
                        else:
                            c_t1_again, a_t1_again = t1_costs[w_slice_t1], t1_alignments[w_slice_t1]

                        total_cost_loop = total_cost_t2 + c_t1_again
                        if distances[k] > total_cost_loop:
                            distances[k] = total_cost_loop
                            predecessors[k] = ('loop', i, j, k, a_t2, a_t1_again)

            # Reconstruct alignment
            final_cost = distances[w_len]
            alignment = []
            position = w_len

            # Build alignment from predecessors
            while position > 0:
                pred = predecessors[position]
                if pred is None:
                    break

                if pred[0] == 'T1_exit':
                    _, i, _, a_t1_exit = pred
                    alignment = a_t1_exit + alignment
                    position = i
                elif pred[0] == 'T1':
                    _, i, _, a_t1 = pred
                    alignment = a_t1 + alignment
                    position = i
                elif pred[0] == 'loop':
                    _, i, j, k, a_t2, a_t1_again = pred
                    alignment = a_t1_again + alignment
                    alignment = a_t2 + alignment
                    position = i

            result = final_cost, alignment

        else:
            raise NotImplementedError(f"Operator {T.operator} not implemented.")

        # Store result in memoization tables
        cost_table[key] = result[0]
        alignment_table[key] = result[1]
        return result

    # Run the algorithm
    return cost(trace_tuple, tree)


def _construct_progress_bar(progress_length, parameters):
    """Create progress bar if enabled"""
    if exec_utils.get_param_value(Parameters.SHOW_PROGRESS_BAR, parameters,
                                  constants.SHOW_PROGRESS_BAR) and importlib.util.find_spec("tqdm"):
        if progress_length > 1:
            from tqdm.auto import tqdm
            return tqdm(total=progress_length, desc="aligning log, completed variants :: ")
    return None


def _destroy_progress_bar(progress):
    """Clean up progress bar"""
    if progress is not None:
        progress.close()
    del progress


def apply_list_tuple_activities(list_tuple_activities: List[Collection[str]], process_tree: ProcessTree,
                                parameters: Optional[Dict[Any, Any]] = None) -> List[Dict[str, Any]]:
    """Apply alignment to a list of trace activity tuples"""
    if parameters is None:
        parameters = {}

    max_total_time = exec_utils.get_param_value(Parameters.MAX_TOTAL_TIME, parameters, sys.maxsize)

    # Convert process tree to binary tree for alignment
    process_tree = generic.process_tree_to_binary_process_tree(process_tree)

    # Optimize by processing unique variants once
    variants = set(tuple(v) for v in list_tuple_activities)
    variants_align = {}

    progress = _construct_progress_bar(len(variants), parameters)

    # Calculate empty trace alignment
    empty_cost, empty_moves = align_trace_with_process_tree([], process_tree)
    empty_cost = round(empty_cost + 10 ** -14, 13)

    # Track time for early termination
    t0 = time.time_ns()

    # Process each variant once
    for v in variants:
        alignment_cost, alignment_moves = align_trace_with_process_tree(v, process_tree)
        alignment_cost = round(alignment_cost + 10 ** -14, 13)

        # Calculate fitness
        trace_len = len(v)
        denominator = empty_cost + trace_len
        fitness = 1.0 - alignment_cost / denominator if denominator > 0 else 0.0

        variants_align[v] = {
            "cost": alignment_cost,
            "alignment": alignment_moves,
            "fitness": fitness
        }

        if progress is not None:
            progress.update()

        # Check if we've exceeded the maximum time
        t1 = time.time_ns()
        if (t1 - t0) / 10 ** 9 > max_total_time:
            _destroy_progress_bar(progress)
            return None

    _destroy_progress_bar(progress)

    # Map results back to original traces
    return [variants_align[tuple(t)] for t in list_tuple_activities]


def apply(log: Union[pd.DataFrame, EventLog], process_tree: ProcessTree, parameters: Optional[Dict[Any, Any]] = None) -> \
        List[Dict[str, Any]]:
    """
    Optimized version that aligns an event log against a process tree model.
    Based on the approach described in:
    Schwanen, Christopher T., Wied Pakusa, and Wil MP van der Aalst. "Process tree alignments."
    Enterprise Design, Operations, and Computing, ser. LNCS, Cham: Springer International Publishing (2024).

    Parameters
    ---------------
    log
        Event log or Pandas dataframe
    process_tree
        Process tree
    parameters
        Parameters of the algorithm, including:
        - Parameters.ACTIVITY_KEY => the attribute to be used as activity
        - Parameters.SHOW_PROGRESS_BAR => shows the progress bar
        - Parameters.MAX_TOTAL_TIME => maximum total time in seconds

    Returns
    ---------------
    aligned_traces
        List that contains the alignment for each trace
    """
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY)

    # Project log on activities and convert to tuples
    list_tuple_activities = project_on_event_attribute(log, activity_key)

    # Apply alignment
    return apply_list_tuple_activities(list_tuple_activities, process_tree, parameters=parameters)
