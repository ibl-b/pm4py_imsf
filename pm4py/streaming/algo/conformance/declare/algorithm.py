from enum import Enum
from pm4py.util import exec_utils
from pm4py.streaming.algo.conformance.declare.variants import automata


class Variants(Enum):
    AUTOMATA = automata


def apply(declare_model, variant=Variants.AUTOMATA, parameters=None):
    """
    Streaming Conformance Checking Algorithm for DECLARE models.
    Attempts to implement state-based checks for all Declare constraint types.
    When a violation occurs, prints out which constraints are violated.

    Streaming algorithm interface implemented.

    Implementation of:
    Maggi, Fabrizio Maria, et al. "Monitoring business constraints with linear temporal logic: An approach based on colored automata." Business Process Management: 9th International Conference, BPM 2011, Clermont-Ferrand, France, August 30-September 2, 2011. Proceedings 9. Springer Berlin Heidelberg, 2011.
    -------

    """
    return exec_utils.get_variant(variant).apply(declare_model, parameters)
