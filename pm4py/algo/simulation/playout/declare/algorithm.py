from pm4py.objects.log.obj import EventLog
from enum import Enum
from pm4py.util import exec_utils
from pm4py.algo.simulation.playout.declare.variants import classic


class Variants(Enum):
    CLASSIC = classic


def apply(declare_model, variant=Variants.CLASSIC, parameters=None) -> EventLog:
    """
    Produce a playout EventLog of a given DECLARE model.
    Optional parameters:
      - n_traces (int): number of traces to generate. Default = 1000
      - min_length (int): minimal length of each trace. Default = 3
      - max_length (int): maximal length of each trace. Default = 15
    """
    return exec_utils.get_variant(variant).apply(declare_model, parameters)
