import pandas as pd
from typing import Optional, Dict, Any, Union
from pm4py.objects.ocel.obj import OCEL


def apply(ocel: OCEL, parameters: Optional[Dict[Any, Any]] = None) -> str:
    """
    Provides a string containing the required process mining domain knowledge for object-centric process mining structures
    (in order for the LLM to produce meaningful queries).

    Parameters
    ---------------
    ocel
        OCEL (2.0) object
    parameters
        Optional parameters of the method

    Returns
    --------------
    pm_knowledge
        String containing the required process mining knowledge
    """
    if parameters is None:
        parameters = {}

    descr = """
If you need to compute the duration of a lifecycle of an object, compute the difference between the timestamp of the last and the first event of the lifecycle.
If you need to compute the variant for an object, aggregate the names of the activities.
    """

    return descr
