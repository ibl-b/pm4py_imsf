from typing import Union, Optional, Dict, Any
import pandas as pd
from sqlite3 import Connection as SQ3_Connection
from pm4py.objects.ocel.obj import OCEL
from pm4py.util import pandas_utils, exec_utils
from pm4py.algo.querying.llm.injection.pm_knowledge.variants import traditional, ocel20


def apply(db: Union[pd.DataFrame, SQ3_Connection, OCEL], variant=None, parameters: Optional[Dict[Any, Any]] = None) -> str:
    """
    Provides a string containing the required process mining domain knowledge
    (in order for the LLM to produce meaningful queries).

    Parameters
    ---------------
    db
        Database
    parameters
        Optional parameters of the method

    Returns
    --------------
    pm_knowledge
        String containing the required process mining knowledge
    """
    if parameters is None:
        parameters = {}

    if variant is None:
        if pandas_utils.check_is_pandas_dataframe(db) or isinstance(db, SQ3_Connection):
            variant = traditional
        elif isinstance(db, OCEL):
            variant = ocel20

    if variant is None:
        return "\n\n"

    return exec_utils.get_variant(variant).apply(db, parameters)
