import pandas as pd
from typing import Optional, Dict, Any, Union
from sqlite3 import Connection as SQ3_Connection


def apply(db: Union[pd.DataFrame, SQ3_Connection], parameters: Optional[Dict[Any, Any]] = None) -> str:
    """
    Provides a string containing the required process mining domain knowledge for traditional process mining structures
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

    descr = """
If you need it, the process variant for a case can be obtained as concatenation of the activities of a case.
If you need it, the duration of a case can be obtained as difference between the timestamp of the first and the last event.
    """

    return descr
