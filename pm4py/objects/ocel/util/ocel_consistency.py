from pm4py.objects.ocel.obj import OCEL
from typing import Optional, Dict, Any
import warnings
import pandas as pd


def apply(ocel: OCEL, parameters: Optional[Dict[Any, Any]] = None) -> OCEL:
    """
    Forces the consistency of the OCEL, ensuring that the event/object identifier,
    event/object type are of type string and non-empty.

    Parameters
    --------------
    ocel
        OCEL
    parameters
        Possible parameters of the method

    Returns
    --------------
    ocel
        Consistent OCEL
    """
    if parameters is None:
        parameters = {}

    fields = {
        "events": [ocel.event_id_column, ocel.event_activity],
        "objects": [ocel.object_id_column, ocel.object_type_column],
        "relations": [
            ocel.event_id_column,
            ocel.object_id_column,
            ocel.event_activity,
            ocel.object_type_column,
        ],
        "o2o": [ocel.object_id_column, ocel.object_id_column + "_2"],
        "e2e": [ocel.event_id_column, ocel.event_id_column + "_2"],
        "object_changes": [ocel.object_id_column],
    }

    # Process each dataframe only once instead of per field
    for tab, columns in fields.items():
        df = getattr(ocel, tab)

        # Process all columns in a single operation
        # 1. Drop rows with any NA values in specified columns
        df = df.dropna(subset=columns, how="any")

        # 2. Convert columns to string type in a single operation
        df[columns] = df[columns].astype("string")

        # 3. Drop rows with empty strings
        # Create a mask for rows where all specified columns have non-empty strings
        mask = pd.Series(True, index=df.index)
        for col in columns:
            mask = mask & (df[col].str.len() > 0)
        df = df[mask]

        setattr(ocel, tab, df)

    # Check uniqueness - only compute nunique once per dataframe
    events_df = ocel.events
    objects_df = ocel.objects

    num_ev_ids = events_df[ocel.event_id_column].nunique()
    num_obj_ids = objects_df[ocel.object_id_column].nunique()

    if num_ev_ids < len(events_df):
        warnings.warn("The event identifiers in the OCEL are not unique!")

    if num_obj_ids < len(objects_df):
        warnings.warn("The object identifiers in the OCEL are not unique!")

    return ocel
