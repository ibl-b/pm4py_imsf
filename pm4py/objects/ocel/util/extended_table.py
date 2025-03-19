import ast
from enum import Enum
from typing import Optional, Dict, Any, List

import pandas as pd
import importlib.util

from pm4py.objects.ocel import constants
from pm4py.objects.ocel.obj import OCEL
from pm4py.util import exec_utils, pandas_utils, constants as pm4_constants
from pm4py.objects.log.util import dataframe_utils


class Parameters(Enum):
    OBJECT_TYPE_PREFIX = constants.PARAM_OBJECT_TYPE_PREFIX_EXTENDED
    EVENT_ID = constants.PARAM_EVENT_ID
    EVENT_ACTIVITY = constants.PARAM_EVENT_ACTIVITY
    EVENT_TIMESTAMP = constants.PARAM_EVENT_TIMESTAMP
    OBJECT_ID = constants.PARAM_OBJECT_ID
    OBJECT_TYPE = constants.PARAM_OBJECT_TYPE
    INTERNAL_INDEX = constants.PARAM_INTERNAL_INDEX


def _construct_progress_bar(progress_length):
    if importlib.util.find_spec("tqdm"):
        if progress_length > 1:
            from tqdm.auto import tqdm

            return tqdm(
                total=progress_length,
                desc="importing OCEL, parsed rows :: ",
            )
    return None


def _destroy_progress_bar(progress):
    if progress is not None:
        progress.close()
    del progress


def safe_parse_list(value):
    """
    Safely parse a string into a list using ast.literal_eval.

    Args:
        value: The value to parse

    Returns:
        A list if the parsing was successful, otherwise an empty list
    """
    if isinstance(value, str) and value.startswith('['):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    return []


def get_ocel_from_extended_table(
        df: pd.DataFrame,
        objects_df: Optional[Dict[Any, Any]] = None,
        parameters: Optional[Dict[Any, Any]] = None,
) -> OCEL:
    """
    Get an OCEL object from an extended table format.

    Args:
        df: The DataFrame in extended table format
        objects_df: Optional DataFrame of objects
        parameters: Optional parameters dictionary

    Returns:
        An OCEL object
    """
    if parameters is None:
        parameters = {}

    # Extract parameters
    object_type_prefix = exec_utils.get_param_value(
        Parameters.OBJECT_TYPE_PREFIX,
        parameters,
        constants.DEFAULT_OBJECT_TYPE_PREFIX_EXTENDED,
    )
    event_activity = exec_utils.get_param_value(
        Parameters.EVENT_ACTIVITY, parameters, constants.DEFAULT_EVENT_ACTIVITY
    )
    event_id = exec_utils.get_param_value(
        Parameters.EVENT_ID, parameters, constants.DEFAULT_EVENT_ID
    )
    event_timestamp = exec_utils.get_param_value(
        Parameters.EVENT_TIMESTAMP,
        parameters,
        constants.DEFAULT_EVENT_TIMESTAMP,
    )
    object_id_column = exec_utils.get_param_value(
        Parameters.OBJECT_ID, parameters, constants.DEFAULT_OBJECT_ID
    )
    object_type_column = exec_utils.get_param_value(
        Parameters.OBJECT_TYPE, parameters, constants.DEFAULT_OBJECT_TYPE
    )
    internal_index = exec_utils.get_param_value(
        Parameters.INTERNAL_INDEX, parameters, constants.DEFAULT_INTERNAL_INDEX
    )

    # Identify columns efficiently
    object_type_columns = [col for col in df.columns if col.startswith(object_type_prefix)]
    non_object_type_columns = [col for col in df.columns if not col.startswith(object_type_prefix)]

    # Pre-compute object type mappings (do this once outside the loop)
    object_type_mapping = {ot: ot.split(object_type_prefix)[1] for ot in object_type_columns}

    # Use lists for relation data (more memory efficient than constantly growing dicts)
    ev_ids = []
    ev_activities = []
    ev_timestamps = []
    obj_ids = []
    obj_types = []

    # Prepare sets for tracking unique objects
    objects_set = {ot: set() for ot in object_type_columns}

    # Select only necessary columns for processing
    required_columns = [event_id, event_activity, event_timestamp] + object_type_columns

    progress = _construct_progress_bar(len(df))

    # Process rows efficiently
    for _, row in df[required_columns].iterrows():
        ev_id = row[event_id]
        ev_activity = row[event_activity]
        ev_timestamp = row[event_timestamp]

        for ot in object_type_columns:
            # Parse the list of objects safely
            obj_list = safe_parse_list(row[ot])
            ot_striped = object_type_mapping[ot]

            # Update the set of unique objects
            objects_set[ot].update(obj_list)

            # Add relation data
            for obj in obj_list:
                ev_ids.append(ev_id)
                ev_activities.append(ev_activity)
                ev_timestamps.append(ev_timestamp)
                obj_ids.append(obj)
                obj_types.append(ot_striped)

        if progress is not None:
            progress.update()

    _destroy_progress_bar(progress)

    # Create relations DataFrame in one go (more efficient than incremental building)
    relations = pd.DataFrame({
        event_id: ev_ids,
        event_activity: ev_activities,
        event_timestamp: ev_timestamps,
        object_id_column: obj_ids,
        object_type_column: obj_types
    })

    # Free memory
    del ev_ids, ev_activities, ev_timestamps, obj_ids, obj_types

    # Create objects DataFrame if not provided
    if objects_df is None:
        obj_type_list = []
        obj_id_list = []

        for ot in object_type_columns:
            ot_striped = object_type_mapping[ot]
            ot_objects = list(objects_set[ot])

            # More efficient than multiple appends
            obj_type_list.extend([ot_striped] * len(ot_objects))
            obj_id_list.extend(ot_objects)

        objects_df = pd.DataFrame({
            object_type_column: obj_type_list,
            object_id_column: obj_id_list
        })

        # Free memory
        del obj_type_list, obj_id_list

    # Free memory
    del objects_set

    # Process the events DataFrame (only non-object columns)
    events_df = df[non_object_type_columns].copy()

    # Convert timestamp columns
    events_df = dataframe_utils.convert_timestamp_columns_in_df(
        events_df,
        timest_format=pm4_constants.DEFAULT_TIMESTAMP_PARSE_FORMAT,
        timest_columns=[event_timestamp],
    )

    relations = dataframe_utils.convert_timestamp_columns_in_df(
        relations,
        timest_format=pm4_constants.DEFAULT_TIMESTAMP_PARSE_FORMAT,
        timest_columns=[event_timestamp],
    )

    # Add internal index for sorting
    events_df = pandas_utils.insert_index(
        events_df, internal_index, copy_dataframe=False, reset_index=False
    )
    relations = pandas_utils.insert_index(
        relations, internal_index, reset_index=False, copy_dataframe=False
    )

    # Sort by timestamp and index
    events_df = events_df.sort_values([event_timestamp, internal_index])
    relations = relations.sort_values([event_timestamp, internal_index])

    # Remove temporary index columns
    del events_df[internal_index]
    del relations[internal_index]

    # Create and return OCEL object
    return OCEL(
        events=events_df,
        objects=objects_df,
        relations=relations,
        parameters=parameters,
    )
