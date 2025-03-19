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
        objects_df: Optional[pd.DataFrame] = None,
        parameters: Optional[Dict[Any, Any]] = None,
        chunk_size: int = 50000,  # Default chunk size
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

    # Parse timestamp column upfront in the original DataFrame
    df = dataframe_utils.convert_timestamp_columns_in_df(
        df,
        timest_format=pm4_constants.DEFAULT_TIMESTAMP_PARSE_FORMAT,
        timest_columns=[event_timestamp],
    )

    # Identify columns efficiently
    object_type_columns = [col for col in df.columns if col.startswith(object_type_prefix)]
    non_object_type_columns = [col for col in df.columns if not col.startswith(object_type_prefix)]

    # Pre-compute object type mappings
    object_type_mapping = {ot: ot.split(object_type_prefix)[1] for ot in object_type_columns}

    # Create events DataFrame (only non-object columns)
    events_df = df[non_object_type_columns].copy()

    # Add internal index for sorting events
    events_df = pandas_utils.insert_index(
        events_df, internal_index, copy_dataframe=False, reset_index=False
    )

    # Sort by timestamp and index
    events_df = events_df.sort_values([event_timestamp, internal_index])

    # Prepare data structures for relations
    ev_ids = []
    ev_activities = []
    ev_timestamps = []
    obj_ids = []
    obj_types = []

    # Track unique objects if needed
    unique_objects = {ot: list() for ot in object_type_columns} if objects_df is None else None

    # Initialize progress bar
    progress = _construct_progress_bar(len(df))

    # Create a filtered DataFrame with only needed columns
    needed_columns = [event_id, event_activity, event_timestamp] + object_type_columns
    filtered_df = df[needed_columns]

    # Process DataFrame in chunks to avoid memory issues
    # Use the chunk_size parameter from function arguments
    total_rows = len(filtered_df)

    for chunk_start in range(0, total_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_rows)

        # Extract a chunk
        chunk = filtered_df.iloc[chunk_start:chunk_end]

        # Convert small chunk to records for faster processing
        chunk_records = chunk.to_dict('records')

        # Process records in the current chunk
        for record in chunk_records:
            for ot in object_type_columns:
                obj_list = safe_parse_list(record[ot])
                if obj_list:
                    ot_striped = object_type_mapping[ot]

                    # Update unique objects if tracking
                    if unique_objects is not None:
                        unique_objects[ot].extend(obj_list)

                    # Extend relation data efficiently
                    n_objs = len(obj_list)
                    ev_ids.extend([record[event_id]] * n_objs)
                    ev_activities.extend([record[event_activity]] * n_objs)
                    ev_timestamps.extend([record[event_timestamp]] * n_objs)
                    obj_ids.extend(obj_list)
                    obj_types.extend([ot_striped] * n_objs)

            # Update progress (1 item at a time)
            if progress is not None:
                progress.update(1)

        for ot in unique_objects:
            unique_objects[ot] = list(set(unique_objects[ot]))

        # Free memory
        del chunk_records

    for ot in unique_objects:
        unique_objects[ot] = set(unique_objects[ot])

    # Clean up progress bar
    _destroy_progress_bar(progress)

    # Create relations DataFrame in one go
    if ev_ids:
        relations = pd.DataFrame({
            event_id: ev_ids,
            event_activity: ev_activities,
            event_timestamp: ev_timestamps,
            object_id_column: obj_ids,
            object_type_column: obj_types
        })

        # Add internal index for sorting
        relations = pandas_utils.insert_index(
            relations, internal_index, reset_index=False, copy_dataframe=False
        )

        # Sort by timestamp and index
        relations = relations.sort_values([event_timestamp, internal_index])

        # Remove temporary index column
        del relations[internal_index]
    else:
        # Create empty DataFrame with correct columns
        relations = pd.DataFrame(columns=[
            event_id, event_activity, event_timestamp, object_id_column, object_type_column
        ])

    # Remove temporary index column from events
    del events_df[internal_index]

    # Free memory
    del ev_ids, ev_activities, ev_timestamps, obj_ids, obj_types

    # Create objects DataFrame if not provided
    if objects_df is None:
        obj_types_list = []
        obj_ids_list = []

        for ot in object_type_columns:
            ot_striped = object_type_mapping[ot]
            obj_ids = list(unique_objects[ot])

            if obj_ids:
                obj_types_list.extend([ot_striped] * len(obj_ids))
                obj_ids_list.extend(obj_ids)

        objects_df = pd.DataFrame({
            object_type_column: obj_types_list,
            object_id_column: obj_ids_list
        })

        # Free memory
        del obj_types_list, obj_ids_list, unique_objects

    # Create and return OCEL object
    return OCEL(
        events=events_df,
        objects=objects_df,
        relations=relations,
        parameters=parameters,
    )