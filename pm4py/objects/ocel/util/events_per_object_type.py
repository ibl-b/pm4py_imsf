'''
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
'''
from enum import Enum
from pm4py.util import exec_utils, pandas_utils
from pm4py.objects.ocel.obj import OCEL
from pm4py.objects.ocel import constants as ocel_constants
from typing import Optional, Dict, Any


class Parameters(Enum):
    EVENT_ID = ocel_constants.PARAM_EVENT_ID
    OBJECT_TYPE = ocel_constants.PARAM_OBJECT_TYPE


def apply(ocel: OCEL, parameters: Optional[Dict[Any, Any]] = None):
    """
    Returns for each object type the number of events related to at least one object of the given type.

    Parameters
    ----------------
    ocel
        Object-centric event log
    parameters
        Parameters of the algorithm, including:
        - Parameters.EVENT_ID => the event identifier
        - Parameters.OBJECT_TYPE => the object type column

    Returns
    -----------------
    dictio
        Dictionary associating to each object type the number of events related to at least one object of the given
        type.
    """
    if parameters is None:
        parameters = {}

    event_id = exec_utils.get_param_value(Parameters.EVENT_ID, parameters, ocel.event_id_column)
    object_type = exec_utils.get_param_value(Parameters.OBJECT_TYPE, parameters, ocel.object_type_column)

    object_types = pandas_utils.format_unique(ocel.objects[object_type].unique())

    ret = {}

    for ot in object_types:
        ret[ot] = ocel.relations[ocel.relations[object_type] == ot][event_id].nunique()

    return ret
