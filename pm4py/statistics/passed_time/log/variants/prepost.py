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
from pm4py.algo.discovery.dfg.variants import native, performance
from typing import Optional, Dict, Any
from pm4py.objects.log.obj import EventLog
from pm4py.objects.conversion.log import converter as log_converter


def apply(log: EventLog, activity: str, parameters: Optional[Dict[Any, Any]] = None) -> Dict[str, Any]:
    """
    Gets the time passed from each preceding activity and to each succeeding activity

    Parameters
    -------------
    log
        Log
    activity
        Activity that we are considering
    parameters
        Possible parameters of the algorithm

    Returns
    -------------
    dictio
        Dictionary containing a 'pre' key with the
        list of aggregated times from each preceding activity to the given activity
        and a 'post' key with the list of aggregates times from the given activity
        to each succeeding activity
    """
    if parameters is None:
        parameters = {}

    log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)

    dfg_frequency = native.native(log, parameters=parameters)
    dfg_performance = performance.performance(log, parameters=parameters)

    pre = []
    sum_perf_post = 0.0
    sum_acti_post = 0.0
    post = []
    sum_perf_pre = 0.0
    sum_acti_pre = 0.0

    for entry in dfg_performance.keys():
        if entry[1] == activity:
            pre.append([entry[0], float(dfg_performance[entry]), int(dfg_frequency[entry])])
            sum_perf_pre = sum_perf_pre + float(dfg_performance[entry]) * float(dfg_frequency[entry])
            sum_acti_pre = sum_acti_pre + float(dfg_frequency[entry])
        if entry[0] == activity:
            post.append([entry[1], float(dfg_performance[entry]), int(dfg_frequency[entry])])
            sum_perf_post = sum_perf_post + float(dfg_performance[entry]) * float(dfg_frequency[entry])
            sum_acti_post = sum_acti_post + float(dfg_frequency[entry])

    perf_acti_pre = 0.0
    if sum_acti_pre > 0:
        perf_acti_pre = sum_perf_pre / sum_acti_pre
    perf_acti_post = 0.0
    if sum_acti_post > 0:
        perf_acti_post = sum_perf_post / sum_acti_post

    return {"pre": pre, "post": post, "post_avg_perf": perf_acti_post, "pre_avg_perf": perf_acti_pre}
