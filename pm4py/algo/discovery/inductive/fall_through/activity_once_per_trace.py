from collections import Counter
from typing import Any, Optional, Dict
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.algo.discovery.inductive.fall_through.activity_concurrent import ActivityConcurrentUVCL
from pm4py.util.compression import util as comut


class ActivityOncePerTraceUVCL(ActivityConcurrentUVCL):
    @classmethod
    def _get_candidate(
        cls,
        obj: IMDataStructureUVCL,
        pool=None,
        manager=None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        # Initialize candidates as a set of all activities
        candidates = set(comut.get_alphabet(obj.data_structure))

        for t in obj.data_structure:
            # Use a Counter to count occurrences of each activity in the trace
            activity_counts = Counter(t)
            # Create a set of activities that occur exactly once in the trace
            activities_once = {activity for activity, count in activity_counts.items() if count == 1}
            # Intersect with the existing candidates
            candidates &= activities_once
            # Early exit if no candidates remain
            if not candidates:
                return None

        # Return any one of the remaining candidates
        return next(iter(candidates)) if candidates else None
