"""
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
"""

from typing import TypeVar, Generic, Dict, Any, Optional

from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructureUVCL,
    IMDataStructureLog,
)
from pm4py.algo.discovery.inductive.fall_through.synthesis import SynthesisUVCL
from pm4py.algo.discovery.inductive.variants.abc import InductiveMinerFramework
from pm4py.algo.discovery.inductive.variants.instances import IMInstance
from pm4py.objects.process_tree.obj import ProcessTree


T = TypeVar("T", bound=IMDataStructureLog)


class IMSFS(Generic[T], InductiveMinerFramework[T]):

    def instance(self) -> IMInstance:
        return IMInstance.IMsfs


class IMSFSUVCL(IMSFS[IMDataStructureUVCL]):
    def apply(
        self,
        obj: IMDataStructureUVCL,
        parameters: Optional[Dict[str, Any]] = None,
        second_iteration: bool = False,
    ) -> ProcessTree:
        # TODO empty traces hier nötig? siehe andere varianten

        tree = self.apply_base_cases(obj, parameters)
        # if tree is None:
        # cut = self.find_cut(obj, parameters)
        # if cut is not None:
        # tree = self._recurse(cut[0], cut[1], parameters=parameters)
        if tree is None:
            ft = SynthesisUVCL.apply(obj, parameters)
            # recurse baut baum stück für stück auf --> muss vermutlich überschrieben werden, damit ganzer baum eingefügt werden kann
            # tree = self._recurse(ft[0], ft[1], parameters=parameters)
            tree = ft[0]

        return tree
