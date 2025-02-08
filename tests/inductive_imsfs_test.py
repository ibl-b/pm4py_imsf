import unittest

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import vis
from pm4py.objects.conversion.process_tree import converter as process_tree_converter

import importlib
from pm4py.algo.discovery.inductive.fall_through import synthesis


class IMSFSTest(unittest.TestCase):
    def test_imsfs(self, log_name="LisaB_Testlogs/BPI2012_olog.xes"):
        # to avoid static method warnings in tests,
        # that by construction of the unittest package have to be expressed in such way
        self.dummy_variable = "dummy_value"
        log = xes_importer.apply(log_name)
        variant = inductive_miner.Variants.IMsfs
        # inductive_miner.apply(log, variant=variant)

        process_tree = inductive_miner.apply(
            log,
            variant=variant,
        )
        net, initial_marking, final_marking = process_tree_converter.apply(process_tree)
        vis.view_petri_net(net, initial_marking, final_marking, format="svg")


if __name__ == "__main__":
    unittest.main()
