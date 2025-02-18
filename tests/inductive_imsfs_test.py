import unittest
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.inductive.variants import imsfs
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import vis
from pm4py.objects.conversion.process_tree import converter as process_tree_converter


class IMSFSTest(unittest.TestCase):
    log = "LisaB_Testlogs/Road_Traffic_Fine.xes"
    pnml_net = "LisaB_Testlogs/Roadtraffic_10.pnml"

    def test_compare_imf(self, log_name=log):
        # to avoid static method warnings in tests,
        # that by construction of the unittest package have to be expressed in such way
        self.dummy_variable = "dummy_value"
        #log = xes_importer.apply(log_name)
        #variant = inductive_miner.Variants.IM
        #process_tree = inductive_miner.apply(log, variant=variant)
        #net, initial_marking, final_marking = process_tree_converter.apply(process_tree)
        #vis.view_petri_net(net, initial_marking, final_marking, format="svg")

    def test_compare_pnml(self, net_name=pnml_net):
        # to avoid static method warnings in tests,
        # that by construction of the unittest package have to be expressed in such way
        self.dummy_variable = "dummy_value"
        #net, initial_marking, final_marking = pm4py.read.read_pnml(net_name)
        #pm4py.view_petri_net(net, initial_marking, final_marking, format="svg")

    def test_imsfs(self, log_name=log):
        # to avoid static method warnings in tests,
        # that by construction of the unittest package have to be expressed in such way
        self.dummy_variable = "dummy_value"
        log = xes_importer.apply(log_name)
        variant = inductive_miner.Variants.IMsfs

        process_tree = inductive_miner.apply(
            log,
            variant=variant,
        )
        #net, initial_marking, final_marking = process_tree_converter.apply(process_tree)
        
        net, initial_marking, final_marking = imsfs.IMSFSUVCL.convert_process_tree(process_tree)
        vis.view_petri_net(net, initial_marking, final_marking, format="svg")


if __name__ == "__main__":
    unittest.main()
