import unittest
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.inductive.variants import imsf
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import vis
from pm4py.objects.conversion.process_tree import converter as process_tree_converter
from pm4py.analysis import check_is_workflow_net
from pm4py.analysis import check_soundness
from pm4py.util.compression import util as comut

import time
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.objects.petri_net.utils import petri_utils


class IMSFSTest(unittest.TestCase):
    
    log = "bpi2012_olog.xes"
    

    def test_compare_imf(self, log_name=log):
        # to avoid static method warnings in tests,
        # that by construction of the unittest package have to be expressed in such way
        self.dummy_variable = "dummy_value"
        #log = xes_importer.apply(log_name)
        #start = time.time()
        #net, initial_marking, final_marking = pm4py.discovery.discover_petri_net_inductive(log, variant="im")
        #end = time.time()
        #print(f"Laufzeit IM: {end - start:.3f} Sekunden")
        #vis.view_petri_net(net, initial_marking, final_marking, format="svg")
        #pm4py.write_pnml(net, initial_marking, final_marking, "bpi2020DD_IM.pnml")


    def test_imsfs(self, log_name=log):
        # to avoid static method warnings in tests,
        # that by construction of the unittest package have to be expressed in such way
        self.dummy_variable = "dummy_value"
        log = xes_importer.apply(log_name)
        start = time.time()
        net, initial_marking, final_marking = pm4py.discovery.discover_petri_net_inductive(log, disable_fallthroughs=True, variant="IMSF")
        end = time.time()
        print(f"Laufzeit IMSFS: {end - start:.3f} Sekunden")

        vis.view_petri_net(net, initial_marking, final_marking, format="svg")
        pm4py.write_pnml(net, initial_marking, final_marking, "bpi2020DD_IMSFS.pnml")
        

    def test_net(self, log_name=log):
        self.dummy_variable = "dummy_value"
        file = "IMSF_Nets/roadtraffic_IMSFS_oSTFT_komplett.pnml"
        log = xes_importer.apply(log_name)
        
        net, im, fm = pm4py.read_pnml(file)
        vis.view_petri_net(net, im, fm, format="svg")

        results = pm4py.conformance.fitness_token_based_replay(log, net, im, fm)
        #results = pm4py.conformance.conformance_diagnostics_token_based_replay(log, net, im, fm)
        results_p = pm4py.conformance.precision_token_based_replay(log, net, im, fm)
        results_gen = pm4py.conformance.generalization_tbr(log, net, im, fm)
        alignment_result = pm4py.conformance.fitness_alignments(log, net, im, fm)
        simp_cyc = simplicity_evaluator.apply(net, variant=simplicity_evaluator.Variants.EXTENDED_CYCLOMATIC)
        simp_arc = simplicity_evaluator.apply(net, variant=simplicity_evaluator.Variants.SIMPLICITY_ARC_DEGREE)
        simp_ec = simplicity_evaluator.apply(net, variant=simplicity_evaluator.Variants.EXTENDED_CARDOSO)
        print(f"fit tb im: {results}")
        print(f"prec tb im: {results_p}")
        print(f"fit alig: {alignment_result}")
        print(f"Gen: {results_gen}")
        print(f"simp cyc: {simp_cyc}")
        print(f"simp arc: {simp_arc}")
        print(f"simp ec: {simp_ec}")
        
       

if __name__ == "__main__":
    unittest.main()
