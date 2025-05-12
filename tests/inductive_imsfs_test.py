import unittest
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.inductive.variants import imsfs
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import vis
from pm4py.objects.conversion.process_tree import converter as process_tree_converter
from pm4py.analysis import check_is_workflow_net
from pm4py.analysis import check_soundness

import time
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.objects.petri_net.utils import petri_utils


class IMSFSTest(unittest.TestCase):
    #log = "LisaB_Testlogs/25-04-10/roadtraffic_FT_oSTFT_0.001.xes"
    log = "LisaB_Testlogs/BPIC15_2.xes"
    

    def test_compare_imf(self, log_name=log):
        # to avoid static method warnings in tests,
        # that by construction of the unittest package have to be expressed in such way
        self.dummy_variable = "dummy_value"
        log = xes_importer.apply(log_name)
        #start = time.time()
        net, initial_marking, final_marking = pm4py.discovery.discover_petri_net_inductive(log, variant="im")
        #end = time.time()
        #print(f"Laufzeit IM: {end - start:.3f} Sekunden")
        #vis.view_petri_net(net, initial_marking, final_marking, format="svg")
        #pm4py.write_pnml(net, initial_marking, final_marking, "bpi2020DD_IM.pnml")


    def test_imsfs(self, log_name=log):
        # to avoid static method warnings in tests,
        # that by construction of the unittest package have to be expressed in such way
        self.dummy_variable = "dummy_value"
        #log = xes_importer.apply(log_name)
        #start = time.time()
        #net, initial_marking, final_marking = pm4py.discovery.discover_petri_net_inductive(log, disable_fallthroughs=False, variant="IMSFS")
        #end = time.time()
        #print(f"Laufzeit IMSFS: {end - start:.3f} Sekunden")

        #vis.view_petri_net(net, initial_marking, final_marking, format="svg")
        #pm4py.write_pnml(net, initial_marking, final_marking, "bpi2020DD_IMSFS.pnml")
        

    def test_net(self, log_name=log):
        self.dummy_variable = "dummy_value"
        #log_name_im = "LisaB_Testlogs/25-04-10/roadtraffic_oSTFT_FT.xes"
        #log_name_imsfs = "LisaB_Testlogs/25-04-10/roadtraffic_IMSFS_synth_0.001.xes"
        #file = "LisaB_Testlogs/25-04-10/roadtraffic_IM_FT_oSTFT_0.001.pnml"
        #file_imsfs = "LisaB_Testlogs/25-04-10/BPI2020DD_IMSFS_komplett_0.001.pnml"
        #file_imsfs_f = "LisaB_Testlogs/25-03-31_Netze/roadtraffic_synth_komplett_0.001f.pnml"
        #log_im = xes_importer.apply(log_name)
        #log_imsfs = xes_importer.apply(log_name_im)
        #log = xes_importer.apply(log_name)
        
        #net, im, fm = pm4py.read_pnml(file)
        #inet, iim, ifm = pm4py.read_pnml(file_imsfs)
        #vis.view_petri_net(inet, iim, ifm, format="svg")
        #vis.view_petri_net(net, im, fm, format="svg")

        #results = pm4py.conformance.fitness_token_based_replay(log_im, net, im, fm)
        #results = pm4py.conformance.conformance_diagnostics_token_based_replay(log_im, net, iim, ifm)
        #results_p = pm4py.conformance.precision_token_based_replay(log_im, net, im, fm)
        #results_gen = pm4py.conformance.generalization_tbr(log_im, net, im, fm)
        #alignment_result = pm4py.conformance.fitness_alignments(log_im, net, im, fm)
        #alignment_result_p = pm4py.conformance.precision_alignments(log_im, net, im, fm)
        #results_cyc = simplicity_evaluator.apply(net, variant=simplicity_evaluator.Variants.EXTENDED_CYCLOMATIC)
        #alignment_arc = simplicity_evaluator.apply(net, variant=simplicity_evaluator.Variants.SIMPLICITY_ARC_DEGREE)
        #alignment_ec = simplicity_evaluator.apply(net, variant=simplicity_evaluator.Variants.EXTENDED_CARDOSO)
        #print(f"fit tb im: {results}")
        #print(f"prec tb im: {results_p}")
        #print(f"Gen: {results_gen}")
        #print(f"prec_alig: {alignment_result_p}")
        #print(f"simp cyc: {results_cyc}")
        #print(f"simp arc: {alignment_arc}")
        
        #print(f"simp ec: {alignment_ec}")
        #imsfsnet, iim, ifm = pm4py.read_pnml(file_imsfs)
        #result = pm4py.conformance.conformance_diagnostics_token_based_replay(log, imsfsnet, iim, ifm)
        #imsfsnet_f, iimf, ifmf = pm4py.read_pnml(file_imsfs_f)
        #imsfsresults = pm4py.conformance.precision_token_based_replay(log_im, inet, iim, ifm)
        #imsfsresults_f = pm4py.conformance.precision_token_based_replay(log, imsfsnet_f, iimf, ifmf)
        #imsfsresults_f = pm4py.conformance.fitness_token_based_replay(log_im, inet, iim, ifm)
        #ialignment_result = pm4py.conformance.fitness_alignments(log_im, inet, iim, ifm)
        #ialignment_result_f = pm4py.conformance.precision_alignments(log_im, inet, iim, ifm)
        #ialignment_result_f = pm4py.conformance.precision_alignments(log, imsfsnet_f, iimf, ifmf)
        #ialignment_result = pm4py.conformance.fitness_alignments(log, imsfsnet, iim, ifm)
        #imsfsresults_g = pm4py.conformance.generalization_tbr(log_im, inet, iim, ifm)
        #imsfs_fresults = pm4py.conformance.generalization_tbr(log, imsfsnet_f, iimf, ifmf)
        #imsfsresults_se = simplicity_evaluator.apply(inet, variant=simplicity_evaluator.Variants.EXTENDED_CYCLOMATIC)
        #imsfsresults_f = simplicity_evaluator.apply(imsfsnet_f, variant=simplicity_evaluator.Variants.EXTENDED_CYCLOMATIC)
        #imsfsresults_card = simplicity_evaluator.apply(inet, variant=simplicity_evaluator.Variants.EXTENDED_CARDOSO)
        #imsfsresults_simp = simplicity_evaluator.apply(inet, variant=simplicity_evaluator.Variants.SIMPLICITY_ARC_DEGREE)
        #ialignment_result_f = simplicity_evaluator.apply(imsfsnet_f, variant=simplicity_evaluator.Variants.EXTENDED_CARDOSO)
        #detailed = pm4py.conformance.conformance_diagnostics_token_based_replay(log, imsfsnet, iim, ifm)
        #print(f"Prec TB synth ohne ss: {imsfsresults}")
        #print(f"Fitness TB : {imsfsresults_f}")
        #print(f"Fitness Alig  : {ialignment_result}")
        #print(f"Precision Alig : {ialignment_result_f}")
        #print(f"Gen synth : {imsfsresults_g}")
        #print(f"Simp EC : {imsfsresults_se}")
        #print(f"Simp ECard : {imsfsresults_card}")
        #print(f"Simp arc : {imsfsresults_simp}")
        #imsfsresults = pm4py.conformance.generalization_tbr(log, net, im, fm)
        #print(f"Generalization IMSFS komplett: {imsfsresults}")
        
        
        
        #print(f"Alignment precision komplett: {alignment_result}")

    def test_ft_net(self, log_name=log):
        self.dummy_variable = "dummy_value"
        #file_im = "LisaB_Testlogs/25-03-31_Netze/bpi2012o_im.pnml"
        #file_imsfs = "LisaB_Testlogs/25-03-31_Netze/bpi2012o_synth.pnml"
        #log = xes_importer.apply("LisaB_Testlogs/25-03-31_Netze/bpi2012o_Teillog_start_stop.xes")
        #net, im, fm = pm4py.read_pnml(file_imsfs)
        #results = pm4py.conformance.precision_token_based_replay(log, net, im, fm)
        #alignment_result = pm4py.conformance.precision_alignments(log, net, im, fm)
        #print(f"Token based precision ft: {results}")
        #print(f"Alignment precision ft: {alignment_result}")

    def test_make_start_stop(self, log_name=log):
        self.dummy_variable = "dummy_value"
        #file="LisaB_Testlogs/25-04-10/bpi2012o_IM_FT.pnml"
        #net, im, fm = pm4py.read_pnml(file)
        #for place in net.places:
            #if not place.in_arcs:
                #source = place
            #if not place.out_arcs:
                #sink = place
        #for arc in source.out_arcs:
            #t = arc.target
            #t.name = "Start"
            #t.label = "Start"

        #for arc in sink.in_arcs:
            #t = arc.source
            #t.name = "Stop"
            #t.label = "Stop" 
        #vis.view_petri_net(net, im, fm, format="svg")
        #pm4py.write_pnml(net, im, fm, "bpi2020neu_IMSFS.pnml")

    def test_rem_start_stop(self, log_name=log):
        self.dummy_variable = "dummy_value"
        #file="LisaB_Testlogs/25-04-10/bpi2012o_IMSFS_synth.pnml"
        #im_file= "LisaB_Testlogs/25-04-10/bpi2012o_IM_FT.pnml"
        #inet, iim, ifm = pm4py.read_pnml(im_file)
        #net, im, fm = pm4py.read_pnml(file)
        #start = petri_utils.get_transition_by_name(net, "Start" )
        #stop = petri_utils.get_transition_by_name(net, "Stop")
        #start.name="skip_start"
        #start.label=None
        #stop.name="skip_stop"
        #stop.label=None
        #vis.view_petri_net(net, im, fm, format="svg")
        #pm4py.write_pnml(net, im, fm, "bpi2012oeu_IMSFS.pnml")
        #log_name_im = "LisaB_Testlogs/25-04-10/bpi2012o_IM_FT.xes"
        #log = xes_importer.apply(log_name_im)
        #results_p = pm4py.conformance.precision_token_based_replay(log, net, im, fm)
        #results_gen = pm4py.conformance.generalization_tbr(log, net, im, fm)
        #alignment_result_p = pm4py.conformance.precision_alignments(log, net, im, fm)
        #print(f"prec tb: {results_p}")
        #print(f"Gen: {results_gen}")
        #print(f"prec_alig: {alignment_result_p}")


if __name__ == "__main__":
    unittest.main()
