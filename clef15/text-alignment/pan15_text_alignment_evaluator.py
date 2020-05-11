#!/usr/bin/env python2
# -*- coding: utf8 -*-
"""
    Global Text Alignment evaluator

    Copyright 2015-today

    Project WEBIS
    Author: Steve Göring
"""
import sys
import os
import argparse

from pan15_text_alignment_evaluator_case_level import case_level_performance
from pan15_text_alignment_evaluator_document_level import document_level_performance
from pan15_text_alignment_evaluator_character_level import character_level_performance


def main(args):

    parser = argparse.ArgumentParser(description='Global Text Alignment evaluator', epilog="Steve Göring 2015", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('truthdir', type=str, help='Path to the XML files with plagiarism annotations')
    parser.add_argument('inputdir', type=str, help='Path to the XML files with detection annotations')
    parser.add_argument('outputfile', type=str, help='Name of output file')
    parser.add_argument('--tau-precision', "--taup", type=float, default=0.5, help='Tau precision')
    parser.add_argument('--tau-recall', "--taur", type=float, default=0.5, help='Tau recall')

    parser.add_argument('--det-tag', type=str, default="detected-plagiarism", help='Tag name of the detection annotations')
    parser.add_argument('--plag-tag', type=str, default="plagiarism", help='Tag name suffix of plagiarism annotations')
    parser.add_argument('--micro', action='store_true', help='Compute micro-averaged recall and precision')

    argsdict = vars(parser.parse_args(args))

    micro_averaged = argsdict["micro"]
    plag_path = argsdict["truthdir"]
    plag_tag_name = argsdict["plag_tag"]
    det_path = argsdict["inputdir"]
    det_tag_name = argsdict["det_tag"]
    tau_precision = argsdict["tau_precision"]
    tau_recall = argsdict["tau_recall"]
    outputfilename = argsdict["outputfile"]

    result = {}

    try:
        result["plagdet"], result["recall"], result["precision"], result["granularity"], result["documents"] = character_level_performance(micro_averaged, plag_path, plag_tag_name, det_path, det_tag_name)
    except Exception, e:
        result["plagdet"], result["recall"], result["precision"], result["granularity"], result["documents"] = 5 * [0]
        result["char_level_error"] = str(e)

    try:
        result["document_level_precision"], result["document_level_recall"], result["document_level_fmeasure"] = document_level_performance(plag_path, det_path, os.path.join(plag_path, 'pairs'), tau_recall, tau_precision)
    except Exception, e:
        result["document_level_precision"], result["document_level_recall"], result["document_level_fmeasure"] = 3 * [0]
        result["document_level_error"] = str(e)

    try:
        result["case_level_precision"], result["case_level_recall"], result["case_level_fmeasure"] = case_level_performance(plag_path, det_path, os.path.join(plag_path, 'pairs'), tau_recall, tau_precision)
    except Exception, e:
        result["case_level_precision"], result["case_level_recall"], result["case_level_fmeasure"] = 3 * [0]
        result["case_level_error"] = str(e)

    output = "{\n"
    for i in sorted(list(result.keys())):
        output += """ "{}" : "{}", \n""".format(i, result[i])
    output += "}"
    print(output)

    prototext_filename = os.path.splitext(outputfilename)[0] + ".prototext"

    prototext_file = open(prototext_filename, "w")

    for i in sorted(list(result.keys())):
        proto = "measure{\n  " + """key  : "{}"\n  value: "{}"\n""".format(i, result[i]) + "}\n"
        prototext_file.write(proto)

    prototext_file.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
