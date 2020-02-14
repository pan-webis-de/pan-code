#!/usr/bin/env python

"""Calculates the measures for the PAN18 style change detection task"""

from __future__ import division

import json
import os
import getopt
import sys

evaluationOutputFileName = "evaluation.prototext"


def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["inputDataset=", "inputRun=", "outputDir="]
        opts, _ = getopt.getopt(sys.argv[1:], "d:r:o:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputDataset = "undefined"
    inputRun = "undefined"
    outputDir = "undefined"

    for opt, arg in opts:
        if opt in ("-d", "--inputDataset"):
            inputDataset = arg
        elif opt in ("-r", "--inputRun"):
            inputRun = arg
        elif opt in ("-o", "--outputDir"):
            outputDir = arg
        else:
            assert False, "Unknown option."
    if inputDataset == "undefined":
        sys.exit("Input dataset is undefined. Use option -d or --inputDataset.")
    elif not os.path.exists(inputDataset):
        sys.exit("The input dataset folder does not exist (%s)." % inputDataset)

    if inputRun == "undefined":
        sys.exit("Input run is undefined. Use option -r or --inputRun.")
    elif not os.path.exists(inputRun):
        sys.exit("The input run folder does not exist (%s)." % inputRun)

    if outputDir == "undefined":
        sys.exit("Output path undefined. Use option -o or --outputDir.")
    elif not os.path.exists(outputDir):
        os.mkdir(outputDir)

    return (inputDataset, inputRun, outputDir)


def getMeasureString(measureName, value):
    """Returns the string represenation of one measure with its value."""
    return "measure{\n  key: \"" + measureName + "\"\n  value: \"" + str(value) + "\"\n}"


########## MAIN ##########


def main(inputDataset, inputRun, outputDir):
    """Main method of this module."""

    totalProblemsCount = 0
    totalCorrectCount = 0
    solvedProblemsCount = 0
    solvedCorrectCount = 0


    for file in os.listdir(inputDataset):
        if file.endswith(".txt"):
            totalProblemsCount = totalProblemsCount + 1
            filePrefix = os.path.splitext(file)[0]
            truthFileName = inputDataset + "/" + filePrefix + ".truth"

            with open(truthFileName) as truthFile:
                producedTruthFilename = inputRun + "/" + filePrefix + ".truth"
                if os.path.exists(producedTruthFilename):
                    solvedProblemsCount = solvedProblemsCount + 1
                    with open(producedTruthFilename) as producedTruthFile:
                        truthData = json.load(truthFile)
                        producedData = json.load(producedTruthFile)

                        if not "changes" in truthData:
                            sys.exit("There is no 'changes' key in " + truthFileName)
                        if not "changes" in producedData:
                            sys.exit("There is no 'changes' key in " + producedTruthFilename)

                        if truthData['changes'] == producedData['changes']:
                            totalCorrectCount = totalCorrectCount + 1
                            solvedCorrectCount = solvedCorrectCount + 1



    if totalProblemsCount == 0:
        sys.exit("The input dataset folder (%s) contains no style change detection problems. Be sure to pass the correct folder (option -d)." % inputDataset)

    if solvedProblemsCount == 0:
        sys.exit("The output folder (%s) contains no solved style change detection problems (.truth-files). Be sure to pass the correct folder (option -r)." % inputRun)

    print("total problems: %s" % totalProblemsCount)
    print("total correct: %s" % totalCorrectCount)
    print("solved problems: %s" % solvedProblemsCount)
    print("solved correct: %s\n" % solvedCorrectCount)

    outStr = getMeasureString("accuracy", totalCorrectCount / totalProblemsCount)
    outStr += "\n" + getMeasureString("accuracy_solved", solvedCorrectCount / solvedProblemsCount)

    print(outStr)

    with open(outputDir + "/" + evaluationOutputFileName, 'w') as outFile:
        outFile.write(outStr)

    print("\nThe results have been written to the output folder.")


if __name__ == '__main__':
    main(*parse_options())

