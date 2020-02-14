#!/usr/bin/env python

"""Calculates the measures for the PAN19 style change detection task"""

from __future__ import division

import json
import os
import getopt
import sys
import numpy as np

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


def OrdinalClassificationIndex(cMatrix):
    """ computes the OCI for a given confusion matrix proposed by Cardoso and Recardo, Measuring the performance of ordinal classification, International Journal of Pattern Recognition and Artificial Intelligence (2011)
    Ported to Python from the original MATLAB script provided by the authors."""
    K = np.shape(cMatrix)[0]
    N = np.sum(cMatrix)
    ggamma = 1
    bbeta  = 0.75/np.power(N*(K-1),ggamma)

    helperM2 = np.zeros_like(cMatrix)

    for r in range(0,K):
        for c in range(0,K):
            helperM2[r][c] = cMatrix[r][c] * np.power((abs(r-c)), ggamma)

    TotalDispersion = (np.power(np.sum(helperM2),(1/ggamma)))
    helperM1 = cMatrix/(TotalDispersion+N)

    errMatrix = np.zeros_like(cMatrix, dtype=np.float)
    errMatrix[0][0] = 1 - helperM1[0][0] + bbeta*helperM2[0][0]

    for r in range(1,K):
        c=0
        errMatrix[r][c] = errMatrix[r-1][c] - helperM1[r][c] + bbeta*helperM2[r][c]

    for c in range(1,K):
        r=0
        errMatrix[r][c] = errMatrix[r][c-1] - helperM1[r][c] + bbeta*helperM2[r][c]

    for c in range(1,K):
        for r in range(1,K):
            costup = errMatrix[r-1, c]
            costleft = errMatrix[r, c-1]
            lefttopcost = errMatrix[r-1, c-1]
            aux = np.min([costup, costleft, lefttopcost])
            errMatrix[r][c] = aux - helperM1[r][c] + bbeta*helperM2[r][c]

    return errMatrix[-1][-1]


def read_authors(filename):
    with open(filename) as f:
        data = json.load(f)

        if not "authors" in data:
            sys.exit("There is no 'authors' key in " + filename)

        try:
            num_authors = int(data["authors"])
            if num_authors <= 0:
                sys.exit("Number of authors cannot be 0 or negative: " + filename)

            return num_authors

        except:
            sys.exit("Cannot parse number of authors in " + filename)


def get_max_authors(inputDataset, inputRun):
    '''extracts the maximum number of authors appearing in the ground truth and in the predictions'''

    max_authors = 0

    files = []
    files += [inputDataset + "/" + f for f in os.listdir(inputDataset)]
    files += [inputRun + "/" + f for f in os.listdir(inputRun)]

    for file in files:
        if file.endswith(".truth"):
            num_authors = read_authors(file)
            if num_authors > max_authors:
                max_authors = num_authors

    return max_authors





########## MAIN ##########


def main(inputDataset, inputRun, outputDir):
    """Main method of this module."""

    total_problems_count = 0
    total_sm_correct_count = 0

    max_authors = get_max_authors(inputDataset, inputRun)
    confusion_matrix = np.zeros((max_authors, max_authors), dtype=int)

    for file in os.listdir(inputDataset):
        if file.endswith(".txt"):
            total_problems_count = total_problems_count + 1
            filePrefix = os.path.splitext(file)[0]
            truthFileName = inputDataset + "/" + filePrefix + ".truth"
            producedTruthFilename = inputRun + "/" + filePrefix + ".truth"

            if not os.path.exists(producedTruthFilename):
                sys.exit(truthFileName + " is missing")


            num_correct = read_authors(truthFileName)
            num_predicted = read_authors(producedTruthFilename)
            num_predicted = np.random.randint(1,6)

            if (num_correct == 1 and num_predicted == 1) or (num_correct > 1 and num_predicted > 1):
                total_sm_correct_count += 1

            if num_correct > 1:
                confusion_matrix[num_correct-1][num_predicted-1] += 1


    if total_problems_count == 0:
        sys.exit("The input dataset folder (%s) contains no style change detection problems. Be sure to pass the correct folder (option -d)." % inputDataset)

    # print(confusion_matrix)
    # print("\n")

    accuracy_sm = total_sm_correct_count / total_problems_count
    OCI = OrdinalClassificationIndex(confusion_matrix)
    rank = (accuracy_sm + (1-OCI)) / 2

    outStr = getMeasureString("accuracy_sm", accuracy_sm)
    outStr += getMeasureString("OCI", OCI)
    outStr += getMeasureString("rank",  rank)

    print(outStr)

    with open(outputDir + "/" + evaluationOutputFileName, 'w') as outFile:
        outFile.write(outStr)

    print("\nThe results have been written to the output folder.")


if __name__ == '__main__':
    main(*parse_options())

