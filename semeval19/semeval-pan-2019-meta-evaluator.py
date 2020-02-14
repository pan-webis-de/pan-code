#!/usr/bin/env python

"""Calculates the measures for the PAN19 hyperpartisan news detection task meta classification"""
# Version: 2019-02-04

# Parameters:
# --inputDataset=<directory>
#   Directory that contains the ground truth txt file with the articles for which a prediction should have been made.
# --inputRun=<directory>
#   Directory that contains the prediction for the articles in the ground truth txt file. The format of the .txt file should be, one article per line:
#     <article id> <prediction> [<confidence>]
#   where:
#     - article id   corresponds to the first column in the articles and ground truth files
#     - prediction   is either "true" (hyperpartisan) or "false" (not hyperpartisan)
#     - confidence   is an optional value to describe the confidence of the predictor in the prediction---the higher, the more confident. If missing, a value of 1 is used. However, the absolute values are unimportant: this may just be used in the future to order the predictions, for example to calculate ROC curves.
# --outputDir=<directory>
#   Directory to which the evaluation will be written. Will be created if it does not exist.

from __future__ import division

import json
import os
import getopt
import sys
import csv

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

groundTruth = {}
def main(inputDataset, inputRun, outputDir):
    """Main method of this module."""

    for file in os.listdir(inputDataset):
        if file.endswith(".txt"):
            with open(inputDataset + "/" + file) as inputRunFile:
                reader = csv.reader(inputRunFile, delimiter=' ')
                for row in reader:
                    articleId = row[0]
                    hyperpartisan = row[1]
                    groundTruth[articleId] = hyperpartisan.lower()

    truePositivesCount  = 0 # run:true  groundtruth:true
    trueNegativesCount  = 0 # run:false groundtruth:false
    falsePositivesCount = 0 # run:true  groundtruth:false
    falseNegativesCount = 0 # run:false groundtruth:true

    # read in predictions
    for file in os.listdir(inputRun):
        if file.endswith(".txt"):
            with open(inputRun + "/" + file) as inputRunFile:
                lines = inputRunFile.readlines()
                for line in lines:
                    values = line.rstrip('\n').split()
                    articleId = values[0]
                    prediction = values[1].lower()
                    confidence = 1
                    if (len(values) > 2):
                        confidence = values[2]
                    
                    hyperpartisan = groundTruth[articleId]
                    del groundTruth[articleId]
                    if hyperpartisan == "true":
                        if prediction == "true":
                            truePositivesCount += 1
                        else:
                            falseNegativesCount += 1
                    else:
                        if prediction == "true":
                            falsePositivesCount += 1
                        else:
                            trueNegativesCount += 1

    predictionCount = truePositivesCount + trueNegativesCount + falsePositivesCount + falseNegativesCount
    if len(groundTruth) > 0:
        missing = len(groundTruth)
        print("WARNING: Missing %s predictions that will be treated as non-hyperpartisan\n" % len(groundTruth))
        for articleId, hyperpartisan in groundTruth.items():
            predictionCount += 1
            # prediction is by default false
            if hyperpartisan == "true":
                falseNegativesCount += 1
            else:
                trueNegativesCount += 1

    print("true positives: %s" % truePositivesCount)
    print("true negatives: %s" % trueNegativesCount)
    print("false positives: %s" % falsePositivesCount)
    print("false negatives: %s\n" % falseNegativesCount)

    accuracy  = (truePositivesCount + trueNegativesCount) / predictionCount
    precision = truePositivesCount / (truePositivesCount + falsePositivesCount)
    recall    = truePositivesCount / (truePositivesCount + falseNegativesCount)
    f1        = 2 * precision * recall / (precision + recall)

    outStr = getMeasureString("accuracy", accuracy)
    outStr += "\n" + getMeasureString("precision", precision)
    outStr += "\n" + getMeasureString("recall", recall)
    outStr += "\n" + getMeasureString("f1", f1)
    print(outStr)

    with open(outputDir + "/" + evaluationOutputFileName, 'w') as outFile:
        outFile.write(outStr)

    print("\nThe results have been written to the output folder.")


if __name__ == '__main__':
    main(*parse_options())

