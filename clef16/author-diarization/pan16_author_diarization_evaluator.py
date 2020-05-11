#!/usr/bin/env python

"""Calculates the measures for the PAN16 diarization task"""

from __future__ import division

import json
import pan09
import os
import getopt
import sys


evaluationOutputFileName = "evaluation.prototext"


########## INTRINSIC PLAGIARISM DETECTION EVALUATION ##########

def getPlagiarismCases(jsonFile):
    """parses the plagiarism cases out of a clustering solution file, i.e., returns all clusters except the main author cluster."""
    with open(jsonFile) as data_file:
        data = json.load(data_file)

        clusterIndex = 0
        maxCharCount = 0
        mainAuthorClusterIndex = 0

        for cluster in data["authors"]:
            charCount = 0
            for fromToPosition in cluster:
                charCount += fromToPosition["to"] - fromToPosition["from"]

            if charCount > maxCharCount:
                maxCharCount = charCount
                mainAuthorClusterIndex = clusterIndex

            clusterIndex += 1

        # print("main author cluster index: %d (%d)" % (mainAuthorClusterIndex, charCount))

        annotations = []

        clusterIndex = 0
        for cluster in data["authors"]:
            if clusterIndex != mainAuthorClusterIndex:
                for fromToPosition in cluster:
                    offset = fromToPosition["from"]
                    length = fromToPosition["to"] - offset
                    annotations.append(pan09.Annotation('ann', offset, length, '', 0, 0, False))
            clusterIndex += 1

        return annotations



########## DIARIZATION EVALUATION ##########


def getFromTos(jsonFile):
    """Reads and returns the clusters (from-to's) from the given json file."""
    with open(jsonFile) as file:
        data = json.load(file)
        return data["authors"]


def getCharOverlapCount(from1, to1, from2, to2):
    """Calculates the number of overlapping characters of the two given areas."""

    #order such that from1 is always prior from2
    if from1 > from2:
        tmp = from1
        from1 = from2
        from2 = tmp
        tmp = to1
        to1 = to2
        to2 = tmp

    if from2 >= from1 and from2 <= to1:
        if to2 > to1:
            return to1 - from2 + 1
        else:
            return to2 - from2 + 1
    else:
        return 0



def getClusterOverlapCount(cluster1, cluster2):
    """Calculates the number of overlapping characters ot the two given clusters (i.e., lists of from-to's)."""
    overlapCount = 0
    for fromTo1 in cluster1:
        for fromTo2 in cluster2:
            overlapCount += getCharOverlapCount(fromTo1["from"], fromTo1["to"], fromTo2["from"], fromTo2["to"])

    return overlapCount


def getClusterCharCount(cluster):
    """Calculates the total number of characters occurring in a cluster."""
    charCount = 0
    for fromTo in cluster:
        charCount += fromTo["to"] - fromTo["from"] + 1;
    return charCount


def computeMeasure(testClusterFromTos, truthClusterFromTos):
    """Computes the BCubed precision, if testClusterFromTos = testClusters and truthClusterFromTos = truthClusters. Calculates the BCUBED recall if the clusters are passed interchanged."""
    allClustersCharCount = 0;
    score = 0.0;
    for cluster1 in testClusterFromTos:
        clusterCharCount = getClusterCharCount(cluster1)
        allClustersCharCount += clusterCharCount
        for cluster2 in truthClusterFromTos:
            overlap = getClusterOverlapCount(cluster1, cluster2)
            clusterScore = 0
            if overlap > 0:
                clusterScore = overlap * overlap / clusterCharCount
                score += clusterScore

    return score / allClustersCharCount


def computeBCubedMeasures(clusteringJson, groundTruthJson):
    """Computes the BCubed precision, recall and F-score."""
    clusteringFromTos = getFromTos(clusteringJson)
    groundTruthFromTos = getFromTos(groundTruthJson)
    #pprint(clusteringFromTos)
    #pprint(groundTruthFromTos)

    recall = computeMeasure(groundTruthFromTos, clusteringFromTos)
    precision = computeMeasure(clusteringFromTos, groundTruthFromTos)

    return precision, recall




########## COMMON METHODS ##########


def fscore(rec, prec, beta=1.0):
    """Computes the F_{beta}-score of given precision and recall values."""
    if (rec == 0 and prec == 0) or prec < 0 or rec < 0:
        return 0.0
    return (1.0 + beta) * (prec * rec / (beta**2 * prec + rec))


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


def getPlagiarismDetectionMeasuresString(totalMicroRecall, totalMicroPrecision, totalMacroRecall, totalMacroPrecision):
    """Generates and returns the measures string for the plagiarism detection evaluation results."""
    outStr = getMeasureString("micro-recall", totalMicroRecall)
    outStr += "\n" + getMeasureString("micro-precision", totalMicroPrecision)
    outStr += "\n" + getMeasureString("micro-f", fscore(totalMicroRecall, totalMicroPrecision))
    outStr += "\n" + getMeasureString("macro-recall", totalMacroRecall)
    outStr += "\n" + getMeasureString("macro-precision", totalMacroPrecision)
    outStr += "\n" + getMeasureString("macro-f", fscore(totalMacroRecall, totalMacroPrecision))

    return outStr;

def getDiarizationMeasuresString(totalBCubedRecall, totalBCubedPrecision):
    """Generates and returns the measures string for the diarization evaluation results."""
    outStr = getMeasureString("bcubed-recall", totalBCubedRecall)
    outStr += "\n" + getMeasureString("bcubed-precision", totalBCubedPrecision)
    outStr += "\n" + getMeasureString("bcubed-f", fscore(totalBCubedRecall, totalBCubedPrecision))

    return outStr;

########## MAIN ##########


def main(inputDataset, inputRun, outputDir):
    """Main method of this module."""

    totalMicroRecall = 0
    totalMicroPrecision = 0
    totalMacroRecall = 0
    totalMacroPrecision = 0
    totalBCubedRecall = 0
    totalBCubedPrecision = 0

    plagProblemsCount = 0
    diarizationProblemsCount = 0


    for file in os.listdir(inputDataset):
        if file.endswith(".txt"):
            filePrefix = os.path.splitext(file)[0]
            with open(inputDataset + "/" + filePrefix + ".meta") as data_file:
                data = json.load(data_file)
                producedTruthFile = inputRun + "/" + filePrefix + ".truth"
                groundTruthFile = inputDataset + "/" + filePrefix + ".truth"

                if not "type" in data or data["type"] == "plagiarism":
                    plagProblemsCount += 1
                    if os.path.exists(producedTruthFile):
                        groundTruthCases = getPlagiarismCases(groundTruthFile)
                        detectionCases = getPlagiarismCases(producedTruthFile)

                        microRecall, microPrecision = pan09.micro_avg_recall_and_precision(groundTruthCases, detectionCases)
                        macroRecall, macroPrecision = pan09.macro_avg_recall_and_precision(groundTruthCases, detectionCases)

                        totalMicroRecall += microRecall
                        totalMicroPrecision += microPrecision
                        totalMacroRecall += macroRecall
                        totalMacroPrecision += macroPrecision

                elif data["type"] == "diarization":
                    diarizationProblemsCount += 1
                    if os.path.exists(producedTruthFile):
                        bcubedRecall, bcubedPrecision = computeBCubedMeasures(producedTruthFile, groundTruthFile)
                        totalBCubedRecall += bcubedRecall
                        totalBCubedPrecision += bcubedPrecision


    if plagProblemsCount == 0 and diarizationProblemsCount == 0:
        sys.exit("The input dataset folder (%s) contains no plagiarism detection or diarization problems. Be sure to pass the correct folder (option -d)." % inputDataset)

    outStr = ""

    if plagProblemsCount > 0:
        totalMicroRecall = totalMicroRecall / plagProblemsCount
        totalMicroPrecision = totalMicroPrecision / plagProblemsCount
        totalMacroRecall = totalMacroRecall / plagProblemsCount
        totalMacroPrecision = totalMacroPrecision / plagProblemsCount
        outStr = getPlagiarismDetectionMeasuresString(totalMicroRecall, totalMicroPrecision, totalMacroRecall, totalMacroPrecision)

    if diarizationProblemsCount > 0:
        totalBCubedRecall = totalBCubedRecall / diarizationProblemsCount
        totalBCubedPrecision = totalBCubedPrecision / diarizationProblemsCount
        outStr += getDiarizationMeasuresString(totalBCubedRecall, totalBCubedPrecision)

    with open(outputDir + "/" + evaluationOutputFileName, 'w') as outFile:
        outFile.write(outStr)

    print("%s" % ("The results have been written to the output folder."))
    print("%s" % outStr)

if __name__ == '__main__':
    main(*parse_options())

