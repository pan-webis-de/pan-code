#!/usr/bin/env python

"""Calculates the measures for the PAN17 style breach detection task"""

from __future__ import division

import json
import os
import getopt
import sys
import windowdiff
import winpr


evaluationOutputFileName = "evaluation.prototext"



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

def computeMeasures(inputText, groundTruthData, producedData):
    """Computes WindowDiff and WinPR for the given data"""

    (groundTruthWordBorders, producedDataWordBorders, totalWordCount) = getWordPositionsFromCharacterPositions(inputText, groundTruthData["borders"], producedData["borders"])

    groundTruthStrArray = []
    producedStrArray = []

    for i in range(0,totalWordCount-1):
        if i in groundTruthWordBorders: groundTruthStrArray.append("1")
        else: groundTruthStrArray.append("0")

        if i in producedDataWordBorders: producedStrArray.append("1")
        else: producedStrArray.append("0")

    # mark last border
    groundTruthStrArray.append("1")
    producedStrArray.append("1")

    groundTruthString = ''.join(groundTruthStrArray)
    producedString = ''.join(producedStrArray)

    halfSegmentLength = 0
    if len(groundTruthWordBorders) == 0: halfSegmentLength = round(totalWordCount / 2)
    else: halfSegmentLength = round(totalWordCount / (len(groundTruthWordBorders) + 1) / 2)

    winDiff = windowdiff.windowdiff(groundTruthString, producedString, halfSegmentLength)
    winPR = winpr.WinPR(groundTruthString, producedString, halfSegmentLength)
    winP = winPR.precision()
    winR = winPR.recall()

    #print("winDiff: ", winDiff)
    #print("winR: ", winR)
    #print("winP: ", winP)
    #print("winF: ", fscore(winR, winP))

    return (winDiff, winR, winP, fscore(winR, winP))


def getWordPositionsFromCharacterPositions(text, groundTruthCharPositions, producedCharPositions):
    wordCount = 0
    groundTruthWordPositions = []
    producedWordPositions = []

    for i in range(1,len(text)-1):
        if text[i] == ' ' and text[i-1] != ' ':
            wordCount = wordCount + 1
        if i in groundTruthCharPositions:
            groundTruthWordPositions.append(wordCount)
        if i in producedCharPositions:
            producedWordPositions.append(wordCount)

    return (groundTruthWordPositions, producedWordPositions, wordCount+1)


########## MAIN ##########


def main(inputDataset, inputRun, outputDir):
    """Main method of this module."""

    problemsCount = 0
    totalWinDiff = 0
    totalWinR = 0
    totalWinP = 0
    totalWinF = 0

    for file in os.listdir(inputDataset):
        if file.endswith(".txt"):
            problemsCount = problemsCount + 1
            filePrefix = os.path.splitext(file)[0]
            txtFileName = inputDataset + "/" + filePrefix + ".txt"
            truthFileName = inputDataset + "/" + filePrefix + ".truth"

            inputText = ""
            with open(txtFileName) as txtFile:
                inputText = txtFile.read()

            with open(truthFileName) as truthFile:
                #print("%s" % ("processing " + filePrefix))
                producedTruthFilename = inputRun + "/" + filePrefix + ".truth"
                if os.path.exists(producedTruthFilename):
                    with open(producedTruthFilename) as producedTruthFile:
                        truthData = json.load(truthFile)
                        producedData = json.load(producedTruthFile)

                        if not "borders" in truthData:
                            sys.exit("There is no 'borders' key in " + truthFileName)
                        if not "borders" in producedData:
                            sys.exit("There is no 'borders' key in " + producedTruthFilename)

                        (winDiff, winR, winP, winF) = computeMeasures(inputText, truthData, producedData)
                        totalWinDiff += winDiff
                        totalWinR += winR
                        totalWinP += winP
                        totalWinF += winF

    if problemsCount == 0:
        sys.exit("The input dataset folder (%s) contains no style breach detection problems. Be sure to pass the correct folder (option -d)." % inputDataset)

    totalWinDiff = totalWinDiff / problemsCount
    totalWinR = totalWinR / problemsCount
    totalWinP = totalWinP / problemsCount
    totalWinF = totalWinF / problemsCount

    outStr = getMeasureString("windowDiff", totalWinDiff)
    outStr += "\n" + getMeasureString("winP", totalWinP)
    outStr += "\n" + getMeasureString("winR", totalWinR)
    outStr += "\n" + getMeasureString("winF", totalWinF)

    with open(outputDir + "/" + evaluationOutputFileName, 'w') as outFile:
        outFile.write(outStr)

    print("%s" % ("The results have been written to the output folder."))
    print("%s" % outStr)

if __name__ == '__main__':
    main(*parse_options())

