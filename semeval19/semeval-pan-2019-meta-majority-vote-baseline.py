#!/usr/bin/env python

"""Majority vote baseline for the PAN19 hyperpartisan news detection task meta evaluation"""
# Version: 2019-02-07

# Parameters:
# --inputDataset=<directory>
#   Directory that contains the articles txt file with the articles for which a prediction should be made.
# --outputDir=<directory>
#   Directory to which the predictions will be written. Will be created if it does not exist.

from __future__ import division

import os
import getopt
import sys
import csv

runOutputFileName = "prediction.txt"


def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["inputDataset=", "outputDir="]
        opts, _ = getopt.getopt(sys.argv[1:], "d:o:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputDataset = "undefined"
    outputDir = "undefined"

    for opt, arg in opts:
        if opt in ("-d", "--inputDataset"):
            inputDataset = arg
        elif opt in ("-o", "--outputDir"):
            outputDir = arg
        else:
            assert False, "Unknown option."
    if inputDataset == "undefined":
        sys.exit("Input dataset, the directory that contains the articles txt file, is undefined. Use option -d or --inputDataset.")
    elif not os.path.exists(inputDataset):
        sys.exit("The input dataset folder does not exist (%s)." % inputDataset)

    if outputDir == "undefined":
        sys.exit("Output path, the directory into which the predictions should be written, is undefined. Use option -o or --outputDir.")
    elif not os.path.exists(outputDir):
        os.mkdir(outputDir)

    return (inputDataset, outputDir)


########## MAIN ##########


def main(inputDataset, outputDir):
    """Main method of this module."""

    with open(outputDir + "/" + runOutputFileName, 'w') as outFile:
        for file in os.listdir(inputDataset):
            if file.endswith(".txt"):
                with open(inputDataset + "/" + file) as inputRunFile:
                    reader = csv.reader(inputRunFile, delimiter=' ')
                    for row in reader:
                        articleId = row[0]
                        prediction = "false"
                        trues = 0
                        for i in range(2, len(row)):
                            if row[i] == "true":
                                trues += 1

                        if trues >= (len(row) - 1) / 2:
                            prediction = "true"

                        outFile.write(articleId + " " + prediction + "\n")



    print("The predictions have been written to the output folder.")


if __name__ == '__main__':
    main(*parse_options())

