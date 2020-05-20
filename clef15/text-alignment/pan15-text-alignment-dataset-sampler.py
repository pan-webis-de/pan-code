#!/usr/bin/env python

import sys
import os
from os import remove, mkdir
from os.path import exists, join, isdir
from shutil import copy2, move
from random import shuffle

class TextAlignmentDatasetSampler():

    DATASET = "."

    def sample_dataset(self):
        self.sample_obfuscation_dirs()
        self.copy_missing_srcsusp_files()


    def sample_obfuscation_dirs(self):
        dirs = [d for d in os.listdir(self.DATASET) if isdir(join(self.DATASET, d)) and not d == "src" and not d == "susp"]
        for d in dirs:
            self.sample_obfuscation_dir(d)
        self.regenerate_global_pairs_file(dirs)


    def sample_obfuscation_dir(self, dirname1):
        dirname2 = dirname1 + "2"
        dir1 = join(self.DATASET, dirname1)
        dir2 = join(self.DATASET, dirname2)
        pairs1 = join(dir1, "pairs")
        pairs2 = join(dir2, "pairs")
        mkdir(dir2)
        self.divide_pairs(pairs1, pairs2)
        self.move_xml_files(dir1, dir2, pairs2)
        self.divide_srcsusp("susp", pairs2, 0)
        self.divide_srcsusp("src", pairs2, 1)


    def divide_pairs(self, pairs1, pairs2):
        lines = self.read_file_lines(pairs1)
        shuffle(lines)
        half1 = lines[:len(lines) // 2]
        half2 = lines[len(lines) // 2:]
        half1.sort()
        half2.sort()
        self.write_file_lines(pairs1, half1)
        self.write_file_lines(pairs2, half2)


    def move_xml_files(self, dir1, dir2, pairs2):
        lines = self.read_file_lines(pairs2)
        xmlfiles = [line.strip().replace(".txt ", "-").replace(".txt", ".xml") for line in lines]
        for xmlfile in xmlfiles:
            file1 = join(dir1, xmlfile)
            file2 = join(dir2, xmlfile)
            move(file1, file2)


    def divide_srcsusp(self, dirname1, pairs2, splitlistentry):
        dirname2 = dirname1 + "2"
        dir1 = join(self.DATASET, dirname1)
        dir2 = join(self.DATASET, dirname2)
        if not exists(dir2):
            mkdir(dir2)
        lines = self.read_file_lines(pairs2)
        files = [line.strip().split()[splitlistentry] for line in lines]
        for f in files:
            file1 = join(dir1, f)
            file2 = join(dir2, f)
            if exists(file1) and not exists(file2):
                move(file1, file2)


    def regenerate_global_pairs_file(self, dirnames):
        globalpairs1 = join(self.DATASET, "pairs")
        globalpairs2 = join(self.DATASET, "pairs2")
        if exists(globalpairs1):
            remove(globalpairs1)
        for dirname1 in dirnames:
            dirname2 = dirname1 + "2"
            pairs1 = join(self.DATASET, dirname1, "pairs")
            pairs2 = join(self.DATASET, dirname2, "pairs")
            self.read_and_append(pairs1, globalpairs1)
            self.read_and_append(pairs2, globalpairs2)
        self.sort_file_lines(globalpairs1)
        self.sort_file_lines(globalpairs2)


    def read_and_append(self, filetoread, filetoappend):
        lines = self.read_file_lines(filetoread)
        self.write_file_lines(filetoappend, lines, mode="a")


    def sort_file_lines(self, filetosort):
        lines = self.read_file_lines(filetosort)
        lines.sort()
        self.write_file_lines(filetosort, lines);


    def copy_missing_srcsusp_files(self):
        pairs = join(self.DATASET, "pairs")
        lines = self.read_file_lines(pairs)
        lines = [line.strip().split() for line in lines]
        for line in lines:
            susp1 = join(self.DATASET, "susp", line[0])
            susp2 = join(self.DATASET, "susp2", line[0])
            src1 = join(self.DATASET, "src", line[1])
            src2 = join(self.DATASET, "src2", line[1])
            if not exists(susp1):
                copy2(susp2, susp1)
            if not exists(src1):
                copy2(src2, src1)


    def read_file_lines(self, filetoread):
        with open(filetoread, "r") as filein:
            lines = filein.readlines()
        # Append a newline in case the last line didn't end with one.
        lines[-1] = lines[-1].rstrip('\n') + '\n'
        return lines


    def write_file_lines(self, filtetowrite, lines, mode="w"):
        with open(filtetowrite, mode) as fileout:
            fileout.writelines(lines)




if __name__ == '__main__':
    if len(sys.argv) == 0:
       print("Usage: python pan15-text-alignment-dataset-sampler.py <path/to/corpus>")
    TextAlignmentDatasetSampler.DATASET = sys.argv.pop()
    tads = TextAlignmentDatasetSampler()
    tads.sample_dataset()

