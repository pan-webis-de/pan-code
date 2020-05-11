# Natural Language Toolkit: WinPR
#
# Copyright (C) 2001-2012 NLTK Project
# Author: Martin Scaiano <martin@scaiano.com>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT
##########################################################################
# WinPR
# Scaiano, M., Inkpen, D.
# Getting More from Segmentation Evaluation
# NAACL HLT 2012, pp. 362-366
##########################################################################

import sys

def windowsize(gold, boundary="1"):
    return len(gold) / gold.count(boundary)

class WinPR:
    tp = 0.0 # True Positives
    tn = 0.0 # True Negatives
    fp = 0.0 # False Positives
    fn = 0.0 # False Negatives
    
    k = 1 # window size
    
    def __init__(self,gold_seg, hypo_seg, k, boundary="1"):
        self.k = k
        if len(gold_seg) != len(hypo_seg):
            sys.exit("Segmentations have unequal length")
    
        for i in range(len(gold_seg)+1 - self.k):
            self.update( gold_seg[i:i+k].count(boundary), hypo_seg[i:i+k].count(boundary) )
        

    def precision(self):
        if self.tp == 0:
            return 0
        return self.tp / (self.tp + self.fp)
    
    def recall(self):
        if self.tp == 0:
            return 0
        return self.tp / (self.tp + self.fn)
    
    def update(self, goldCount, hypoCount ):
        self.tp += min( goldCount, hypoCount )
        self.tn += self.k - max( goldCount, hypoCount )
        self.fn += max( 0, goldCount - hypoCount )
        self.fp += max( 0, hypoCount - goldCount )
        
    def write(self):
        print("tp: ", self.tp)
        print("tn: ", self.tn)
        print("fp: ", self.fp)
        print("fn: ", self.fn)
        print("precision: ", self.precision())
        print("recall: ", self.recall())
    
    
def demo():
    s1 = "00000010000000001000000"
    s2 = "00000001000000010000000"
    s3 = "00010000000000000001000"
    print("s1:", s1)
    print("s2:", s2)
    print("s3:", s3)


    print("")
    #print("windowdiff(s1, s1, 3) = ", windowdiff(s1, s1, 3)
    print("winpr(s1, s1, 3) = ", WinPR(s1, s1, 3).write())
    
    print("")
    #print("windowdiff(s1, s2, 3) = ", windowdiff(s1, s2, 3)
    print("winpr(s1, s2, 3) = ", WinPR(s1, s2, 3).write())
    
    print("")
    #print("windowdiff(s2, s3, 3) = ", windowdiff(s2, s3, 3)
    print("winpr(s2, s3, 3) = ", WinPR(s2, s3, 3).write())
    
    s4 = "0000"
    s5 = "1111"
    
    print("")
    print("s4:", s4)
    print("s5:", s5)
    #print("windowdiff(s4, s5, 2) = ", windowdiff(s4, s5, 2)
    print("winpr(s4, s5, 2) = ", WinPR(s4, s5, 2).write())
    
    # the following three examples are manually calculated at http://scaiano.com/martin/SegEval.html
    
    gold = "000010010"
    hypo = "001000010"
    print("")
    print("gold:", gold)
    print("hypo:", hypo)
    #print("windowdiff(gold, hypo, 2) = ", windowdiff(gold, hypo, 2)
    print("winpr(gold, hypo, 2) = ", WinPR(gold, hypo, 2).write())
    
    gold = "000010010"
    hypo = "001010011"
    print("")
    print("gold:", gold)
    print("hypo:", hypo)
    #print("windowdiff(gold, hypo, 2) = ", windowdiff(gold, hypo, 2)
    print("winpr(gold, hypo, 2) = ", WinPR(gold, hypo, 2).write())
    
    return

     