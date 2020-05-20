# Natural Language Toolkit: Windowdiff
#
# Copyright (C) 2001-2012 NLTK Project
# Author: Edward Loper <edloper@gradient.cis.upenn.edu>
#         Steven Bird <sb@csse.unimelb.edu.au>
# Updated By Martin Scaiano <martin@scaiano.com>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT

##########################################################################
# Windowdiff
# Pevzner, L., and Hearst, M., A Critique and Improvement of
#   an Evaluation Metric for Text Segmentation,
# Computational Linguistics,, 28 (1), March 2002, pp. 19-36
##########################################################################

#from WindowDiff import *
import sys

def windowsize(gold, boundary="1"):
    return len(gold) / gold.count(boundary)

def windowdiff(seg1, seg2, k, boundary="1"):
    """
    Compute the windowdiff score for a pair of segmentations.  A segmentation is any sequence
    over a vocabulary of two items (e.g. "0", "1"), where the specified boundary value is used
    to mark the edge of a segmentation.


    """

    if len(seg1) != len(seg2):
        sys.exit("Segmentations have unequal length")
    wd = 0.0

    
    for i in range(len(seg1)+1 - k):
        # or if abs( seg1[i:i+k+1].count(boundary) - seg2[i:i+k+1].count(boundary) ) > 0:
        if seg1[i:i+k].count(boundary) != seg2[i:i+k].count(boundary):
            wd += 1.0
        
    return wd / (len(seg1)+1 - k)



    

def demo():
    s1 = "00000010000000001000000"
    s2 = "00000001000000010000000"
    s3 = "00010000000000000001000"
    print("%s" % ( "s1:" + s1))
    print("%s" % ("s2:" + s2))
    print("%s" % ("s3:" + s3))

    print("%s" % (""))
    print("%s" % ("windowdiff(s1, s1, 3) = "  + str(windowdiff(s1, s1, 3))))
    #print "winpr(s1, s1, 3) = ", WinPR(s1, s1, 3).write()
    
    print("%s" % (""))
    print("%s" % ("windowdiff(s1, s2, 3) = " + str(windowdiff(s1, s2, 3))))
    #print "winpr(s1, s2, 3) = ", WinPR(s1, s2, 3).write()
    
    print("%s" % (""))
    print("%s" % ("windowdiff(s2, s3, 3) = " + str(windowdiff(s2, s3, 3))))
    #print "winpr(s2, s3, 3) = ", WinPR(s2, s3, 3).write()
    
    s4 = "0000"
    s5 = "1111"
    
    print("%s" % ( ""))
    print ("%s" % ("s4:" + s4))
    print ("%s" % ("s5:" + s5))
    print ("%s" % ("windowdiff(s4, s5, 2) = " + str(windowdiff(s4, s5, 2))))
    #print "winpr(s4, s5, 2) = ", WinPR(s4, s5, 2).write()
    
    # the following three examples are manually calculated at http://scaiano.com/martin/SegEval.html
    
    gold = "000010010"
    hypo = "001000010"

    print ("%s" % ("gold:" + gold))
    print ("%s" % ("hypo:" + hypo))
    print ("%s" % ("windowdiff(gold, hypo, 2) = " + str(windowdiff(gold, hypo, 2))))
    #print "winpr(gold, hypo, 2) = ", WinPR(gold, hypo, 2).write()
    
    gold = "000010010"
    hypo = "001010011"

    print ("%s" % (""))
    print ("%s" % ("gold:" + gold))
    print ("%s" % ("hypo:" + hypo))
    print ("%s" % ("windowdiff(gold, hypo, 2) = " + str(windowdiff(gold, hypo, 2))))
    #print "winpr(gold, hypo, 2) = ", WinPR(gold, hypo, 2).write()
    
    return

