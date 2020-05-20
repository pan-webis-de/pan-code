#!/usr/bin/env python

import os
import sys
import codecs
import random as rn



def pan13rnd(base_path, out_file=sys.stdout, encoding='utf-8-sig', scores_flg=False):

    if isinstance(out_file, str):
        #If a ouput filename string has been given then create the output file
        out_file = codecs.open(out_file, 'w', encoding)

    for pth_lst in sorted( os.walk(base_path).next()[1] ):

        score = round(rn.random(), 3)
        
        if score >= 0.5:
            y_bin = 'Y'
        else:
            y_bin = 'N'

        rnd_delete = rn.gammavariate(5.0, 1.0)
        rnd_dash = rn.gammavariate(9.0, 5.0)

        if scores_flg:
        
            if rnd_delete >= 2:
                out_file.write(pth_lst+' '+y_bin+' '+str(score)+'\n')
            else:
                if rnd_dash >=4:
                    out_file.write(pth_lst+' - '+str(score)+'\n')

        else:

            if rnd_delete >= 2:
                out_file.write(pth_lst+' '+y_bin+'\n')
            else:
                if rnd_dash >=4:
                    out_file.write(pth_lst+' -\n')

    out_file.close()



if __name__=='__main__':

    """NOTE: Scprit below will be executed only when this module will run as main script """

    #Use this three lines in case you want use any other function but pan13rnd
    #func = sys.argv[0]
    #func = func.replace('./','')
    #func = func.replace('.py','')

    #Directoy use only pan13rnd
    func = 'pan13rnd'

    scores = False
    encoding = None

    for flgs in sys.argv:
        
        if flgs == '-S':
            scores = True
        if len(flgs) > 2 and flgs[0] == '-':
            encoding = flgs[1::]
    
    if scores and encoding:

        if len(sys.argv) == 5:
            globals()[func](sys.argv[3], sys.argv[4], encoding, scores)
        elif len(sys.argv) == 4:
            globals()[func](sys.argv[3], sys.stdout, encoding, scores)
        else:
            print "Invalid number of arguments"

    elif scores:

        if len(sys.argv) == 4:
            globals()[func](sys.argv[2], sys.argv[3], scores_flg=scores)
        elif len(sys.argv) == 3:
            globals()[func](sys.argv[2], sys.stdout, scores_flg=scores)
        else:
            print "Invalid number of arguments"  

    elif encoding:

        if len(sys.argv) == 4:
            globals()[func](sys.argv[2], sys.argv[3], encoding)
        elif len(sys.argv) == 3:
            globals()[func](sys.argv[2], sys.stdout, encoding)
        else:
            print "Invalid number of arguments" 
    
    else:

        if len(sys.argv) == 3:
            globals()[func](sys.argv[1], sys.argv[2])
        elif len(sys.argv) == 2:
            globals()[func](sys.argv[1])
        else:
            print "Invalid number of arguments (" + str( len(sys.argv)-1 ) + ")"
    
