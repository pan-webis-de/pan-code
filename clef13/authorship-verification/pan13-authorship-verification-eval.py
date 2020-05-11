#!/usr/bin/env python

import os
import sys
import codecs



def roc_curve(truth_d, scr_d):
    """Receiver Operating Characteristic (ROC)"""

    pos_sum = sum( [y=='Y' for y in truth_d.values()] )
    neg_sum = len(truth_d) - pos_sum

    if pos_sum == 0:
        pos_sum = 1
    if neg_sum == 0:
        neg_sum = 1

    pos_cnt = 0
    neg_cnt = 0

    scr_ybin_lst = list()
    for key, bin_val in truth_d.items():

        if key in scr_d:
            bin_int = 1 if bin_val == 'Y' else 0
            scr_ybin_lst.append( (scr_d[key], bin_int) )
        else:
            #add no provided aswares with negative value in Ground-Truth file
            neg_cnt += 1 if bin_val == 'N' else 0

    scr_ybin_srd_lst = sorted(scr_ybin_lst, key=lambda scr_ybin_lst: scr_ybin_lst[0], reverse=True)

    tp_rate = list()
    fp_rate = list()    

    last_scr = -1
    #append [0, neg_cnt / float(neg_sum)]
    #tp_rate.append( 0.0 )
    #fp_rate.append( neg_cnt / float(neg_sum) )

    for i, (scr, y) in enumerate(scr_ybin_srd_lst):

        if scr != last_scr:
            tp_rate.append( pos_cnt / float(pos_sum) )
            fp_rate.append( neg_cnt / float(neg_sum) )
            last_scr = scr

        if int(y) == 1:
            pos_cnt += 1
        elif int(y) == 0 or int(y) == -1:
            neg_cnt += 1
        else:
            raise Exception("Incompatible input: -1, 0, 1 values supported for Y binary")
    
    #Append last point if not already
    tp_rate.append( pos_cnt / float(pos_sum) )
    fp_rate.append( neg_cnt / float(neg_sum) )

    norm = 1.0 #float(pos_sum*neg_sum)

    return tp_rate, fp_rate, norm 



def auc(roc_curve):

    if isinstance(roc_curve, str):
        roc_curve = list(eval(roc_curve))

    y = roc_curve[0]
    x = roc_curve[1]

    dx = [float(x1 - x2) for x1, x2 in zip(x[1::],x)]
    dx0 = [0]
    dx0.extend(dx)

    h = [(y1 + y2)/2.0 for y1, y2 in zip(y[1::],y)]
    h0 = [0]
    h0.extend(h)

    return sum( [dx*y for dx, y in zip(dx0, h0)] ) / roc_curve[2]



def pan13truth(truth_file, encoding='utf-8-sig'):

    with codecs.open(truth_file, 'r', encoding) as fin:

        truth_d = dict()

        last_class = None

        for line in fin:

            fields_lst = line.split()

            if fields_lst:

                if last_class != fields_lst[0][0:2]:
                    last_class = fields_lst[0][0:2]
                    truth_d[last_class] = dict()
                    last_row_id_lst = list()

                if fields_lst[1] == 'Y' or fields_lst[1] == 'N':

                    if fields_lst[0] in last_row_id_lst:
                        raise Exception('Unique answer per row violation in True/Answars file - Row ID: '+fields_lst[0]+' encountered twice')
                    else:
                        last_row_id_lst.append( fields_lst[0] )

                    truth_d[ fields_lst[0][0:2] ][ fields_lst[0] ] = fields_lst[1]

                else:
                    raise Exception("Answer violation in True/Answers file - 'Y' or 'N' expected")
 
    return truth_d



def pan13performance(truth_d, in_file, out_file=sys.stdout, encoding='utf-8-sig', detail=None):

    if isinstance(out_file, str):
        #If a ouput filename string has been given then create the output file
        out_file = codecs.open(out_file, 'w', encoding)

    class_RR_d = dict()
    class_RV_lst = dict()

    roc_flag = False
    
    err_msg1 = "Invalid Results File: Missing Scores in resaults sequence\n"
    err_msg2 = "Invalid Results File (with scores): 'Y', 'N' and '-' are the only valid binary values\n"

    #Start processing the input file
    with codecs.open(in_file, 'r', encoding) as fin:
        
        last_class_lst = list()
        res_row_range = dict()
        last_cls_row = dict()

        for line in fin:

            fields_lst = line.split()

            if fields_lst:
            
                if  fields_lst[0][0:2] not in last_class_lst:

                    if fields_lst[0][0:2] not in truth_d:
                        err_msg0 = 'Unknown class tag '
                        err_msg0 += fields_lst[0].encode(encoding)
                        err_msg0 += '\n'
                        raise Exception(err_msg0)
                                            
                    last_class = fields_lst[0][0:2]
                    last_class_lst.append( last_class )
                    
                    last_cls_row[last_class] = list()

                    rel_doc_num = len(truth_d[last_class])

                    class_RR_d[last_class] = [0.0, 0.0, float(rel_doc_num), 0.0, 0.0]

                    if len(fields_lst) == 3:
                        class_RV_lst[last_class] = dict()
                        roc_flag = True

                    elif roc_flag:
                        out_file.write( err_msg1 )
                        raise Exception( err_msg2 )

                    res_row_range[last_class] = truth_d[last_class].keys()
                
                if fields_lst[0] not in last_cls_row[ fields_lst[0][0:2] ] and fields_lst[0] in res_row_range[ fields_lst[0][0:2] ]:
                    
                    if truth_d[ fields_lst[0][0:2] ][ fields_lst[0] ] == fields_lst[1]:
                        class_RR_d[ fields_lst[0][0:2] ][0] += 1.0

                    if fields_lst[1] == 'Y' or fields_lst[1] == 'N':
                        class_RR_d[ fields_lst[0][0:2] ][1] += 1.0
                    elif fields_lst[1] != '-':
                        out_file.write( err_msg2 )
                        raise Exception(err_msg2)

                    if len(fields_lst) == 3:

                        if fields_lst[1] != 'Y' and fields_lst[1] != 'N' and fields_lst[1] != '-':
                            out_file.write( err_msg2 )
                            raise Exception(err_msg2)

                        class_RV_lst[ fields_lst[0][0:2] ][fields_lst[0]] = float( fields_lst[2] )

                    elif roc_flag:
                        out_file.write( err_msg1 )
                        raise Exception( err_msg1 )

                    last_cls_row[ fields_lst[0][0:2] ].append( fields_lst[0] )

    #Calculate P, R and Total P and R
    total_correct_num = 0
    total_retrieved_num = 0 
    total_problems_num = 0
    for vl in class_RR_d.values():

        if vl[1] != 0.0 and vl[2] != 0.0:
            #Calculate Precision
            vl[3] = round(vl[0] / vl[1], 3)
            #Calculate Recall
            vl[4] = round(vl[0] / vl[2], 3)
        else:
            vl[4] = vl[3] = 0.0

        total_correct_num += vl[0]
        total_retrieved_num += vl[1] 
    
    total_problems_num = sum( [len(x) for x in truth_d.values()] )

    #When Zero Precision then Zero Recall (and vice versa)
    if total_retrieved_num != 0.0 and total_problems_num != 0.0:
        Total_P = round( total_correct_num / float(total_retrieved_num), 3)
        Total_R = round( total_correct_num / float(total_problems_num), 3 )
    else: 
        Total_P = Total_R = 0.0

    if roc_flag:
        #Return ROC Curves instead of scores/bin list 
        total_val = dict()
        for key, val in class_RV_lst.items():
            #
            total_val.update( val )
            #
            roc = roc_curve(truth_d[key], class_RV_lst[key])
            #
            rn_roc_0 = [ round(v, 3) for v in roc[0] ]
            rn_roc_1 = [ round(v, 3) for v in roc[1] ]
            class_RV_lst[key] = (rn_roc_0, rn_roc_1, [ round( auc(roc), 3) ])

        total_trth = dict()
        for key in truth_d:
            total_trth.update( truth_d[key] )   

        total_roc = roc_curve(total_trth, total_val)
        total_rn_roc_tpr = [ round(v, 3) for v in total_roc[0] ]
        total_rn_roc_fpr = [ round(v, 3) for v in total_roc[1] ]
        total_auc = round( auc(total_roc), 3 )

    out_str = "\n\n"
    
    out_str += '{"Recall":"'+str(Total_R)+'"}\n'
    out_str += '{"Precision":"'+str(Total_P)+'"}\n'
    if detail:
        out_str += '{"Correct answers":"'+str(total_correct_num)+'"}\n'
        out_str += '{"Given answers":"'+str(total_retrieved_num)+'"}\n'
        out_str += '{"Problems":"'+str(total_problems_num)+'"}\n'

    if roc_flag:
        out_str += '{"AUC":"'+str(total_auc)+'"}\n'
        if detail:
            out_str += '{"TPR":"'+str(total_rn_roc_tpr)+'"}\n'
            out_str += '{"FPR":"'+str(total_rn_roc_fpr)+'"}\n'
            

    for key, vl in class_RR_d.items():
        out_str += '{"'+key+'-Recall":"'+str(vl[4])+'"}\n'
        out_str += '{"'+key+'-Precision":"'+str(vl[3])+'"}\n'
        if detail:
            out_str += '{"'+key+'-Correct answers":"'+str(vl[0])+'"}\n'
            out_str += '{"'+key+'-Given answers":"'+str(vl[1])+'"}\n'
            out_str += '{"'+key+'-Problems":"'+str(vl[2])+'"}\n'
        if roc_flag:
            out_str +='{"'+key+'-AUC":"'+str(class_RV_lst[ key ][2][0])+'"}\n'
            if detail:
                out_str += '{"'+key+'-TPR":"'+str(class_RV_lst[ key ][0])+'"}\n'
                out_str += '{"'+key+'-FPR":"'+str(class_RV_lst[ key ][1])+'"}\n'

    out_str += "\n"

    out_file.write(out_str)
    
    out_file.close()
    
    return class_RR_d, class_RV_lst



if __name__=='__main__':

    """NOTE: Scprit below will be executed only when this module will run as main script """

    #Use this three lines in case you want use any other function but pan13performace
    #func = sys.argv[0]
    #func = func.replace('./','')
    #func = func.replace('.py','')

    #Directoy use only pan13performace
    func = 'pan13performance'

    detail = False
    encoding = None

    for flgs in sys.argv:
        
        if flgs == '-d':
            detail = True
        if len(flgs) > 2 and flgs[0] == '-':
            encoding = flgs[1::]
    
    if detail and encoding:

        if len(sys.argv) == 6:

            truth_d = pan13truth(sys.argv[3])

            globals()[func](truth_d, sys.argv[4], sys.argv[5], encoding, detail)

        elif len(sys.argv) == 5:

            truth_d = pan13truth(sys.argv[3])

            globals()[func](truth_d, sys.argv[4], sys.stdout, encoding, detail)

        else:
            print "Invalid number of arguments"

    elif detail:

        if len(sys.argv) == 5:

            truth_d = pan13truth(sys.argv[2])

            globals()[func](truth_d, sys.argv[3], sys.argv[4], detail=detail)

        elif len(sys.argv) == 4:

            truth_d = pan13truth(sys.argv[2])

            globals()[func](truth_d, sys.argv[3], sys.stdout, detail=detail)

        else:
            print "Invalid number of arguments"  

    elif encoding:

        if len(sys.argv) == 5:

            truth_d = pan13truth(sys.argv[2])

            globals()[func](truth_d, sys.argv[3], sys.argv[4], encoding)

        elif len(sys.argv) == 4:

            truth_d = pan13truth(sys.argv[2])

            globals()[func](truth_d, sys.argv[3], sys.stdout, encoding)

        else:
            print "Invalid number of arguments" 
    
    else:

        if len(sys.argv) == 4:

            truth_d = pan13truth(sys.argv[1])

            globals()[func](truth_d, sys.argv[2], sys.argv[3])

        elif len(sys.argv) == 3:

            truth_d = pan13truth(sys.argv[1])

            globals()[func](truth_d, sys.argv[2])

        else:
            print "Invalid number of arguments (" + str( len(sys.argv)-1 ) + ")"
