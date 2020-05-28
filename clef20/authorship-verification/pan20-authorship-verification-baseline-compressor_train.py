# -*- coding: utf-8 -*-

"""
 A baseline authorship verificaion method based on text compression. 
 Given two texts text1 and text2 it calculates the cross-entropy of text2 using the Prediction by Partical Matching (PPM) compression model of text1 and vice-versa.
 Then, the mean and absolute difference of the two cross-entropies are used to estimate a score in [0,1] indicating the probability the two texts are written by the same author.
 The prediction model is based on logistic regression and can be trained using a collection of training cases (pairs of texts by the same or different authors).
 Since the verification cases with a score exactly equal to 0.5 are considered to be left unanswered, a radius around this value is used to determine what range of scores will correspond to the predetermined value of 0.5.
 
 The method is based on the following paper:
     William J. Teahan and David J. Harper. Using compression-based language models for text categorization. In Language Modeling and Information Retrieval, pp. 141-165, 2003
 The current implementation is based on the code developed in the framework of a reproducibility study:
     M. Potthast, et al. Who Wrote the Web? Revisiting Influential Author Identification Research Applicable to Information Retrieval. In Proc. of the 38th European Conference on IR Research (ECIR 16), March 2016.
     https://github.com/pan-webis-de/teahan03
 Questions/comments: stamatatos@aegean.gr

 It can be applied to datasets of PAN-20 cross-domain authorship verification task.
 See details here: http://pan.webis.de/clef20/pan20-web/author-identification.html
 Dependencies:
 - Python 2.7 or 3.6 (we recommend the Anaconda Python distribution)

 Usage from command line: 
    > python pan20-authorship-verification-baseline-compressor.py -i EVALUATION-FILE -o OUTPUT-DIRECTORY [-m MODEL-FILE]
 EVALUATION-DIRECTORY (str) is the full path name to a PAN-20 collection of verification cases (each case is a pair of texts)
 OUTPUT-DIRECTORY (str) is an existing folder where the predictions are saved in the PAN-20 format
 Optional parameter:
     MODEL-FILE (str) is the full path name to the trained model (default=model_small.joblib, a model already trained on the small training dataset released by PAN-20 using logistic regression with PPM order = 5)
	 RADIUS (float) is the radius around the threshold 0.5 to leave verification cases unanswered (dedault = 0.05). All cases with a value in [0.5-RADIUS, 0.5+RADIUS] are left unanswered.
 
 Example:
     > python pan20-authorship-verification-baseline-compressor.py -i "mydata/pan20-authorship-verification-test-corpus.jsonl" -o "mydata/pan20-answers" -m "mydata/model_small.joblib"

 Additional functions (train_data and train_model) are provided to prepare training data and train a new model.
 
 Supplementary files:
	data-small.txt: training data extracted from the small dataset provided by PAN-20 authorship verification task
    model.joblib: trained model using logistic regression, PPM order=5, using data of data-small.txt
"""

from __future__ import print_function
from math import log
import os
import json
import time
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from joblib import dump, load
import decimal
import numpy as np
import shutil


class Model(object):
    # cnt - count of characters read
    # modelOrder - order of the model
    # orders - List of Order-Objects
    # alphSize - size of the alphabet
    def __init__(self, order, alphSize):
        self.cnt = 0
        self.alphSize = alphSize
        self.modelOrder = order
        self.orders = []
        for i in range(order + 1):
            self.orders.append(Order(i))

    # print the model
    # TODO: Output becomes too long, reordering on the screen has to be made
    def printModel(self):
        s = "Total characters read: " + str(self.cnt) + "\n"
        for i in range(self.modelOrder + 1):
            self.printOrder(i)

    # print a specific order of the model
    # TODO: Output becomes too long, reordering on the screen has to be made
    def printOrder(self, n):
        o = self.orders[n]
        s = "Order " + str(n) + ": (" + str(o.cnt) + ")\n"
        for cont in o.contexts:
            if(n > 0):
                s += "  '" + cont + "': (" + str(o.contexts[cont].cnt) + ")\n"
            for char in o.contexts[cont].chars:
                s += "     '" + char + "': " + \
                    str(o.contexts[cont].chars[char]) + "\n"
        s += "\n"
        print(s)

    # updates the model with a character c in context cont
    def update(self, c, cont):
        if len(cont) > self.modelOrder:
            raise NameError("Context is longer than model order!")

        order = self.orders[len(cont)]
        if not order.hasContext(cont):
            order.addContext(cont)
        context = order.contexts[cont]
        if not context.hasChar(c):
            context.addChar(c)
        context.incCharCount(c)
        order.cnt += 1
        if (order.n > 0):
            self.update(c, cont[1:])
        else:
            self.cnt += 1

    # updates the model with a string
    def read(self, s):
        if (len(s) == 0):
            return
        for i in range(len(s)):
            cont = ""
            if (i != 0 and i - self.modelOrder <= 0):
                cont = s[0:i]
            else:
                cont = s[i - self.modelOrder:i]
            self.update(s[i], cont)

    # return the models probability of character c in content cont
    def p(self, c, cont):
        if len(cont) > self.modelOrder:
            raise NameError("Context is longer than order!")

        order = self.orders[len(cont)]
        if not order.hasContext(cont):
            if (order.n == 0):
                return 1.0 / self.alphSize
            return self.p(c, cont[1:])

        context = order.contexts[cont]
        if not context.hasChar(c):
            if (order.n == 0):
                return 1.0 / self.alphSize
            return self.p(c, cont[1:])
        return float(context.getCharCount(c)) / context.cnt

    # merge this model with another model m, esentially the values for every
    # character in every context are added
    def merge(self, m):
        if self.modelOrder != m.modelOrder:
            raise NameError("Models must have the same order to be merged")
        if self.alphSize != m.alphSize:
            raise NameError("Models must have the same alphabet to be merged")
        self.cnt += m.cnt
        for i in range(self.modelOrder + 1):
            self.orders[i].merge(m.orders[i])

    # make this model the negation of another model m, presuming that this
    # model was made my merging all models
    def negate(self, m):
        if self.modelOrder != m.modelOrder or self.alphSize != m.alphSize or self.cnt < m.cnt:
            raise NameError("Model does not contain the Model to be negated")
        self.cnt -= m.cnt
        for i in range(self.modelOrder + 1):
            self.orders[i].negate(m.orders[i])


class Order(object):
    # n - whicht order
    # cnt - character count of this order
    # contexts - Dictionary of contexts in this order
    def __init__(self, n):
        self.n = n
        self.cnt = 0
        self.contexts = {}

    def hasContext(self, context):
        return context in self.contexts

    def addContext(self, context):
        self.contexts[context] = Context()

    def merge(self, o):
        self.cnt += o.cnt
        for c in o.contexts:
            if not self.hasContext(c):
                self.contexts[c] = o.contexts[c]
            else:
                self.contexts[c].merge(o.contexts[c])

    def negate(self, o):
        if self.cnt < o.cnt:
            raise NameError(
                "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
        self.cnt -= o.cnt
        for c in o.contexts:
            if not self.hasContext(c):
                raise NameError(
                    "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
            else:
                self.contexts[c].negate(o.contexts[c])
        empty = [c for c in self.contexts if len(self.contexts[c].chars) == 0]
        for c in empty:
            del self.contexts[c]


class Context(object):
    # chars - Dictionary containing character counts of the given context
    # cnt - character count of this context
    def __init__(self):
        self.chars = {}
        self.cnt = 0

    def hasChar(self, c):
        return c in self.chars

    def addChar(self, c):
        self.chars[c] = 0

    def incCharCount(self, c):
        self.cnt += 1
        self.chars[c] += 1

    def getCharCount(self, c):
        return self.chars[c]

    def merge(self, cont):
        self.cnt += cont.cnt
        for c in cont.chars:
            if not self.hasChar(c):
                self.chars[c] = cont.chars[c]
            else:
                self.chars[c] += cont.chars[c]

    def negate(self, cont):
        if self.cnt < cont.cnt:
            raise NameError(
                "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
        self.cnt -= cont.cnt
        for c in cont.chars:
            if (not self.hasChar(c)) or (self.chars[c] < cont.chars[c]):
                raise NameError(
                    "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
            else:
                self.chars[c] -= cont.chars[c]
        empty = [c for c in self.chars if self.chars[c] == 0]
        for c in empty:
            del self.chars[c]

# calculates the cross-entropy of the string 's' using model 'm'
def h(m, s):
    n = len(s)
    h = 0
    for i in range(n):
        if i == 0:
            context = ""
        elif i <= m.modelOrder:
            context = s[0:i]
        else:
            context = s[i - m.modelOrder:i]
        h -= log(m.p(s[i], context), 2)
    return h / n

# Calculates the cross-entropy of text2 using the model of text1 and vice-versa
# Returns the mean and the absolute difference of the two cross-entropies
def distance(text1,text2,ppm_order=5):
    mod1 = Model(ppm_order, 256)
    mod1.read(text1)
    d1=h(mod1, text2)
    mod2 = Model(ppm_order, 256)
    mod2.read(text2)
    d2=h(mod2, text1)
    return [round((d1+d2)/2.0,4),round(abs(d1-d2),4)]

# Prepares training data 
# For each verification case it calculates the mean and absolute differences of cross-entropies
def train_data(train_file,truth_file,out_file,ppm_order=5):
    with open(truth_file,'r') as tfp:
        labels=[]
        for line in tfp:
            labels.append(json.loads(line))
    with open(train_file,'r') as fp:
        data=[]
        tr_labels=[]
        tr_data={}
        for i,line in enumerate(fp):
            X=json.loads(line)
            true_label=[x for x in labels if x["id"] == X["id"] ][0]
            D=distance(X['pair'][0],X['pair'][1],ppm_order)
            if true_label["same"]==True:
                tl=1
            else: tl=0
            data.append(D)
            tr_labels.append(tl)
            print(i,X['id'],D[0],true_label["same"])

        # Saves training data
        tr_data["data"]=data
        tr_data["labels"]=tr_labels
        with open(out_file, 'w') as outf:
            json.dump(tr_data, outf)


# Trains the logistic regression model
def train_model(train_data_file,output_model_file):
    with open(train_data_file) as fp:
        D1=json.load(fp)
        X_train = D1['data']
        y_train = D1['labels']
    X_train = np.stack(X_train, axis=0)
    y_train = np.array(y_train)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    dump(logreg, output_model_file)


# Applies the model to evaluation data
# Produces an output file (answers.jsonl) with predictions
def apply_model(eval_data_file,output_folder,model_file,radius):
    start_time = time.time()
    model = load(model_file) 
    answers=[]
    with open(eval_data_file,'r') as fp:
        for i,line in enumerate(fp):
            X=json.loads(line)
            D=distance(X['pair'][0],X['pair'][1],ppm_order=5)
            pred = model.predict_proba(np.array(D).reshape((1, 2)))
            # All values around 0.5 are transformed to 0.5
            if pred[0,1] >= 0.5 - radius and pred[0,1] <= 0.5 + radius:
                pred[0,1] = 0.5
            print(i+1,X['id'],round(pred[0,1],3))
            answers.append({'id': X['id'],'value': round(pred[0,1],3)})
    with open(output_folder+os.sep+'answers.jsonl', 'w') as outfile:
        for ans in answers:
            json.dump(ans, outfile)
            outfile.write('\n')
    print('elapsed time:', time.time() - start_time)


def main():
    parser = argparse.ArgumentParser(description='PAN-20 Cross-domain Authorship Verification task: Baseline Compressor')
    parser.add_argument('-input_pairs', type=str, required=True,
                        help='Path to the jsonl-file with the input pairs')
    parser.add_argument('-input_truth', type=str, required=True,
                        help='Path to the ground truth-file for the input pairs')
    parser.add_argument('-features', type=str, required=True,
                        help='json-file for features extracted from input pairs')
    parser.add_argument('-test_pairs', type=str, required=True,
                        help='Path to the jsonl-file with the test pairs')
    parser.add_argument('-output', type=str, required=True,
                        help='Path to the output folder for the predictions.\
                                 (Will be overwritten if it exist already.)')
    parser.add_argument('-model', type=str, default='model_small.joblib', help='Full path name to the model file')
    parser.add_argument('-r', type=float, default=0.05, help='Radius around 0.5 to leave verification cases unanswered')
    args = parser.parse_args()

    try:
        shutil.rmtree(args.output)
    except FileNotFoundError:
        pass
    os.mkdir(args.output)

    # prepare training data
    train_data(args.input_pairs, args.input_truth, args.features, ppm_order=5)

    # train model
    train_model(args.features, args.model)

    # apply model
    apply_model(args.test_pairs, args.output, args.model, args.r)


if __name__ == '__main__':
    main()