# -*- coding: utf-8 -*-

"""
 A baseline authorship attribution method
 based on a character n-gram representation
 and a linear SVM classifier
 for Python 2.7
 Questions/comments: stamatatos@aegean.gr

 It can be applied to datasets of PAN-18 cross-domain authorship attribution task
 See details here: http://pan.webis.de/clef18/pan18-web/author-identification.html
 Dependencies:
 - Python 2.7 or 3.6 (we recommend the Anaconda Python distribution)
 - scikit-learn

 Usage from command line: 
    > python pan18-cdaa-baseline.py -i EVALUATION-DIRECTORY -o OUTPUT-DIRECTORY [-n N-GRAM-ORDER] [-ft FREQUENCY-THRESHOLD] [-c CLASSIFIER]
 EVALUATION-DIRECTORY (str) is the main folder of a PAN-18 collection of attribution problems
 OUTPUT-DIRECTORY (str) is an existing folder where the predictions are saved in the PAN-18 format
 Optional parameters of the model:
   N-GRAM-ORDER (int) is the length of character n-grams (default=3)
   FREQUENCY-THRESHOLD (int) is the curoff threshold used to filter out rare n-grams (default=5)
   CLASSIFIER (str) is either 'OneVsOne' or 'OneVsRest' version of SVM (default=OneVsRest)
   
 Example:
     > python pan18-cdaa-baseline.py -i "mydata/pan18-cdaa-development-corpus" -o "mydata/pan18-answers"
"""

from __future__ import print_function
import os
import glob
import json
import argparse
import time
import codecs
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

def represent_text(text,n):
    # Extracts all character 'n'-grams from  a 'text'
    if n>0:
        tokens = [text[i:i+n] for i in range(len(text)-n+1)]
    frequency = defaultdict(int)
    for token in tokens:
        frequency[token] += 1
    return frequency

def read_files(path,label):
    # Reads all text files located in the 'path' and assigns them to 'label' class
    files = glob.glob(path+os.sep+label+os.sep+'*.txt')
    texts=[]
    for i,v in enumerate(files):
        f=codecs.open(v,'r',encoding='utf-8')
        texts.append((f.read(),label))
        f.close()
    return texts

def extract_vocabulary(texts,n,ft):
    # Extracts all characer 'n'-grams occurring at least 'ft' times in a set of 'texts'
    occurrences=defaultdict(int)
    for (text,label) in texts:
        text_occurrences=represent_text(text,n)
        for ngram in text_occurrences:
            if ngram in occurrences:
                occurrences[ngram]+=text_occurrences[ngram]
            else:
                occurrences[ngram]=text_occurrences[ngram]
    vocabulary=[]
    for i in occurrences.keys():
        if occurrences[i]>=ft:
            vocabulary.append(i)
    return vocabulary

def baseline(path,outpath,n=3,ft=5,classifier='OneVsRest'):
    start_time = time.time()
    # Reading information about the collection
    infocollection = path+os.sep+'collection-info.json'
    problems = []
    language = []
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
            language.append(attrib['language'])
    
    for index,problem in enumerate(problems):
        print(problem)
        # Reading information about the problem
        infoproblem = path+os.sep+problem+os.sep+'problem-info.json'
        candidates = []
        with open(infoproblem, 'r') as f:
            fj = json.load(f)
            unk_folder = fj['unknown-folder']
            for attrib in fj['candidate-authors']:
                candidates.append(attrib['author-name'])
        # Building training set
        train_docs=[]
        for candidate in candidates:
            train_docs.extend(read_files(path+os.sep+problem,candidate))
        train_texts = [text for i,(text,label) in enumerate(train_docs)]
        train_labels = [label for i,(text,label) in enumerate(train_docs)]
        vocabulary = extract_vocabulary(train_docs,n,ft)
        vectorizer = CountVectorizer(analyzer='char',ngram_range=(n,n),lowercase=False,vocabulary=vocabulary)
        train_data = vectorizer.fit_transform(train_texts)
        train_data = train_data.astype(float)
        for i,v in enumerate(train_texts):
            train_data[i]=train_data[i]/len(train_texts[i])
        print('\t', 'language: ', language[index])
        print('\t', len(candidates), 'candidate authors')
        print('\t', len(train_texts), 'known texts')
        print('\t', 'vocabulary size:', len(vocabulary))
        # Building test set
        test_docs=read_files(path+os.sep+problem,unk_folder)
        test_texts = [text for i,(text,label) in enumerate(test_docs)]
        test_data = vectorizer.transform(test_texts)
        test_data = test_data.astype(float)
        for i,v in enumerate(test_texts):
            test_data[i]=test_data[i]/len(test_texts[i])
        print('\t', len(test_texts), 'unknown texts')
        # Applying SVM
        max_abs_scaler = preprocessing.MaxAbsScaler()
        scaled_train_data = max_abs_scaler.fit_transform(train_data)
        scaled_test_data = max_abs_scaler.transform(test_data)
        if classifier=='OneVsOne':
            clf=OneVsOneClassifier(LinearSVC(C=1)).fit(scaled_train_data, train_labels)
        else:
            clf=OneVsRestClassifier(LinearSVC(C=1)).fit(scaled_train_data, train_labels)
        predictions=clf.predict(scaled_test_data)
        # Writing output file
        out_data=[]
        unk_filelist = glob.glob(path+os.sep+problem+os.sep+unk_folder+os.sep+'*.txt')
        pathlen=len(path+os.sep+problem+os.sep+unk_folder+os.sep)
        for i,v in enumerate(predictions):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        with open(outpath+os.sep+'answers-'+problem+'.json', 'w') as f:
            json.dump(out_data, f, indent=4)
        print('\t', 'answers saved to file','answers-'+problem+'.json')
    print('elapsed time:', time.time() - start_time)

def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='PAN-18 Baseline Authorship Attribution Method')
    parser.add_argument('-i', type=str, help='Path to the main folder of a collection of attribution problems')
    parser.add_argument('-o', type=str, help='Path to an output folder')
    parser.add_argument('-n', type=int, default=3, help='n-gram order (default=3)')
    parser.add_argument('-ft', type=int, default=5, help='frequency threshold (default=5)')
    parser.add_argument('-c', type=str, default='OneVsRest', help='OneVsRest or OneVsOne (default=OneVsRest)')
    args = parser.parse_args()
    if not args.i:
        print('ERROR: The input folder is required')
        parser.exit(1)
    if not args.o:
        print('ERROR: The output folder is required')
        parser.exit(1)
    
    baseline(args.i, args.o, args.n, args.ft, args.c)

if __name__ == '__main__':
    main()