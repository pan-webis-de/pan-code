#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Evaluation script for the Cross-Domain Authorship Attribution task @PAN2018.
We use the F1 metric (macro-average) as implemented in scikit-learn:
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
We include the following ad hoc rules:
- If authors are predicted which were not seen during training,
  these predictions will count as false predictions ('<UNK>' class)
  and they will negatively effect performance.
- If texts are left unattributed they will assigned to the ('<UNK>'
  class) and they will negatively effect performance.
- The <UNK> class is excluded from the macro-average across classes.
- If multiple test attributions are given for a single unknown document,
  only the first one will be taken into consideration.

Dependencies:
- Python 2.7 or 3.6 (we recommend the Anaconda Python distribution)
- scikit-learn
- matplotlib

Usage from the command line:
>>> python pan18-cdaa-evaluator-single.py -g GROUND-TRUTH-FILE -p PREDICTIONS-FILE [-c CONFUSION-MATRIX-FILE]
where
    GROUND-TRUTH-FILE is the path to the  (json) ground truth file of an attribution problem
    PREDICTIONS-FILE is the path to the  (json) predictions file of an attribution problem
    CONFUSION-MATRIX-FILE is the path to the file where the image of the confusion matrix will be saved (optionally)

Example: 
>>> python pan18-cdaa-evaluator-single.py -g "/mydata/pan18-cdaa-development-corpus/problem00001/ground-truth.json" -p "/mydata/pan18-answers/answers-problem00001.json"
    
# References:
@article{scikit-learn,
 title={Scikit-learn: Machine Learning in {P}ython},
 author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal={Journal of Machine Learning Research},
 volume={12},
 pages={2825--2830},
 year={2011}
}
"""
import argparse
import os
import json
import warnings
from itertools import product

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("seaborn-deep")

import numpy as np

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def macro_f1(gt, pred, cm_path=None):
    """Compute macro-averaged F1-scores according the ad hoc
    rules discussed at the top of this file.
    Parameters
    ----------
    gt : dict
        Ground truth, where keys indicate text file names
        (e.g. `unknown00002.txt`), and values represent
        author labels (e.g. `candidate00003`)
    pred : dict
        Predicted attribution, where keys indicate text file names
        (e.g. `unknown00002.txt`), and values represent
        author labels (e.g. `candidate00003`)
    cm_path : str (default: None)
        Path to where to write the confusion matrix image. If `None`,
        no confusion matrix is created.
    Returns
    -------
    f1 : float
        Macro-averaged F1-score
    """

    actual_authors = list(gt.values())
    encoder = LabelEncoder().fit(['<UNK>'] + actual_authors)

    text_ids, gold_authors, silver_authors = [], [], []
    for text_id in sorted(gt):
        text_ids.append(text_id)
        gold_authors.append(gt[text_id])
        try:
            silver_authors.append(pred[text_id])
        except KeyError:
            # missing attributions get <UNK>:
            silver_authors.append('<UNK>')

    assert len(text_ids) == len(gold_authors)
    assert len(text_ids) == len(silver_authors)

    # replace non-existent silver authors with '<UNK>':
    silver_authors = [a if a in encoder.classes_ else '<UNK>' 
                      for a in silver_authors]

    gold_author_ints = encoder.transform(gold_authors)
    silver_author_ints = encoder.transform(silver_authors)

    # get F1 for individual classes (and suppress warnings):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f1 = f1_score(gold_author_ints,
                  silver_author_ints,
                  labels=list(set(gold_author_ints)),
                  average='macro')

    # save the confusion matrix
    if cm_path:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cm = confusion_matrix(gold_author_ints, silver_author_ints)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            np.set_printoptions(precision=2)
    
            plt.figure(figsize=(20, 20))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.tick_params(labelsize=12)
            plt.title('Confusion matrix')
            plt.colorbar()
            if len(encoder.classes_)==len(cm):
                tick_marks = np.arange(len(encoder.classes_))
                plt.xticks(tick_marks, encoder.classes_, rotation=90)
                plt.yticks(tick_marks, encoder.classes_)
            else:
                tick_marks = np.arange(len(encoder.classes_[1:]))
                plt.xticks(tick_marks, encoder.classes_[1:], rotation=90)
                plt.yticks(tick_marks, encoder.classes_[1:])
    
            thresh = cm.max() / 2.
            for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, round(cm[i, j], 2),
                         horizontalalignment='center',
                         color='white' if cm[i, j] > thresh else 'black')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.plot()
            plt.savefig(cm_path)

    return f1

def main():
    logging.info('>>> Evaluation Cross-Domain Authorship Attribution @PAN2018 <<<')
    parser = argparse.ArgumentParser(description='Evaluation script AA@PAN2018')
    parser.add_argument('-g', type=str,
                        help='Path to ground truth file (json formatted)')
    parser.add_argument('-p', type=str,
                        help='Path to system predictions (json formatted)')
    parser.add_argument('-c', type=str, 
                        help='Path to plot confusion matrix (optional)')
    args = parser.parse_args()
    if not args.g:
        print('ERROR: The ground truth file is required')
        parser.exit(1)
    if not args.p:
        print('ERROR: The predictions file is required')
        parser.exit(1)
    logging.info(args)

    gt = {}
    with open(args.g, 'r') as f:
        for attrib in json.load(f)['ground_truth']:
            gt[attrib['unknown-text']] = attrib['true-author']

    pred = {}
    with open(args.p, 'r') as f:
        for attrib in json.load(f):
            if attrib['unknown-text'] not in pred:
                pred[attrib['unknown-text']] = attrib['predicted-author']

    f1 = macro_f1(gt=gt, pred=pred, cm_path=args.c)
    logging.info('MACRO-AVERAGED F1: %f',f1)
	
    logging.info('>>> Evaluation done <<<')

if __name__ == '__main__':
    main()