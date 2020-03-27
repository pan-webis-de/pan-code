#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Evaluation script for the Cross-Domain Authorship Verification task @PAN2020.

## Measures
The following evaluation measures are provided:
    - F1-score [Pedregosa et al. 2011]
    - Area-Under-the-Curve [Pedregosa et al. 2011]
    - c@1 [Peñas and Rodrigo 2011; Stamatatos 2014]
    - f_05_u_score [Bevendorff et al. 2019]

## Formats
The script requires two folders, one for the ground truth (gold standard)
and one for the system predictions. Each folder should contain at least one,
but potentially more files, that are formatted using the `jsonl`-convention,
whereby each line should contain a valid json-string: e.g.

``` json
    {"problem_id": "1", "value": 0.123}
    {"problem_id": "2", "value": 0.5}
    {"problem_id": "3", "value": 0.888}
```

Only files will be considered that:
- have the `.jsonl` extension
- are properly encoded as UTF-8.

Please note:
    * For the F1-score, all scores are will binarized using
      the conventional thresholds:
        * score < 0.5 -> 0
        * score > 0.5 -> 0
    * A score of *exactly* 0.5, will be considered a non-decision.
    * All answers which are present in the ground truth, but which
      are *not* provided by the system, will automatically be set to 0.5.

## Dependencies:
- Python 3.6+ (we recommend the Anaconda Python distribution)
- scikit-learn

## Usage

From the command line:

>>> python pan20-verif-evaluator.py -i COLLECTION -a ANSWERS -o OUTPUT

where
    COLLECTION is the path to the main folder of the evaluation collection
    ANSWERS is the path to the answers folder of a submitted method
    OUTPUT is the path to the folder where the results of the evaluation will be saved

Example: 

>>> python pan20-verif-evaluator.py -i "/mydata/pan20-verif-development-corpus" -a "/mydata/pan20-answers" -o "/mydata/pan20-evaluation"

## References
- E. Stamatatos, et al. Overview of the Author Identification
  Task at PAN 2014. CLEF Working Notes (2014): 877-897.
- Pedregosa, F. et al. Scikit-learn: Machine Learning in Python,
  Journal of Machine Learning Research 12 (2011), 2825--2830.
- A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
  In Proc. of the 49th Annual Meeting of the Association for
  Computational Linguistics, Vol. 1, pages 1415-1424, 2011.
- Bevendorff et al. Generalizing Unmasking for Short Texts,
  Proceedings of NAACL (2019), 654-659.

"""

import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def binarize(y, threshold=0.5):
    y[y >= threshold] = 1
    y[y < threshold] = 0

    return y

def f1(true_y, pred_y, threshold=0.5):
    """
    Calculates the verification accuracy, assuming that every
    `score >= 0.5` represents an attribution.

    Parameters
    ----------
    prediction_scores : array [n_problems]

        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.

    ground_truth_scores : array [n_problems]

        The gold annotations provided for each problem.
        Will typically be `0` or `1`.

    Returns
    ----------
    acc = The number of correct attributions.

    References
    ----------
        E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
    """
    true_y = binarize(true_y, threshold=threshold)
    pred_y = binarize(pred_y, threshold=threshold)

    return f1_score(true_y, pred_y)


def auc(true_y, pred_y, threshold=0.5):
    """
    Calculates the AUC score (Area Under the Curve), a well-known
    scalar evaluation score for binary classifiers. This score
    also considers "unanswered" problem, where score = 0.5.

    Parameters
    ----------
    prediction_scores : array [n_problems]

        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.

    ground_truth_scores : array [n_problems]

        The gold annotations provided for each problem.
        Will typically be `0` or `1`.

    Returns
    ----------
    auc = the Area Under the Curve.

    References
    ----------
        E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.

    """
    true_y = binarize(true_y, threshold=threshold)
    return roc_auc_score(true_y, pred_y)


def c_at_1(true_y, pred_y, threshold=0.5):
    """
    Calculates the c@1 score, an evaluation method specific to the
    PAN competition. This method rewards predictions which leave
    some problems unanswered (score = 0.5). See:

        A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.

    Parameters
    ----------
    prediction_scores : array [n_problems]

        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.

    ground_truth_scores : array [n_problems]

        The gold annotations provided for each problem.
        Will always be `0` or `1`.

    Returns
    ----------
    c@1 = the c@1 measure (which accounts for unanswered
        problems.)


    References
    ----------
        - E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
        - A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.

    """

    pred_y = binarize(pred_y, threshold=threshold)

    n = float(len(pred_y))
    nc, nu = 0.0, 0.0

    for gt_score, pred_score in zip(true_y, pred_y):
        if pred_score == 0.5:
            nu += 1
        elif (pred_score > 0.5) == (gt_score > 0.5):
            nc += 1.0
    
    return (1 / n) * (nc + (nu * nc / n))


def f_05_u_score(true_y, pred_y, threshold=0.5, pos_label=1):
    """
    Return F0.5u score of prediction.

    :param true_y: true labels
    :param pred_y: predicted labels
    :param threshold: indication for non-decisions
    :param pos_label: positive class label (default = 1)
    :return: F0.5u score
    """

    true_y = binarize(true_y, threshold=threshold)

    n_tp = 0
    n_fn = 0
    n_fp = 0
    n_u = 0

    for i, pred in enumerate(pred_y):
        if pred == threshold:
            n_u += 1
        elif pred == pos_label and pred == true_y[i]:
            n_tp += 1
        elif pred == pos_label and pred != true_y[i]:
            n_fp += 1
        elif true_y[i] == pos_label and pred != true_y[i]:
            n_fn += 1

    return (1.25 * n_tp) / (1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp)


def load_folder(folder):
    problems = {}
    for fn in sorted(glob.glob(f'{folder}/*.jsonl')):
        for line in open(fn):
            d =  json.loads(line.strip())
            problems[d['problem_id']] = d['value']
    return problems


def evaluate_all(true_y, pred_y, threshold=0.5):
    """
    Convenience function: calculates all PAN20 evaluation measures
    and returns them as a dict
    """

    results = {'f1':  f1(true_y, pred_y, threshold=threshold),
               'auc': auc(true_y, pred_y, threshold=threshold),
               'c@1': c_at_1(true_y, pred_y, threshold=threshold),
               'F0.5u': f_05_u_score(true_y, pred_y, threshold=threshold)}
    
    results['overall'] = np.mean(tuple(results.values()))

    for k, v in results.items():
        results[k] = round(v, 3)

    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluation script AA@PAN2020')
    parser.add_argument('-i', type=str,
                        help='Path to the ground truth scores')
    parser.add_argument('-a', type=str,
                        help='Path to answers folder (system prediction)')
    parser.add_argument('-o', type=str, 
                        help='Path to output files')
    parser.add_argument('-threshold', type=float, default=0.5,
                        help='Binarization threshold (default=0.5)')
    args = parser.parse_args()

    # validate:
    if not args.i:
        raise ValueError('The collection path is required')
    if not args.a:
        raise ValueError('The answers folder is required')
    if not args.o:
        raise ValueError('The output path is required')
    
    # load:
    gt = load_folder(args.i)
    pred = load_folder(args.a)

    print(f'-> {len(gt)} problems in ground truth')
    print(f'-> {len(pred)} solutions explicitly proposed')

    # default missing problems to 0.5
    for probl_id in sorted(gt):
        if probl_id not in pred:
            pred[probl_id] = 0.5
    
    # sanity check:    
    assert len(gt) == len(pred)
    assert set(gt.keys()).union(set(pred)) == set(gt.keys())
    
    # align the scores:
    scores = [(gt[k], pred[k]) for k in sorted(gt)]
    gt, pred = zip(*scores)
    gt = np.array(gt, dtype=np.float64)
    pred = np.array(pred, dtype=np.float64)
    
    assert len(gt) == len(pred)

    # evaluate:
    results = evaluate_all(gt, pred, threshold=args.threshold)
    print(results)

    with open(args.o + os.sep + 'out.json', 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)

if __name__ == '__main__':
    main()