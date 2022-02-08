#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Evaluation script for the Cross-Domain Authorship Verification task @PAN2022.
## Measures
The following evaluation measures are provided:
    - F1-score [Pedregosa et al. 2011]
    - Area-Under-the-Curve [Pedregosa et al. 2011]
    - c@1 [Peñas and Rodrigo 2011; Stamatatos 2014]
    - f_05_u_score [Bevendorff et al. 2019]
    - the complement of the Brier score loss [Pedregosa et al. 2011]
Systems will be evaluated, taking all of the measures into account.
## Formats
The script requires two files, one for the ground truth (gold standard)
and one for the system predictions. These files should be formatted using
the `jsonl`-convention, whereby each line should contain a valid
json-string: e.g.
``` json
    {"id": "1", "value": 0.123}
    {"id": "2", "value": 0.5}
    {"id": "3", "value": 0.888}
```
Only files will be considered that:
- have the `.jsonl` extension
- are properly encoded as UTF-8.
Please note:
    * For the c@1, all scores are will binarized using
      the conventional thresholds:
        * score < 0.5 -> 0
        * score > 0.5 -> 1
    * A score of *exactly* 0.5, will be considered a non-decision.
    * All problems which are present in the ground truth, but which
      are *not* provided an answer to by the system, will automatically
      be set to 0.5.
    * Non-answers are removed for the F1 score calculation below, but they
      are taken into account by the AUC and Brier score.
## Dependencies:
- Python 3.6+ (we recommend the Anaconda Python distribution)
- scikit-learn
## Usage
From the command line:
>>> python pan22-verif-evaluator.py -i TRUTH -a ANSWERS -o OUTPUT
where
    TRUTH is the path to the folder of 'truth.jsonl' file with the ground truth
    ANSWERS is the path to the folder of the 'answers.jsonl' file for a submitted method
    OUTPUT is the path to the folder where the results of the evaluation will be saved
Example: 
>>> python pan22_verif_evaluator.py -i "datasets/test_truth" \
        -a "submitted_models/answers" \
        -o "pan22-evaluation"
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
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss


def binarize(y, threshold=0.5, triple_valued=False):
    y = np.array(y)
    y = np.ma.fix_invalid(y, fill_value=threshold)
    if triple_valued:
        y[y > threshold] = 1
    else:
        y[y >= threshold] = 1
    y[y < threshold] = 0
    return y


def auc(true_y, pred_y):
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
    try:
        return roc_auc_score(true_y, pred_y)
    except ValueError:
        return 0.0


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

    n = float(len(pred_y))
    nc, nu = 0.0, 0.0

    for gt_score, pred_score in zip(true_y, pred_y):
        if pred_score == 0.5:
            nu += 1
        elif (pred_score > 0.5) == (gt_score > 0.5):
            nc += 1.0
    
    return (1 / n) * (nc + (nu * nc / n))


def f1(true_y, pred_y):
    """
    Assesses verification performance, assuming that every
    `score > 0.5` represents a same-author pair decision.
    Note that all non-decisions (scores == 0.5) are ignored
    by this metric.
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
    true_y_filtered, pred_y_filtered = [], []

    for true, pred in zip(true_y, pred_y):
        if pred != 0.5:
            true_y_filtered.append(true)
            pred_y_filtered.append(pred)
    
    pred_y_filtered = binarize(pred_y_filtered)

    return f1_score(true_y_filtered, pred_y_filtered)


def f_05_u_score(true_y, pred_y, pos_label=1, threshold=0.5):
    """
    Return F0.5u score of prediction.
    :param true_y: true labels
    :param pred_y: predicted labels
    :param threshold: indication for non-decisions (default = 0.5)
    :param pos_label: positive class label (default = 1)
    :return: F0.5u score
    """

    pred_y = binarize(pred_y, triple_valued=True)

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

def brier_score(true_y, pred_y):
    """
    Calculates the complement of the Brier score loss (which is bounded
    to the [0-1]), so that higher scores indicate better performance.
    This score also considers "unanswered" problem, where score = 0.5.
    We use the Brier implementation in scikit-learn [Pedregosa et al.
    2011].
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
    brier = float
        the complement of the Brier score
    References
    ----------
    - Pedregosa, F. et al. Scikit-learn: Machine Learning in Python,
      Journal of Machine Learning Research 12 (2011), 2825--2830.
    """
    try:
        return 1 - brier_score_loss(true_y, pred_y)
    except ValueError:
        return 0.0


def load_file(fn):
    problems = {}
    for line in open(fn):
        d =  json.loads(line.strip())
        if 'value' in d:
            problems[d['id']] = d['value']
        else:
            problems[d['id']] = int(d['same'])
    return problems


def evaluate_all(true_y, pred_y):
    """
    Convenience function: calculates all PAN20 evaluation measures
    and returns them as a dict, including the 'overall' score, which
    is the mean of the individual metrics (0 >= metric >= 1). All 
    scores get rounded to three digits.
    """

    results = {'auc': auc(true_y, pred_y),
               'c@1': c_at_1(true_y, pred_y),
               'f_05_u': f_05_u_score(true_y, pred_y),
               'F1': f1(true_y, pred_y),
               'brier': brier_score(true_y, pred_y)
              }
    
    results['overall'] = np.mean(list(results.values()))

    for k, v in results.items():
        results[k] = round(v, 3)

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluation script AA@PAN2020')
    parser.add_argument('-i', type=str,
                        help='Path to the folder of jsonl-file with ground truth')
    parser.add_argument('-a', type=str,
                        help='Path to the folder of jsonl-file with the answers (system predictions)')
    parser.add_argument('-o', type=str, 
                        help='Path to output files')
    args = parser.parse_args()

    # validate:
    if not args.i:
        raise ValueError('The ground truth path is required')
    if not args.a:
        raise ValueError('The answers path is required')
    if not args.o:
        raise ValueError('The output folder path is required')
    
    # load:
    gt = load_file(f"{args.i}/truth.jsonl")
    pred = load_file(f"{args.a}/answers.jsonl")

    print('->', len(gt), 'problems in ground truth')
    print('->', len(pred), 'solutions explicitly proposed')

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
    results = evaluate_all(gt, pred)
    print(results)

    with open(args.o + os.sep + 'out.json', 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)
    
    with open(args.o + os.sep + 'evaluation.prototext', 'w') as f:
        for metric, score in results.items():
            f.write('measure {\n')
            f.write(' key: "' + metric + '"\n')
            f.write(' value: "' + str(score) + '"\n')
            f.write('}\n')

if __name__ == '__main__':
    main()
