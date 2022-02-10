#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Naive, Distance-Based Baseline
## Introduction
This baseline offers a naive, yet fast solution to the 
PAN2022 track on authorship verification. All documents
are represented using a bag-of-character-ngrams model,
that is TFIDF weighted. The cosine similarity between
each document pair in the calibration data set is
calculated. Finally, the resulting similarities are
optimized, and projected through a simple rescaling
operation, so that they can function as pseudo-probabi-
lities, indiciating the likelihood that a document-pair
is a same-author pair. Via a grid search, the optimal
verification threshold is determined, taking into account
that some difficult problems can be left unanswered.
Through setting `num_iterations` to an integer > 0,
a bootstrapped variant of this procedure can be used.
In this case, the similarity calculation is applied in
an iterative procedure to a randomly sampled subset of
the available features. The average similarity is then
used downstream. This imputation procedure is inspired
by the imposters approach. 
## Dependencies
- Python 3.6+ (we recommend the Anaconda Python distribution)
- scikit-learn, numpy, scipy
- pan22_verif_evaluator.py
Example usage from the command line to train the model:
>>> python pan22-verif-baseline-cngdist.py \
          --train \
          --model_dir="models/baseline" \
          -p="datasets/pan22-authorship-verification-training" \
          -t="datasets/pan22-authorship-verification-training" \
          -num_iterations=0 
Example usage from the command line to test the model:
>>> python pan22-verif-baseline-cngdist.py \
          --model_dir="models/baseline" \
          -i="datasets/pan22-authorship-verification-test" \
          -num_iterations=0 \
          -output="out"
"""

import argparse
import json
import os
import random
from pathlib import Path
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.spatial.distance import cosine
import pickle

from pan22_verif_evaluator import evaluate_all


def cosine_sim(a, b):
#    print(a, b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def rescale(value, orig_min, orig_max, new_min, new_max):
    """
    Rescales a `value` in the old range defined by
    `orig_min` and `orig_max`, to the new range
    `new_min` and `new_max`. Assumes that
    `orig_min` <= value <= `orig_max`.
    Parameters
    ----------
    value: float, default=None
        The value to be rescaled.
    orig_min: float, default=None
        The minimum of the original range.
    orig_max: float, default=None
        The minimum of the original range.
    new_min: float, default=None
        The minimum of the new range.
    new_max: float, default=None
        The minimum of the new range.
    Returns
    ----------
    new_value: float
        The rescaled value.
    """

    orig_span = orig_max - orig_min
    new_span = new_max - new_min

    try:
        scaled_value = float(value - orig_min) / float(orig_span)
    except ZeroDivisionError:
        orig_span += 1e-6
        scaled_value = float(value - orig_min) / float(orig_span)

    return new_min + (scaled_value * new_span)


def correct_scores(scores, p1, p2):
    for sc in scores:
        if sc <= p1:
            yield rescale(sc, 0, p1, 0, 0.49)
        elif p1 < sc < p2:
            yield 0.5
        else:
            yield rescale(sc, p2, 1, 0.51, 1)  # np.array(list


def train(input_pairs, input_truth, model_directory, vocab_size, ngram_size, num_iterations, dropout):
    gold = {}
    for line in open(input_truth):
        d = json.loads(line.strip())
        gold[d['id']] = int(d['same'])

    # truncation for development purposes
    cutoff = 0
    if cutoff:
        gold = dict(random.sample(gold.items(), cutoff))
        print(len(gold))

    texts = []
    for line in open(input_pairs,encoding='utf8'):
        d = json.loads(line.strip())
        if d['id'] in gold:
            texts.extend(d['pair'])

    print('-> constructing vectorizer')
    vectorizer = TfidfVectorizer(max_features=vocab_size, analyzer='char',
                                 ngram_range=(ngram_size, ngram_size))
    vectorizer.fit(texts)

    if num_iterations:
        total_feats = len(vectorizer.get_feature_names())
        keep_feats = int(total_feats * dropout)

        rnd_feature_idxs = []
        for _ in range(num_iterations):
            rnd_feature_idxs.append(np.random.choice(total_feats,
                                                     keep_feats,
                                                     replace=False))
        rnd_feature_idxs = np.array(rnd_feature_idxs)

    print('-> calculating pairwise similarities')
    similarities, labels = [], []
    for line in open(input_pairs,encoding='utf8'):
        d = json.loads(line.strip())
        if d['id'] in gold:
            x1, x2 = vectorizer.transform(d['pair']).toarray()
            if num_iterations:
                similarities_ = []
                for i in range(num_iterations):
                    similarities_.append(cosine_sim(x1[rnd_feature_idxs[i, :]],
                                                    x2[rnd_feature_idxs[i, :]]))
                similarities.append(np.mean(similarities_))
            else:
                similarities.append(cosine_sim(x1, x2))
            labels.append(gold[d['id']])

    similarities = np.array(similarities, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    print('-> grid search p1/p2:')
    step_size = 0.01
    thresholds = np.arange(0.01, 0.99, step_size)
    combs = [(p1, p2) for (p1, p2) in combinations(thresholds, 2) if p1 < p2]

    params = {}
    for p1, p2 in combs:
        corrected_scores = np.array(list(correct_scores(similarities, p1=p1, p2=p2)))
        score = evaluate_all(pred_y=corrected_scores, true_y=labels)
        params[(p1, p2)] = score['overall']
    opt_p1, opt_p2 = max(params, key=params.get)
    print('optimal p1/p2:', opt_p1, opt_p2)

    corrected_scores = np.array(list(correct_scores(similarities, p1=opt_p1, p2=opt_p2)))
    print('optimal score:', evaluate_all(pred_y=corrected_scores, true_y=labels))

    print('-> determining optimal threshold')
    scores = []
    for th in np.linspace(0.25, 0.75, 1000):
        adjusted = (corrected_scores >= th) * 1
        scores.append((th,
                       f1_score(labels, adjusted),
                       precision_score(labels, adjusted),
                       recall_score(labels, adjusted)))
    thresholds, f1s, precisions, recalls = zip(*scores)

    max_idx = np.array(f1s).argmax()
    max_f1 = f1s[max_idx]
    max_th = thresholds[max_idx]
    print(f'Dev results -> F1={max_f1} at th={max_th}')

    pickle.dump(vectorizer, open(model_directory / 'vectorizer.pickle', 'wb'))
    pickle.dump(opt_p1, open(model_directory / 'opt_p1.pickle', 'wb'))
    pickle.dump(opt_p2, open(model_directory / 'opt_p2.pickle', 'wb'))
    if num_iterations:
        pickle.dump(rnd_feature_idxs, open(model_directory / 'rnd_feature_idxs.pickle', 'wb'))


def test(test_pairs, output_dir, model_directory, num_iterations):
    vectorizer = pickle.load(open(model_directory / 'vectorizer.pickle', 'rb'))
    opt_p1 = pickle.load(open(model_directory / 'opt_p1.pickle', 'rb'))
    opt_p2 = pickle.load(open(model_directory / 'opt_p2.pickle', 'rb'))
    print('p1 =',opt_p1,', p2 =',opt_p2)
    if num_iterations:
        rnd_feature_idxs = pickle.load(open(model_directory / 'rnd_feature_idxs.pickle', 'rb'))

    print('-> calculating test similarities')
    with open(output_dir / 'answers.jsonl', 'w') as outf:
        count=0
        for line in open(test_pairs,encoding='utf8'):
            count=count+1
            d = json.loads(line.strip())
            problem_id = d['id']
            x1, x2 = vectorizer.transform(d['pair']).toarray()
            if num_iterations:
                similarities_ = []
                for i in range(num_iterations):
                    similarities_.append(cosine_sim(x1[rnd_feature_idxs[i, :]],
                                                    x2[rnd_feature_idxs[i, :]]))
                    similarity = np.mean(similarities_)
            else:
                similarity = cosine_sim(x1, x2)

            similarity = np.array(list(correct_scores([similarity], p1=opt_p1, p2=opt_p2)))[0]
            r = {'id': problem_id, 'value': similarity}
            outf.write(json.dumps(r) + '\n')
        print(count,'cases')
         

def main():
    parser = argparse.ArgumentParser(
        description='PAN-22 Cross-domain Authorship Verification task: Distance-based baseline')
    # data settings:
    parser.add_argument('--train', action='store_true', help='If True, train a model from the given '
                                                             'input pair and input truth. If False, load a model'
                                                             'and test on the test dir')
    parser.add_argument('-p', '--input_pairs', type=str, help='Path to the jsonl-file with the input pairs')
    parser.add_argument('-t', '--input_truth', type=str, help='Path to the ground truth-file for the input pairs')
    parser.add_argument('-i', '--test_dir', type=str, help='Path to the directory that contains the test pairs.jsonl')
    parser.add_argument('-o', '--output', type=str, help='Path to the output folder for the predictions.\
                                                                         (Will be overwritten if it exist already.)')
    parser.add_argument('--model_dir', type=str, default='./baseline-distance-data', help='Path to the directory storing the model')

    # algorithmic settings:
    parser.add_argument('-seed', default=2020, type=int, help='Random seed')
    parser.add_argument('-vocab_size', default=3000, type=int,
                        help='Maximum number of vocabulary items in feature space')
    parser.add_argument('-ngram_size', default=4, type=int, help='Size of the ngrams')
    parser.add_argument('-num_iterations', default=0, type=int, help='Number of iterations (`k`); zero by default')
    parser.add_argument('-dropout', default=.5, type=float, help='Proportion of features to keep in each iteration')

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    random.seed(args.seed)

    model_directory = Path(args.model_dir)

    if args.train:
        if not args.input_pairs or not args.input_truth:
            print("STOP. Missing required parameters: --input_pairs or --input_truth")
            exit(1)
        model_directory.mkdir(parents=True, exist_ok=True)
        train(args.input_pairs + os.sep + 'pairs.jsonl', args.input_truth + os.sep + 'truth.jsonl', model_directory,
                    args.vocab_size, args.ngram_size, args.num_iterations, args.dropout)

    else:
        if not args.test_dir or not args.output:
            print("STOP. Missing required parameters: --test_dir or --output")
            exit(1)
        if not model_directory.exists():
            print("STOP. Model does not exist at " + model_directory)
            exit(1)

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        test(str(Path(args.test_dir)) + os.sep + 'pairs.jsonl',
                Path(output_dir), model_directory, args.num_iterations)


if __name__ == '__main__':
    main()
