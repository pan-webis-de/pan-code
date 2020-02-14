"""
 A reference implementation of the imposters method
 for open-set authorship attribution (Koppel and Winter 2014).

 This implementation tries to stay as close as possible to the original
 paper and the optimal parametrizations reported there. There are, however,
 two important differences:
    + the original paper always compared two individual
    documents to one another (document pair).  The present implementation compares
    individual documents to an author's profile -- it corresponds the
    profile-based approach in Kestemont et al. (2016).
    + because we are dealing with relative large imposter sets, we only use a
    random sample of all the available imposters in each iteration
    (the number is controled by `num_init_impost`).

 It can be applied to datasets of PAN-19 cross-domain authorship attribution task
 See details here: http://pan.webis.de/clef19/pan19-web/author-identification.html

 Dependencies:
 - Python 3.6+ (we recommend the Anaconda Python distribution)
 - scikit-learn
 - numpy
 - numba (useful for fast, custom vector metrics)
 - tqdm (useful for progress monitoring)

 # Usage from the command line:
>>> python pan19-cdaa-baseline-imposters.py -i COLLECTION -o OUTPUT -impo IMPOSTERS
where
    COLLECTION (-i) is the path to the main folder of the evaluation collection
    OUTPUT (-o) is the path to the folder where the results of the evaluation will be saved
    IMPOSTERS (-impo) is the path to the folder with the language-specific imposter collections from PAN-2019

Questions/comments: mike.kestemont@gmail.com

Example:
>>> python pan19-cdaa-baseline-imposters.py -i "/mydata/pan19-cdaa-development-corpus" -o "/mydata/pan19-answers" -impo "/mydata/imposters"

# References:
    - M. Koppel and Y. Winter (2014), Determining if Two
      Documents are by the Same Author, JASIST, 65(1): 178-187.
    - Cha SH (2007). Comprehensive Survey on Distance/Similarity Measures
      between Probability Density Functions. International Journ.
      of Math. Models and Methods in Applied Sciences, 1(4):300-307.
    - Kestemont, M., Stover, J., Koppel, M., Karsdorp, K. & Daelemans,W., 
	  Authenticating the writings of Julius Caesar. In: Expert
      Systems with Applications 63 (2016): pp. 86-96.
"""

from __future__ import print_function
import argparse
import json
import os
import glob
import shutil
from tqdm import tqdm

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from numba import jit

"""
def minmax_(a, b):
    # A (slow) numpy implementation of minmax
    x = np.vstack((a, b))
    return 1 - np.sum(x.min(axis=0)) / (np.sum(x.max(axis=0)) + 1e-8)
"""

@jit
def minmax(x, y, rnd_feature_idxs):
    """
    Calculates the pairwise "minmax" distance between
    two vectors, but limited to the `rnd_feature_idxs`
    specified. Note that this function is symmetric,
    so that `minmax(x, y) = minmax(y, x)`.

    Parameters
    ----------
    x: float array
        The first vector of the vector pair.
    y: float array
        The second vector of the vector pair.
    rnd_feature_idxs: int array
        The list of indexes along which the distance
        has to be calculated (useful for bootstrapping).

    Returns
    ----------
    float: minmax(x, y)
    """

    mins, maxs = 0.0, 0.0
    a, b = 0.0, 0.0

    for i in rnd_feature_idxs:

        a, b = x[i], y[i]

        if a >= b:
            maxs += a
            mins += b
        else:
            maxs += b
            mins += a

    return 1.0 - (mins / (maxs + 1e-6)) # avoid zero division

class Imposters:

    def __init__(self, num_init_impost, num_actual_impost, num_potent_impost,
                 num_iterations, vocab_size, ngram_size, dropout):
        self.num_init_impost = num_init_impost
        self.num_actual_impost = num_actual_impost
        self.num_potent_impost = num_potent_impost
        self.num_iterations = num_iterations
        self.dropout = dropout

        self.vectorizer = TfidfVectorizer(max_features=vocab_size, analyzer='char',
                                          ngram_range=(ngram_size, ngram_size))
        self.fitted_ = False

    def fit(self, candidate_documents, candidates, imposter_documents):
        self.vectorizer.fit(candidate_documents)
        self.total_feats = len(self.vectorizer.get_feature_names())
        self.keep_feats = int(self.total_feats * self.dropout)

        candidate_X = self.vectorizer.transform(candidate_documents).toarray()
        self.imposter_X = self.vectorizer.transform(imposter_documents).toarray()

        # make centroids for each candidate (based on sklearn's nearest centroid clf):
        self.author_enc = LabelEncoder()
        y_ind = self.author_enc.fit_transform(candidates)
        self.classes_ = self.author_enc.classes_

        self.centroids_ = np.empty((len(self.classes_), self.total_feats), dtype=np.float64)
        for cur_class in range(len(self.classes_)):
            center_mask = y_ind == cur_class
            self.centroids_[cur_class] = candidate_X[center_mask].mean(axis=0)

        self.fitted_ = True
        return self

    def predict_proba(self, trg):
        """
        `trg` is assumed to be the unknown text
        """
        trg = self.vectorizer.transform([trg]).toarray()[0]

        probas = np.zeros((len(self.classes_), self.num_iterations))

        for i in tqdm(range(self.num_iterations)):
            # select random features to keep
            rnd_feature_idxs = np.random.choice(self.total_feats,
                                                self.keep_feats,
                                                replace=False)
            init_imp_idxs = np.random.choice(self.imposter_X.shape[0],
                                             self.num_init_impost,
                                             replace=False)

            # get m closest potential imposters
            distances = [minmax(trg, self.imposter_X[j], rnd_feature_idxs) for j in init_imp_idxs]

            # select minimal distance from random subset of n imposters:
            min_dist = np.random.choice(np.sort(distances)[:self.num_actual_impost],
                                        size=self.num_actual_impost,
                                        replace=False).min()

            for cand_idx in range(len(self.classes_)):
                # augment if trg is closer to author than to the closest imposter:
                if minmax(self.centroids_[cand_idx], trg, rnd_feature_idxs) <= min_dist:
                    probas[cand_idx, i] = 1

        probas = np.mean(probas, axis=1)
        return probas


def main():
    parser = argparse.ArgumentParser(description='Imposter verification: reference implementation')

    # data settings:
    parser.add_argument('-i', type=str, required=True,
                        help='Path to the main folder of a collection of attribution problems')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to an output folder')
    parser.add_argument('-impo', type=str, required=True,
                        help='Path to an imposters folder')

    # imposter settings:
    parser.add_argument('--num_init_impost', default=250, type=int,
                        help='Number of initially sampled imposters')
    parser.add_argument('--num_potent_impost', default=125, type=int,
                        help='Number of (most similar) imposters to use (`m`)')
    parser.add_argument('--num_actual_impost', default=25, type=int,
                        help='Number of actual imposters to sample (`n`)')
    parser.add_argument('--num_iterations', default=100, type=int,
                        help='Number of iterations (`k`)')
    parser.add_argument('--threshold', default=0.1, type=float,
                        help='Similarity threshold (`sigma*`)')
    parser.add_argument('--seed', default=2019, type=int,
                        help='Random seed')
    parser.add_argument('--vocab_size', default=10000, type=int,
                        help='Maximmum number of vocabulary items in feature space')
    parser.add_argument('--ngram_size', default=4, type=int,
                        help='Size of the ngrams')
    parser.add_argument('--dropout', default=.5, type=float,
                        help='Proportion of features to keep in each iteration')

    args = parser.parse_args()
    print(args)

    np.random.seed(2019)

    try:
        shutil.rmtree(args.o)
    except FileNotFoundError:
        pass
    os.mkdir(args.o)

    verifier = Imposters(num_init_impost=args.num_init_impost,
                         num_potent_impost=args.num_potent_impost,
                         num_actual_impost=args.num_actual_impost,
                         num_iterations=args.num_iterations,
                         ngram_size=args.ngram_size,
                         vocab_size=args.vocab_size,
                         dropout=args.dropout)

    problems, languages = [], []
    with open(args.i+os.sep+'collection-info.json') as f:
        for attrib in json.loads(f.read()):
            problems.append(attrib['problem-name'])
            languages.append(attrib['language'])

    for problem, language in zip(problems, languages):
        print({problem}, ({language}))

        with open(args.i+os.sep+problem+os.sep+'problem-info.json') as f:
            fj = json.loads(f.read())

        candidates = [a['author-name'] for a in fj['candidate-authors']]
        unk_folder = fj['unknown-folder']

        # candidate author documents:
        candidate_documents, candidate_authors = [], []
        for candidate in candidates:
            for fn in sorted(glob.glob(args.i+os.sep+problem+os.sep+candidate+os.sep+'*.txt')):
                with open(fn) as f:
                    candidate_documents.append(f.read())
                    candidate_authors.append(candidate)
        print('  + loaded ',len(candidate_documents),' candidate documents')

        # load imposters:
        imposters = {}
        for fn in sorted(glob.glob(args.impo+os.sep+language+os.sep+'*.txt')):
            with open(fn) as f:
                imposters[os.path.basename(fn)] = f.read()
        print('  + loaded ',len(imposters),' imposters documents')

        verifier.fit(candidate_documents=candidate_documents,
                     candidates=candidate_authors,
                     imposter_documents=list(imposters.values()))

        # analyze:
        answers = []
        for fn in sorted(glob.glob(args.i+os.sep+problem+os.sep+unk_folder+os.sep+'*.txt')):
            print('    -', fn)
            with open(fn) as f:
                text = f.read()
            probas = verifier.predict_proba(text)
            winning_idx = probas.argmax()
            if probas[winning_idx] >= args.threshold:
                winner = verifier.classes_[winning_idx]
            else:
                winner = '<UNK>'
            answers.append({'unknown-text': os.path.basename(fn),
                            'predicted-author': winner})

        with open(args.o+os.sep+'answers-'+problem+'.json', 'w') as f:
            f.write(json.dumps(answers, indent=4))

if __name__ == '__main__':
    main()