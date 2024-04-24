# Copyright 2024 Janek Bevendorff, Webis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from random import randint
import warnings
from typing import List

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

from pan24_llm_baselines.detectors.detector_base import DetectorBase

__all__ = ['UnmaskingDetector']

warnings.simplefilter('ignore', category=ConvergenceWarning)


def tokenize(text):
    return [text[i:i + 3] for i in range(0, len(text) - 2)]


def get_token_freqs(*token_lists):
    freqs = defaultdict(int)
    for tokens in token_lists:
        for t in tokens:
            freqs[t] += 1
    return freqs


def bootstrap_tokens(tokens, n_tokens):
    return [tokens[randint(0, len(tokens) - 1)] for _ in range(n_tokens)]


def create_chunks(tokens, chunk_size, n_chunks):
    return [bootstrap_tokens(tokens, chunk_size) for _ in range(n_chunks)]


def chunks_to_matrix(chunks, top_token_list):
    mat = []
    for c in chunks:
        freq = get_token_freqs(c)
        mat.append([freq[t] for t in top_token_list])

    return np.array(mat)


def deconstruct(x_left, x_right, rounds, n_delete=5, cv_folds=10, smoothing_kernel_size=3):
    X = np.vstack((x_left, x_right))
    y = np.zeros(len(x_left) + len(x_right))
    y[:len(x_left)] = 1.0
    X, y = shuffle(X, y)

    rounds = min(rounds, (X.shape[1] - 1) // n_delete)
    scores = np.zeros(rounds)
    for i in range(rounds):
        cv = cross_validate(LinearSVC(dual='auto'), X, y, cv=cv_folds, return_estimator=True)
        scores[i] = max(0.0, (cv['test_score'].mean() - .5) * 2)
        X = np.delete(X, np.argsort(cv['estimator'][0].coef_, axis=None)[::-1][:n_delete], axis=1)

    if smoothing_kernel_size:
        scores = np.convolve(scores, np.ones(smoothing_kernel_size) / smoothing_kernel_size, mode='valid')

    return scores


class UnmaskingDetector(DetectorBase):
    """
    LLM detector calculating normalized cumulative sum of the authorship unmasking curve points.

    References:
    ===========
        Koppel, Moshe, and Jonathan Schler. 2004. “Authorship Verification as a One-Class
        Classification Problem.” In Proceedings, Twenty-First International Conference on
        Machine Learning, ICML 2004, 489–95.

        Bevendorff, Janek, Benno Stein, Matthias Hagen, and Martin Potthast. 2019. “Generalizing
        Unmasking for Short Texts.” In Proceedings of the 2019 Conference of the North, 654–59.
        Stroudsburg, PA, USA: Association for Computational Linguistics.
    """
    def __init__(self, rounds=35, top_n=200, cv_folds=10, n_delete=4, chunk_size=700, n_chunks=30):
        """
        :param rounds: number of deconstruction rounds
        :param top_n: number of top tokens to sample
        :param cv_folds: number of cross-validation folds
        :param n_delete: number of features to eliminate in each round
        :param chunk_size: size of bootstrapped chunks
        :param n_chunks: number of chunks to generate
        :return: score in [0, 1] indicating the "humanness" of the text
        """
        self.rounds = rounds
        self.top_n = top_n
        self.cv_folds = cv_folds
        self.n_delete = n_delete
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks

    def _get_score_impl(self, text: List[str]) -> List[float]:
        scores = []
        for t in text:
            tokens_left = tokenize(t[:len(t) // 2])
            tokens_right = tokenize(t[len(t) // 2:])

            chunks_left = create_chunks(tokens_left, self.chunk_size, self.n_chunks)
            chunks_right = create_chunks(tokens_right, self.chunk_size, self.n_chunks)

            token_freqs = get_token_freqs(*chunks_left, *chunks_right)
            most_frequent = sorted(token_freqs.keys(), key=lambda x: token_freqs[x], reverse=True)[:self.top_n]
            x_left = chunks_to_matrix(chunks_left, most_frequent)
            x_right = chunks_to_matrix(chunks_right, most_frequent)

            degen_acc = deconstruct(x_left, x_right, self.rounds, self.n_delete, self.cv_folds)
            scores.append(1.0 - (np.sum(degen_acc) / len(degen_acc)))
        return scores
