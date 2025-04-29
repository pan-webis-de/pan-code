# Copyright 2025 Janek Bevendorff, Webis
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

import typing as t
import types

import torch
import numpy as np

__all__ = ['DetectorBase']


class DetectorBase:
    """
    LLM detector base class.
    """

    def _normalize_scores(self, scores):
        """
        Normalize raw scores to the range [0, 1], whereby 1 indicates the highest probability
        of a text being machine-generated.

        The input is of the same type as the output of :meth:`_predict_impl`.

        By default, this is the identity function, which should be overridden by subclasses as needed.

        :param scores: unnormalized input scores
        :return: normalized output scores
        """
        return NotImplemented

    def _get_score_impl(self, text: t.Iterable[str]) -> t.Union[torch.Tensor, np.ndarray, t.Iterable[float]]:
        """
        Scoring implementation. To be overridden.
        The function should return a list of floats, a Torch tensor, or a Numpy array.
        """
        return NotImplemented

    def get_score(self, text: t.Union[str, t.Iterable[str]], normalize: bool = False) -> \
            t.Union[np.float32, np.ndarray, np.nan]:
        """
        Return scores indicating the probability of the input text(s) being machine-generated.

        Scores can be normalized to represent valid probability values with higher values meaning higher confidence
        in the texts being machine-generated. However, scores between different detector implementations are not
        necessarily comparable. If ``normalize`` is ``False`` (the default), unnormalized raw scores are returned
        instead. Interpretation of these raw scores is entirely implementation-dependent.

        :param text: input text or batch of input texts
        :param normalize: normalize scores to represent probabilities in the range [0, 1]
        :return: score indicating whether the input text being machine-generated
        """
        return_single = isinstance(text, str)
        text = [text] if return_single else text
        scores = self._get_score_impl(text)
        if normalize and scores is not NotImplemented:
            scores = self._normalize_scores(scores)
        scores = _to_numpy(scores, np.float32) if scores is not NotImplemented else _create_nan_array(len(text))
        return scores[0] if return_single else scores

    def _predict_impl(self, text: t.Iterable[str]) -> t.Union[torch.Tensor, np.ndarray, t.Iterable[bool]]:
        """
        Prediction implementation. To be overridden.

        If :meth:`_normalize_scores` is implemented, the default behaviour is to call
        :meth:`_get_score_impl` and threshold the value with ``0.5``. Otherwise, the return
        value is :class:``NotImplemented``.

        Overrides should return a Torch tensor or a Numpy array.
        """
        scores = self._get_score_impl(text)
        if scores is not NotImplemented and (scores := self._normalize_scores(scores)) is not NotImplemented:
            return _to_numpy(scores) >= 0.5
        return NotImplemented

    def predict(self, text: t.Union[str, t.Iterable[str]]) -> t.Union[np.int32, np.ndarray, np.nan]:
        """
        Make a prediction whether the input text(s) were written by a machine.

        :param text: input text or batch of input texts
        :return: boolean classifications of whether inputs are likely machine-generated
        """
        return_single = isinstance(text, str)
        text = [text] if return_single else text
        preds = self._predict_impl(text)
        preds = _to_numpy(preds, np.int32) if preds is not NotImplemented else _create_nan_array(len(text))
        return preds[0] if return_single else preds

    def _predict_with_score_impl(self, text: t.Iterable[str]) -> t.Tuple[
            t.Union[torch.Tensor, np.ndarray], t.Union[torch.Tensor, np.ndarray]]:
        """
        Predict and score implementation. To be overridden.

        If :meth:`_normalize_scores` is implemented, the default implementation will call
        :meth:`_normalize_scores` and threshold the return value with ``0.5`` for predictions.
        Otherwise, the default implementation calls :meth:`predict` and :meth:`get_score`. Subclasses
        should implement a more efficient implementation that avoids doing the same calculation twice.

        Overrides should return a Torch tensor or a Numpy array.
        """
        scores = self._get_score_impl(text)
        if scores is NotImplemented:
            return self._predict_impl(text), NotImplemented

        scores_norm = self._normalize_scores(scores)
        if scores_norm is NotImplemented:
            return self._predict_impl(text), scores

        return _to_numpy(scores_norm) >= 0.5, scores

    def predict_with_score(self, text: t.Union[str, t.Iterable[str]], normalize: bool = False) -> t.Tuple[
            t.Union[np.int32, np.ndarray, np.nan],
            t.Union[np.float32, np.ndarray, np.nan]]:
        """
        Make a prediction whether the input text(s) were written by a machine and return the
        result together with a numerical score.

        This is the same as calling :meth:`predict` and :meth:`get_score`, but may be more efficient,
        since detector models can run their detection only once and output both a predicted class and a score.

        :param text: input text or batch of input texts
        :param normalize: normalize scores
        :return: tuple of (predicted class, score)
        """
        return_single = isinstance(text, str)
        text = [text] if return_single else text
        preds, scores = self._predict_with_score_impl(text)

        if normalize and scores is not NotImplemented:
            scores = self._normalize_scores(scores)
        preds = _to_numpy(preds, np.int32) if preds is not NotImplemented else _create_nan_array(len(text))
        scores = _to_numpy(scores, np.float32) if scores is not NotImplemented else _create_nan_array(len(text))

        if return_single:
            return preds[0], scores[0]
        return preds, scores


def _create_nan_array(n):
    return np.empty(shape=(n,)) * np.nan


def _to_numpy(data, dtype: t.Type = np.float32):
    """
    Convert Torch Tensor or Numpy array to a Python type if necessary.
    """
    if isinstance(data, torch.Tensor):
        data = data.float().cpu().numpy().astype(dtype)
    elif isinstance(data, np.ndarray):
        return data.astype(dtype)
    elif isinstance(data, types.GeneratorType):
        return np.fromiter(data, dtype=dtype)
    return np.array(data, dtype=dtype)
