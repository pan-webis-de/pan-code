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

from typing import List, Union

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
        return scores

    def _get_score_impl(self, text: List[str]) -> List[float]:
        """
        Scoring implementation. To be overridden.
        The function should return a list of floats, a Torch tensor, or a Numpy array.
        """
        raise NotImplementedError('Detector does not supporting scoring.')

    def get_score(self, text: Union[str, List[str]], normalize: bool = True) -> Union[float, List[float]]:
        """
        Return scores indicating the probability of the input text(s) being machine-generated.

        Scores are normalized to represent valid probability values with higher values meaning higher confidence
        in the texts being machine-generated. However, scores between different detector implementations are not
        necessarily comparable. If ``normalize`` is ``False``, unnormalized raw scores are returned instead.
        Interpretation of these raw scores is entirely implementation-dependent.

        :param text: input text or batch of input texts
        :param normalize: normalize scores to represent probabilities in the range [0, 1]
        :return: probability of the input text being machine-generated
        """
        scores = self._get_score_impl([text] if isinstance(text, str) else text)
        if normalize:
            scores = self._normalize_scores(scores)
        return _to_pytype(scores[0]) if isinstance(text, str) else _to_pytype(scores)

    def _predict_impl(self, text: List[str]) -> List[bool]:
        """
        Prediction implementation. To be overridden.
        The function should return a list of bools, a Torch tensor, or a Numpy array.
        """
        raise NotImplementedError('Detector does not supporting binary prediction.')

    def predict(self, text: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """
        Make a prediction whether the input text was written by a machine.

        :param text: input text or batch of input texts
        :return: boolean classifications of whether inputs are likely machine-generated
        """
        pred = self._predict_impl([text] if isinstance(text, str) else text)
        return _to_pytype(pred[0]) if isinstance(text, str) else _to_pytype(pred)


def _to_pytype(data):
    """
    Convert Torch Tensor or Numpy array to a Python type if necessary.
    """
    if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
        return data.tolist()
    return data
