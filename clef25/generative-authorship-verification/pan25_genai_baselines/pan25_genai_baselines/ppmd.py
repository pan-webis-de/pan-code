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

import numpy as np
import pyppmd

from pan25_genai_baselines.detector_base import DetectorBase

__all__ = ['PPMdDetector']


class PPMdDetector(DetectorBase):
    """
    Baseline LLM detector measuring the PPMd compression-based cosine between text halves.

    References:
    ===========
        Sculley, D., and C. E. Brodley. 2006. “Compression and Machine Learning: A New Perspective
        on Feature Space Vectors.” In Data Compression Conference (DCC’06), 332–41. IEEE.

        Halvani, Oren, Christian Winter, and Lukas Graner. 2017. “On the Usefulness of Compression
        Models for Authorship Verification.” In ACM International Conference Proceeding Series. Vol.
        Part F1305. Association for Computing Machinery. https://doi.org/10.1145/3098954.3104050.
    """

    CBC_OFFSET = 0.3125312531253126     # Trained on PAN'25 training set

    def _normalize_scores(self, scores):
        scores += self.CBC_OFFSET

        # Scale and normalize
        scores = 10. * (scores - .5)
        scores = 1. / (1. + np.exp(-scores))
        return scores

    def _get_score_impl(self, text: t.Iterable[str]) -> np.ndarray:
        scores = []
        for t in text:
            cx = len(pyppmd.compress(t[:len(t) // 2]))
            cy = len(pyppmd.compress(t[len(t) // 2:]))
            cxy = len(pyppmd.compress(t))
            score = (cx + cy - cxy) / np.sqrt(cx * cy)
            scores.append(score)
        return np.array(scores)
