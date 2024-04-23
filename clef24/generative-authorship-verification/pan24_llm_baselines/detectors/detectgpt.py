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

from typing import List

import torch

from pan24_llm_baselines.detectors.detector_base import DetectorBase
from pan24_llm_baselines.perturbators.perturbator_base import PerturbatorBase
from pan24_llm_baselines.perturbators.t5_mask import T5MaskPerturbator
from pan24_llm_baselines.util import *

__all__ = ['DetectGPT']


class DetectGPT(DetectorBase):
    """
    DetectGPT LLM detector.

    This is a reimplementation of the original: https://github.com/eric-mitchell/detect-gpt

    References:
    ===========
        Mitchell, Eric, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning,
        and Chelsea Finn. 2023. “DetectGPT: Zero-Shot Machine-Generated Text
        Detection Using Probability Curvature.” arXiv [Cs.CL]. arXiv.
        http://arxiv.org/abs/2301.11305.
    """
    def __init__(self,
                 base_model='tiiuae/falcon-7b',
                 device: TorchDeviceMapType = 'auto',
                 perturbator: PerturbatorBase = None,
                 n_samples=100,
                 batch_size=10,
                 verbose=True,
                 **base_model_args):
        """
        :param base_model: base language model
        :param device: base model device
        :param perturbator: perturbation model (default: T5MaskPerturbator)
        :param n_samples: number of perturbed texts to generate
        :param batch_size: Log-likelihood prediction batch size
        :param verbose: show progress bar
        :param base_model_args: additional base model arguments
        """

        self.n_samples = n_samples
        self.batch_size = batch_size
        self.verbose = verbose
        self.perturbator = perturbator if perturbator else T5MaskPerturbator(device=device)
        self.base_model = load_model(base_model, device_map=device, **base_model_args)
        self.base_tokenizer = load_tokenizer(base_model)

    def _normalize_scores(self, scores):
        return torch.sigmoid(2 * (scores.to(torch.float64) - 2.1))

    @torch.inference_mode()
    def _get_score_impl(self, text: List[str]) -> List[float]:
        perturbed = self.perturbator.perturb(text, n_variants=self.n_samples)

        encoding = tokenize_sequences(text + perturbed, self.base_tokenizer, self.base_model.device, 512)
        ll = -batch_seq_log_likelihood(self.base_model, encoding, self.batch_size, self.verbose)
        ll_orig, ll_pert = ll[:len(text)], ll[len(text):]
        ll_pert = ll_pert.view(len(text), self.n_samples)
        ll_pert_std = ll_pert.std(-1, correction=1)
        ll_pert = ll_pert.mean(-1)

        return (ll_orig - ll_pert) / ll_pert_std
