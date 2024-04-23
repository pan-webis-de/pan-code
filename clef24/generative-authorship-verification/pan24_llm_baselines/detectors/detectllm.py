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

from typing import List, Literal, Tuple

import torch

from pan24_llm_baselines.detectors.detectgpt import DetectGPT
from pan24_llm_baselines.perturbators.perturbator_base import PerturbatorBase
from pan24_llm_baselines.util import *

__all__ = ['DetectLLM']


class DetectLLM(DetectGPT):
    """
    DetectLLM LLM detector.

    This is a reimplementation of the original: https://github.com/mbzuai-nlp/DetectLLM

    References:
    ===========
        Su, Jinyan, Terry Yue Zhuo, Di Wang, and Preslav Nakov. 2023. “DetectLLM: Leveraging
        Log Rank Information for Zero-Shot Detection of Machine-Generated Text.” arXiv [Cs.CL].
        arXiv. http://arxiv.org/abs/2306.05540.
    """
    def __init__(self,
                 scoring_mode: Literal['lrr', 'npr'] = 'lrr',
                 base_model='tiiuae/falcon-7b',
                 device: TorchDeviceMapType = 'auto',
                 perturbator: PerturbatorBase = None,
                 n_samples=20,
                 batch_size=10,
                 verbose=True,
                 **base_model_args):
        """
        :param scoring_mode: ``'lrr'`` (Log-Likelihood Log-Rank Ratio) or
                             ``'npr'`` (Normalized Log-Rank Perturbation)
        :param base_model: base language model
        :param device: base model device
        :param perturbator: perturbation model for NPR (default: T5MaskPerturbator)
        :param n_samples: number of perturbed texts to generate for NPR
        :param batch_size: Log-likelihood prediction batch size
        :param verbose: show progress bar
        :param base_model_args: additional base model arguments
        """
        super().__init__(base_model, device, perturbator, n_samples,
                         batch_size, verbose, **base_model_args)
        self.scoring_mode = scoring_mode

    @property
    def scoring_mode(self):
        return self._scoring_mode

    @scoring_mode.setter
    def scoring_mode(self, mode):
        if mode not in ['lrr', 'npr']:
            raise ValueError(f'Invalid scoring mode: {mode}')
        self._scoring_mode = mode

    def _get_logits(self, text: List[str], verbose_msg: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoding = tokenize_sequences(text, self.base_tokenizer, self.base_model.device, 512)
        yield from model_batch_forward(self.base_model, encoding, self.batch_size, verbose_msg)

    def _normalize_scores(self, scores):
        if self.scoring_mode == 'lrr':
            return torch.sigmoid(10 * (scores.to(torch.float64) - 2))
        if self.scoring_mode == 'npr':
            return torch.sigmoid(10 * (scores.to(torch.float64) - 1.17))
        return scores

    def _lrr(self, text: List[str]) -> torch.Tensor:
        verbose_msg = 'Calculating logits' if self.verbose else None
        ll = []
        lrr = []
        for logits, labels, mask in self._get_logits(text, verbose_msg):
            ll.append(seq_label_cross_entropy(logits, labels, mask).cpu())
            lrr.append(seq_label_log_rank(logits, labels, mask).cpu())
        return torch.cat(ll) / torch.cat(lrr)

    def _npr(self, text: List[str]) -> torch.Tensor:
        perturbed = self.perturbator.perturb(text, n_variants=self.n_samples)

        verbose_msg = 'Calculating logit distribution' if self.verbose else None
        orig_rank = []
        pert_rank = []
        for logits, labels, mask in self._get_logits(text + perturbed, verbose_msg):
            text_idx = len(text) - len(orig_rank)
            if text_idx > 0:
                orig_rank.append(seq_label_log_rank(logits[:text_idx], labels[:text_idx], mask[:text_idx]).cpu())
            if text_idx < logits.shape[0]:
                pert_rank.append(seq_label_log_rank(logits[text_idx:], labels[text_idx:], mask[text_idx:]).cpu())

        orig_rank = torch.cat(orig_rank)
        pert_rank = torch.cat(pert_rank).view(-1, self.n_samples).mean(-1)
        return pert_rank / orig_rank

    @torch.inference_mode()
    def _get_score_impl(self, text: List[str]) -> torch.Tensor:
        if self.scoring_mode == 'lrr':
            return self._lrr(text)
        if self.scoring_mode == 'npr':
            return self._npr(text)
