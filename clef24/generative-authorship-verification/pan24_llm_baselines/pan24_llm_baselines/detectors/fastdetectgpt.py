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

from typing import List, Tuple

import torch
import torch.nn.functional as F

from pan24_llm_baselines.detectors.detectgpt import DetectGPT
from pan24_llm_baselines.util import *

__all__ = ['FastDetectGPT']


class FastDetectGPT(DetectGPT):
    """
    Fast-DetectGPT LLM detector.

    This is a reimplementation of the original: https://github.com/baoguangsheng/fast-detect-gpt/

    References:
    ===========
        Bao, Guangsheng, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang. 2023.
        “Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional
        Probability Curvature.” arXiv [Cs.CL]. arXiv. https://arxiv.org/abs/2310.05130.
    """
    def __init__(self,
                 base_model='tiiuae/falcon-7b',
                 device: TorchDeviceMapType = 'auto',
                 n_samples=10000,
                 batch_size=10,
                 verbose=True,
                 **base_model_args):
        """
        :param base_model: base language model
        :param device: base model device
        :param n_samples: number of samples to draw from distribution
        :param batch_size: Log-likelihood prediction batch size
        :param verbose: show progress bar
        :param base_model_args: additional base model arguments
        """
        super().__init__(base_model, device, None, n_samples, batch_size, verbose, **base_model_args)

    def _normalize_scores(self, scores):
        return torch.sigmoid(1 / 25 * (scores.to(torch.float64) - 70))

    def _proc_batch(self, logits: torch.Tensor, labels: torch.Tensor,
                    mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process batch of texts by generating samples from it and returning the likelihoods.

        :param logits: input logits
        :param labels: input labels
        :return: likelihood of original texts, likelihood of samples
        """
        lls_orig = []
        lls_sampled = []
        for lo, la, ma in zip(logits, labels, mask):
            lls_orig.append(-seq_label_cross_entropy(lo.unsqueeze(0),
                                                     la.view(1, len(la)),
                                                     ma.view(1, len(ma))).cpu())

            lp = F.log_softmax(lo, dim=-1)
            if lp.is_cuda and lp.dtype == torch.bfloat16:
                # Cuda multinomial not yet supported for bfloat16
                lp = lp.to(torch.float16)
            seq_len = ma.sum()
            dist = torch.distributions.Categorical(logits=lp[:seq_len].unsqueeze(0))
            s = dist.sample(torch.Size([self.n_samples]))
            lls_sampled.append(-seq_label_cross_entropy(lo[:seq_len].unsqueeze(0),
                                                        s.permute([1, 2, 0]),
                                                        ma[:seq_len].view(1, seq_len, 1)).cpu())
        return torch.cat(lls_orig), torch.cat(lls_sampled)

    @torch.inference_mode()
    def _get_score_impl(self, text: List[str]) -> List[float]:
        encoding = tokenize_sequences(text, self.base_tokenizer, self.base_model.device, 512)
        verbose_msg = 'Estimating log likelihoods' if self.verbose else None
        ll_orig, ll_samples = zip(*(self._proc_batch(*b) for b in model_batch_forward(
            self.base_model, encoding, self.batch_size, verbose_msg)))

        ll_orig = torch.cat(ll_orig)
        ll_samples = torch.cat(ll_samples)
        ll_samples_std = ll_samples.std(-1, correction=1)
        ll_samples = ll_samples.mean(-1)
        return (ll_orig.squeeze(-1) - ll_samples) / ll_samples_std
