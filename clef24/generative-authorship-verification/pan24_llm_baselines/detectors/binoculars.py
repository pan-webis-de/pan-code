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
import transformers

from pan24_llm_baselines.detectors.detector_base import DetectorBase
from pan24_llm_baselines.util import *

__all__ = ['Binoculars']


class Binoculars(DetectorBase):
    """
    Binoculars LLM detector.

    This is a refactored implementation of the original: https://github.com/ahans30/Binoculars/tree/main

    References:
    ===========
        Hans, Abhimanyu, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi,
        Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2024.
        “Spotting LLMs with Binoculars: Zero-Shot Detection of Machine-Generated
        Text.” arXiv [Cs.CL]. arXiv. http://arxiv.org/abs/2401.12070.
    """
    # Selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
    BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
    BINOCULARS_FPR_THRESHOLD = 0.8536432310785527       # optimized for low-fpr [chosen at 0.01%]

    def __init__(self,
                 mode: Literal['low-fpr', 'accuracy'] = 'low-fpr',
                 observer_name_or_path='tiiuae/falcon-7b',
                 performer_name_or_path='tiiuae/falcon-7b-instruct',
                 device1: TorchDeviceMapType = 'auto',
                 device2: TorchDeviceMapType = 'auto',
                 max_length=512,
                 use_flash_attn=False,
                 quantization_bits=None,
                 **model_args):
        """
        :param mode: prediction mode
        :param observer_name_or_path: observer model
        :param performer_name_or_path: performer model
        :param device1: observer device
        :param device2: performer device
        :param max_length: max number of tokens to analyze
        :param use_flash_attn: use flash attention
        :param quantization_bits: quantize model
        :param model_args: additional model args
        """

        self.scoring_mode = mode
        self.observer_model = load_model(
            observer_name_or_path,
            device_map=device1,
            use_flash_attn=use_flash_attn,
            quantization_bits=quantization_bits,
            **model_args)
        self.tokenizer = load_tokenizer(observer_name_or_path, max_length=max_length)

        self.performer_model = load_model(
            performer_name_or_path,
            device_map=device2,
            use_flash_attn=use_flash_attn,
            quantization_bits=quantization_bits,
            **model_args)
        perf_tokenizer = load_tokenizer(performer_name_or_path)

        if not hasattr(self.tokenizer, 'vocab') or self.tokenizer.vocab != perf_tokenizer.vocab:
            raise ValueError(f'Incompatible tokenizers for {observer_name_or_path} and {performer_name_or_path}.')

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> Tuple[torch.Tensor, torch.Tensor]:
        observer_logits = self.observer_model(**encodings.to(self.observer_model.device)).logits
        performer_logits = self.performer_model(**encodings.to(self.performer_model.device)).logits

        if next(self.observer_model.parameters()).is_cuda:
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def _normalize_scores(self, scores):
        return torch.sigmoid(-10 * self.threshold * (scores.to(torch.float64) - self.threshold))

    @torch.inference_mode()
    def _get_score_impl(self, text: List[str]) -> torch.Tensor:
        encodings = tokenize_sequences(text, self.tokenizer, self.observer_model.device, max_length=512)
        observer_logits, performer_logits = self._get_logits(encodings)
        log_ppl = seq_label_cross_entropy(performer_logits, encodings.input_ids, encodings.attention_mask)
        x_ppl = seq_cross_entropy(observer_logits,
                                  performer_logits.to(self.observer_model.device),
                                  encodings.attention_mask)
        return log_ppl / x_ppl

    @property
    def threshold(self) -> float:
        if self.scoring_mode == 'low-fpr':
            return self.BINOCULARS_FPR_THRESHOLD
        if self.scoring_mode == 'accuracy':
            return self.BINOCULARS_ACCURACY_THRESHOLD
        raise ValueError(f'Invalid scoring mode: {self.scoring_mode}')

    @torch.inference_mode()
    def _predict_impl(self, text: List[str]) -> torch.Tensor:
        return self._get_score_impl(text) < self.threshold
