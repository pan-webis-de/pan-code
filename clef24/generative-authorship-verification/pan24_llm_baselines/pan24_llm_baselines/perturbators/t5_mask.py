import random
from typing import List, Tuple, Union

from more_itertools import batched
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM

from pan24_llm_baselines.util import *
from pan24_llm_baselines.perturbators.perturbator_base import PerturbatorBase

__all__ = ['T5MaskPerturbator']


class T5MaskPerturbator(PerturbatorBase):
    def __init__(self,
                 span_length=2,
                 mask_pct=0.3,
                 buffer_size=1,
                 model_name='t5-large',
                 device: TorchDeviceMapType = 'auto',
                 quantization_bits=None,
                 use_flash_attn=False,
                 max_tokens=512,
                 batch_size=10,
                 verbose=True,
                 **model_args):
        """
        :param span_length: length of token spans to mask out
        :param mask_pct: target percentage of tokens to mask out
        :param buffer_size: minimum buffer around mask tokens
        :param model_name: T5 model variant
        :param device: model device
        :param quantization_bits: quantize model
        :param use_flash_attn: use flash attention
        :param max_tokens: max number of tokens the model can handle
        :param batch_size: perturbation variant batch size
        :param verbose: show progress bar
        :param model_args: additional model arguments
        """
        self.span_length = span_length
        self.mask_pct = mask_pct
        self.mask_buffer_size = buffer_size
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.verbose = verbose

        self.model = load_model(model_name,
                                device_map=device,
                                auto_cls=AutoModelForSeq2SeqLM,
                                quantization_bits=quantization_bits,
                                use_flash_attn=use_flash_attn,
                                **model_args)
        self.tokenizer = load_tokenizer(model_name, model_max_length=max_tokens)

    def _mask_tokens(self, text) -> Tuple[str, int]:
        """
        Randomly replace (white space) token spans with masks.

        :param text: input text
        :return: perturbed text with mask tokens, number of mask tokens
        """
        text = text.strip().split(' ')
        text_len = len(text)
        n_spans_target = int(self.mask_pct * text_len / (self.span_length + self.mask_buffer_size * 2) + 1)
        n_spans_target = min(n_spans_target, 99)    # T5 has a max of 100 sentinel tokens by default

        spans = set()
        del_idx = set()
        while len(spans) < n_spans_target:
            start = random.randint(0, text_len - self.span_length - 1)
            end = start + self.span_length
            if None not in text[max(0, start - self.mask_buffer_size):end + self.mask_buffer_size]:
                # token_ids[start:end] = mask_id
                spans.add(start)
                for i in range(start, end):
                    text[i] = None
                    if i > start:
                        del_idx.add(i)

        spans = sorted(spans)
        for i, idx in enumerate(spans):
            # count down from <extra_id_0> to <extra_id_{len(spans)}>
            text[idx] = f'<extra_id_{i}>'

        # Delete remainder of the spans
        text = ' '.join(t for i, t in enumerate(text) if i not in del_idx)
        return text, len(spans)

    def _generate_fills(self, masked_texts: List[str], num_masks: Union[List[int], torch.Tensor]) -> List[List[str]]:
        """
        Generate a new texts from batch of masked token sequence.
        
        :param masked_texts: batch of masked texts
        :param num_masks: number of masks for each text
        :return: generated sentinel fills
        """

        token_ids = tokenize_sequences(masked_texts, self.tokenizer, max_length=self.max_tokens).input_ids
        stop_ids = torch.tensor([self.tokenizer.encode(f'<extra_id_{n}>')[0] for n in num_masks])
        outputs = self.model.generate(
            input_ids=token_ids.to(self.model.device),
            attention_mask=torch.ones(token_ids.shape, device=self.model.device),
            max_length=min(512, max(num_masks) * 8),
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=stop_ids,
            pad_token_id=self.tokenizer.pad_token_id).cpu()

        fills = []
        for i, output in enumerate(outputs):
            if len(token_ids) > 1:
                # Truncate padded </s> tokens
                output = output[output != self.tokenizer.eos_token_id]

            mask_pos = torch.argwhere(output >= stop_ids[i])
            o = []
            for j in range(len(mask_pos) - 1):
                ids = output[mask_pos[j] + 1:mask_pos[j + 1]]
                o.append(self.tokenizer.decode(ids, skip_special_tokens=True).strip())
            fills.append(o)
        return fills

    def _apply_fills(self, masked_text: List[str], fills: List[List[str]]) -> List[str]:
        """
        Back-substitute masks in original texts with generated fills.

        :param masked_text: batch of original masked texts
        :param fills: batch of fills
        :return: back-substituted text
        """

        texts = []
        for m, f in zip(masked_text, fills):
            m = m.split(' ')
            f_idx = 0
            for i, t in enumerate(m):
                if t == f'<extra_id_{f_idx}>':
                    m[i] = f[f_idx] if f_idx < len(f) else ''
                    f_idx += 1
            texts.append(' '.join(m).strip())
        return texts

    def _perturb_impl(self, text: List[str], n_variants: int) -> List[str]:
        batch_size = self.batch_size if self.batch_size else len(text)
        n_iter = (len(text) * n_variants + 1) // batch_size
        batch_it = batched((t for t in text for _ in range(n_variants)), batch_size)
        if self.verbose:
            batch_it = tqdm(batch_it, desc='Generating perturbations', leave=False, unit=' batch', total=n_iter)

        perturbed = []
        for b in batch_it:
            masked = []
            n_masks = []
            for t in b:
                t, n = self._mask_tokens(t)
                masked.append(t)
                n_masks.append(n)
            fills = self._generate_fills(masked, n_masks)
            perturbed.extend(self._apply_fills(masked, fills))

        return perturbed
