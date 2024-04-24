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

from typing import Dict, Iterable, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

__all__ = [
    'AutoModelClsType',
    'TorchDeviceMapType',
    'seq_cross_entropy',
    'seq_label_log_rank',
    'seq_label_cross_entropy',
    'load_model',
    'load_tokenizer',
    'batch_seq_log_likelihood',
    'model_batch_forward',
    'tokenize_sequences',
]

AutoModelClsType = Type[transformers.models.auto.auto_factory._BaseAutoModelClass]
TorchDeviceMapType = Union[str, Dict[str, Union[int, str, torch.device]], int, torch.device]


@torch.inference_mode()
def load_model(model_name: str,
               device_map: TorchDeviceMapType,
               auto_cls: AutoModelClsType = AutoModelForCausalLM,
               use_flash_attn=False,
               quantization_bits=None,
               trust_remote_code=False,
               torch_dtype: torch.dtype = torch.bfloat16,
               **additional_args) -> transformers.PreTrainedModel:

    model_args = {
        'trust_remote_code': trust_remote_code,
        'torch_dtype': torch_dtype,
        **additional_args
    }

    if use_flash_attn:
        model_args.update({'attn_implementation': 'flash_attention_2'})

    if quantization_bits:
        model_args.update({
            'quantization_config': BitsAndBytesConfig(**{
                f'load_in_{quantization_bits}bit': True,
                f'bnb_{quantization_bits}bit_compute_dtype': torch.bfloat16
            })
        })

    model = auto_cls.from_pretrained(model_name, device_map=device_map, **model_args)
    model.eval()
    return model


@torch.inference_mode()
def load_tokenizer(model_name: str, **tokenizer_args) -> transformers.PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@torch.inference_mode()
def tokenize_sequences(batch: Union[str, Iterable[str]],
                       tokenizer: transformers.PreTrainedTokenizerBase,
                       device: Union[str, torch.device] = None,
                       max_length: int = None,
                       return_tensors='pt',
                       **additional_args) -> transformers.BatchEncoding:
    batch = [batch] if isinstance(batch, str) else batch
    args = dict(
        return_tensors=return_tensors,
        padding='longest' if len(batch) > 1 else False,
        truncation=max_length is not None,
        max_length=max_length,
        return_token_type_ids=False,
    )
    args.update(additional_args)
    encodings = tokenizer(batch, **args)
    if device and return_tensors == 'pt':
        return encodings.to(device)
    return encodings


@torch.inference_mode()
def seq_cross_entropy(p_logits: torch.Tensor, q_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Calculate cross entropy between two batches of sequences of logit distributions.

    :param p_logits: "true" logits
    :param q_logits: predicted logits
    :param mask: padding mask
    :return: per-token cross entropy
    """
    _, seq_length, vocab_size = p_logits.shape
    p_prob = F.softmax(p_logits, -1).view(-1, vocab_size)
    q_logits = q_logits.view(-1, vocab_size)
    ce = F.cross_entropy(input=q_logits, target=p_prob, reduction='none').view(-1, seq_length)
    return (ce * mask).sum(-1) / mask.sum(-1)


@torch.inference_mode()
def seq_label_cross_entropy(logits: torch.Tensor, labels: torch.Tensor,
                            mask: torch.Tensor, shift: bool = True) -> torch.Tensor:
    """
    Calculate sequence cross-entropy values between a batch of predicted next-token logits
    and a batch of truth token ids.

    If ``shift`` is ``True``, ``logits`` and ``labels`` will be shifted by one to match.

    :param logits: next-token logits
    :param labels: (current) token labels as class indices
    :param mask: padding mask
    :param shift: shift next-token logits and labels to match
    :return: average token cross-entropy values
    """
    if shift:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        mask = mask[..., 1:].contiguous()
    mask = mask.bool()
    labels = labels.where(mask, -100)

    _, seq_length, vocab_size = logits.shape
    ll = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), reduction='none', ignore_index=-100)
    return ll.view(-1, seq_length).sum(-1) / mask.sum(-1)


@torch.inference_mode()
def seq_label_log_rank(logits: torch.Tensor, labels: torch.Tensor,
                       mask: torch.Tensor, shift: bool = True) -> torch.Tensor:
    """
    Calculate average sequence token log rank between a batch of predicted next-token
    logits and a batch of truth token ids.

    If ``shift`` is ``True``, ``logits`` and ``labels`` will be shifted by one to match.

    :param logits: next-token logits
    :param labels: (current) token labels as class indices
    :param mask: padding mask
    :param shift: shift next-token logits and labels to match
    :return: average token rank when sorted by likelihood
    """
    if shift:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        mask = mask[..., 1:].contiguous()
    mask = mask.bool()

    matches = logits.argsort(-1, descending=True)
    matches = (matches == labels.unsqueeze(-1)).nonzero()
    matches = matches.where(mask.view(-1, 1), 0)
    matches = matches[..., -1].view(logits.shape[:2])

    return torch.log(matches + 1).sum(-1) / mask.sum(-1)


@torch.inference_mode()
def batch_seq_log_likelihood(model: transformers.PreTrainedModel,
                             encoding: transformers.BatchEncoding,
                             batch_size: Optional[int] = None,
                             verbose: bool = False) -> torch.Tensor:
    """
    Calculate average sequence negative log loss / model log perplexity on a batch of input
    sequences given a causal language model.

    Likelihood estimations are performed in mini batches of size ``batch_size <= len(input batch size)``
    on the model's GPU device.

    :param model: causal LM model
    :param encoding: input encoding
    :param batch_size: mini batch size
    :param verbose: show progress bar
    :return: per-token log likelihood according to the model
    """
    # Simply return forward loss if only one sequence given
    if encoding.input_ids.shape[0] == 1:
        return model(**encoding, labels=encoding.input_ids).loss.cpu().unsqueeze(0)

    verbose_msg = 'Estimating log likelihoods' if verbose else None
    ce_vals = [seq_label_cross_entropy(lo, la, ma).cpu()
               for lo, la, ma in model_batch_forward(model, encoding, batch_size, verbose_msg)]
    return torch.cat(ce_vals)


@torch.inference_mode()
def model_batch_forward(model: transformers.PreTrainedModel,
                        encoding: transformers.BatchEncoding,
                        batch_size: Optional[int] = None,
                        verbose_msg: str = None) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Batched model forward pass on input data.

    :param model: causal LM model
    :param encoding: input encoding
    :param batch_size: batch size (``None`` for single batch)
    :param verbose_msg: show progress bar with message during batched model prediction
    :return: iterator of batched output logits, input labels, attention mask
    """
    batch_size = batch_size or len(encoding.input_ids)
    batch_it = range(0, len(encoding.input_ids), batch_size)
    if verbose_msg:
        batch_it = tqdm(batch_it, desc=verbose_msg, leave=False,
                        total=(len(encoding.input_ids) + 1) // batch_size, unit=' batch')
    for b in batch_it:
        yield (model(**{k: v[b:b + batch_size] for k, v in encoding.items()}).logits,
               encoding.input_ids[b:b + batch_size], encoding.attention_mask[b:b + batch_size])
