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

import torch
import torch.nn.functional as F
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig


__all__ = [
    'AutoModelClsType',
    'TorchDeviceMapType',
    'seq_cross_entropy',
    'seq_label_log_rank',
    'seq_label_cross_entropy',
    'load_model',
    'batch_seq_log_likelihood',
    'model_batch_forward',
    'tokenize_sequences',
]

# noinspection PyProtectedMember
AutoModelClsType = t.Type[transformers.models.auto.auto_factory._BaseAutoModelClass]
TorchDeviceMapType = t.Union[str, t.Dict[str, t.Union[int, str, torch.device]], int, torch.device]


def load_model(model_path_or_name,
               task_type: t.Literal['SEQ_CLS', 'CAUSAL_LM'],
               flash_attn=False,
               quantization_bits: t.Optional[t.Literal[4, 8]] = None,
               output_loading_info=False,
               tokenizer_max_length=None,
               add_eos_token=True,
               **model_kwargs):
    """
    Load a pretrained transformer model for sequence classification or causal language modelling.

    :param model_path_or_name: path to model or Huggingface name
    :param task_type: task type (either ``'SEQ_CLS'`` or ``'CAUSAL_LM'``)
    :param flash_attn: use Flash Attention 2.0
    :param quantization_bits: quantize model to 4 or 8 bits
    :param output_loading_info: output loading info together with model
    :param tokenizer_max_length: truncate sequences in tokenizer
    :param add_eos_token: add EOS token to sequences
    :param model_kwargs: additional model arguments
    :return: tuple of (model, tokenizer) or ((model, loading info), tokenizer)
    """

    if task_type == 'SEQ_CLS':
        autocls = AutoModelForSequenceClassification
    elif task_type == 'CAUSAL_LM':
        autocls = AutoModelForCausalLM
    else:
        raise ValueError(f'Unsupported task type: {task_type}')

    model = autocls.from_pretrained(
        model_path_or_name,
        attn_implementation='flash_attention_2' if flash_attn else 'eager',
        torch_dtype=torch.bfloat16,
        output_loading_info=output_loading_info,
        quantization_config=BitsAndBytesConfig(
            **{f'load_in_{quantization_bits}bit': True},
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4'
        ) if quantization_bits else None,
        **model_kwargs
    )
    load_info = None
    if output_loading_info:
        model, load_info = model

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name,
                                              add_eos_token=add_eos_token,
                                              max_length=tokenizer_max_length)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    if load_info:
        return (model, load_info), tokenizer
    return model, tokenizer


def tokenize_sequences(batch: t.Union[str, t.Iterable[str]],
                       tokenizer: transformers.PreTrainedTokenizerBase,
                       device: t.Union[str, torch.device] = None,
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


def seq_label_cross_entropy(logits: torch.Tensor, labels: torch.Tensor,
                            mask: torch.Tensor, shift: bool = True, aggregate: bool = True) -> torch.Tensor:
    """
    Calculate sequence cross-entropy values between a batch of predicted next-token logits
    and a batch of truth token ids.

    If ``shift`` is ``True``, ``logits`` and ``labels`` will be shifted by one to match.

    :param logits: next-token logits
    :param labels: (current) token labels as class indices
    :param mask: padding mask
    :param shift: shift next-token logits and labels to match
    :param aggregate: aggregate token likelihoods into single mean values
    :return: average token cross-entropy values
    """
    squeeze = False
    if len(labels.shape) == 2:
        labels = labels.unsqueeze(-1)
        mask = mask.unsqueeze(-1)
        squeeze = True

    if shift:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:, :].contiguous()
        mask = mask[..., 1:, :].contiguous()

    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    if not aggregate:
        return -log_likelihood * mask
    ll = -(log_likelihood * mask).sum(dim=1) / mask.sum(dim=1)
    return ll.squeeze(-1) if squeeze else ll


def seq_label_log_rank(logits: torch.Tensor, labels: torch.Tensor,
                       mask: torch.Tensor, shift: bool = True, aggregate: bool = True) -> torch.Tensor:
    """
    Calculate average sequence token log rank between a batch of predicted next-token
    logits and a batch of truth token ids.

    If ``shift`` is ``True``, ``logits`` and ``labels`` will be shifted by one to match.

    :param logits: next-token logits
    :param labels: (current) token labels as class indices
    :param mask: padding mask
    :param shift: shift next-token logits and labels to match
    :param aggregate: aggreate per-token ranks into average sequence value
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
    lr = torch.log(matches + 1)
    if not aggregate:
        return lr

    return lr.sum(-1) / mask.sum(-1)


def batch_seq_log_likelihood(model: transformers.PreTrainedModel,
                             encoding: transformers.BatchEncoding,
                             batch_size: t.Optional[int] = None,
                             verbose: bool = False,
                             aggregate=True) -> torch.Tensor:
    """
    Calculate average sequence negative log loss / model log perplexity on a batch of input
    sequences given a causal language model.

    Likelihood estimations are performed in mini batches of size ``batch_size <= len(input batch size)``
    on the model's GPU device.

    :param model: causal LM model
    :param encoding: input encoding
    :param batch_size: mini batch size
    :param verbose: show progress bar
    :param aggregate: aggregate token likelihoods into single mean values
    :return: per-token log likelihood according to the model
    """
    # Simply return forward loss if only one sequence given
    if aggregate and encoding.input_ids.shape[0] == 1:
        return model(**encoding, labels=encoding.input_ids).loss.unsqueeze(0)

    verbose_msg = 'Estimating log likelihoods' if verbose else None
    ce_vals = [seq_label_cross_entropy(lo, la, ma, aggregate=aggregate)
               for lo, la, ma in model_batch_forward(model, encoding, batch_size, verbose_msg)]
    return torch.cat(ce_vals)


def model_batch_forward(model: transformers.PreTrainedModel,
                        encoding: transformers.BatchEncoding,
                        batch_size: t.Optional[int] = None,
                        verbose_msg: str = None) -> t.Iterable[t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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
