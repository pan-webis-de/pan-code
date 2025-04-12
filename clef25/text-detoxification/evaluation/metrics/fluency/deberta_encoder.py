from typing import Optional

import comet.encoders
import torch
import transformers as tr
from comet.encoders.base import Encoder
from comet.encoders.bert import BERTEncoder
from comet.models.multitask.xcomet_metric import XCOMETMetric
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Layer


class DebertaV2LayerPatched(DebertaV2Layer):
    """
    Just like DeBERTaV2Layer, but handles the situation when attention_mask=None by creating a trivial attention mask of all ones.
    """

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
    ):
        # Handling empty attention_mask:
        if attention_mask is None:
            attention_mask = torch.ones(*hidden_states.shape[:-1]).to(
                hidden_states.device
            )

        # Everything else is the same:
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if output_attentions:
            return (layer_output, att_matrix)
        else:
            return layer_output


class DeBERTaEncoder(BERTEncoder):
    def __init__(
        self, pretrained_model: str, load_pretrained_weights: bool = True
    ) -> None:
        super(Encoder, self).__init__()
        self.tokenizer = tr.AutoTokenizer.from_pretrained(pretrained_model)
        if load_pretrained_weights:
            self.model = tr.AutoModel.from_pretrained(pretrained_model)
        else:
            self.model = tr.AutoModel.from_config(
                tr.AutoConfig.from_pretrained(pretrained_model),
            )
        self.model.encoder.output_hidden_states = True

        self.model.encoder.layer = nn.ModuleList(
            [
                DebertaV2LayerPatched(tr.AutoConfig.from_pretrained(pretrained_model))
                for _ in range(self.model.config.num_hidden_layers)
            ]
        )

    @property
    def size_separator(self):
        """Number of tokens used between two segments. For BERT is just 1 ([SEP])
        but models such as XLM-R use 2 (</s></s>)"""
        return 1

    @property
    def uses_token_type_ids(self):
        return True

    @classmethod
    def from_pretrained(
        cls, pretrained_model: str, load_pretrained_weights: bool = True
    ) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.

        Args:
            pretrained_model (str):Name of the pretrain model to be loaded.
            load_pretrained_weights (bool): If set to True loads the pretrained weights
                from Hugging Face

        Returns:
            Encoder: DeBERTaEncoder object.
        """
        return DeBERTaEncoder(pretrained_model, load_pretrained_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=kwargs.get("token_type_ids", None),
            output_hidden_states=True,
        )
        return {
            "sentemb": model_output.last_hidden_state[:, 0, :],
            "wordemb": model_output.last_hidden_state,
            "all_layers": model_output.hidden_states,
            "attention_mask": attention_mask,
        }


comet.encoders.str2encoder["DeBERTa"] = DeBERTaEncoder


class XCOMETLite(XCOMETMetric, PyTorchModelHubMixin):
    """A wrapper to push xCOMET-Lite model to huggingface."""

    def __init__(
        self,
        encoder_model="DeBERTa",
        pretrained_model="microsoft/mdeberta-v3-base",
        word_layer=8,
        validation_data=[],
        word_level_training=True,
        hidden_sizes=(3072, 1024),
        load_pretrained_weights=False,
        layer_transformation="softmax",
        *args,
        **kwargs
    ):
        super().__init__(
            encoder_model=encoder_model,
            pretrained_model=pretrained_model,
            word_layer=word_layer,
            validation_data=validation_data,
            word_level_training=word_level_training,
            hidden_sizes=hidden_sizes,
            load_pretrained_weights=load_pretrained_weights,
            layer_transformation=layer_transformation,
        )
