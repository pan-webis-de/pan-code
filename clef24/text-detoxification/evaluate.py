#!/usr/bin/env python3

import argparse
from typing import Optional, Type, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(
        model_name: Optional[str] = None,
        model: Optional[AutoModelForSequenceClassification] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        model_class: Optional[Type[AutoModelForSequenceClassification]] = AutoModelForSequenceClassification
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    if model is None:
        if model_name is None:
            raise ValueError('Either model or model_name should be provided')

        if model_class is None:
            raise ValueError('model_class should be provided')

        model = model_class.from_pretrained(model_name)

        if torch.cuda.is_available():
            model.cuda()

    if tokenizer is None:
        if model_name is None:
            raise ValueError('Either tokenizer or model_name should be provided')

        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description='Text Detoxification Evaluation')
    parser.add_argument('-s', '--style-model', type=str, required=True)
    parser.add_argument('-m', '--meaning-model', type=str, required=True)
    parser.add_argument('-c', '--cola-model', type=str, required=True)
    parser.add_argument('-g', '--gold', type=argparse.FileType('rb'), required=True)
    parser.add_argument('pred', type=argparse.FileType('rb'))

    args = parser.parse_args()

    style_model, style_tokenizer = load_model(args.style_model)
    meaning_model, meaning_tokenizer = load_model(args.meaning_model)
    cola_model, cola_tolenizer = load_model(args.cola_model)


if __name__ == '__main__':
    main()
