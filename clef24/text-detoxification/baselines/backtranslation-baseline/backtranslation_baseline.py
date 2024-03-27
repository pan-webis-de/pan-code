#!/usr/bin/env python3

import argparse
import json
import torch
import logging
from typing import List, Union, Tuple, Dict
from tqdm import tqdm
from transformers import (
    M2M100ForConditionalGeneration,
    NllbTokenizerFast,
    BartTokenizerFast,
    T5TokenizerFast,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
    PreTrainedTokenizerFast,
    PreTrainedModel,
)


def get_model(type: str) -> Tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """
    Returns a pre-trained model and tokenizer based on the specified type.

    Args:
        type (str): The type of model to retrieve.
        Valid options are "translator", "en_detoxifier", and "ru_detoxifier".

    Returns:
        (PreTrainedModel, PreTrainedTokenizer)
    Raises:
        ValueError: If an invalid type choice is provided.

    Examples:
        model, tokenizer = get_model("translator")
        model, tokenizer = get_model("en_detoxifier")
        model, tokenizer = get_model("ru_detoxifier")
    """
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model_types: Dict[str, Tuple[str, PreTrainedModel, PreTrainedTokenizerFast]] = {
        "translator": (
            "facebook/nllb-200-distilled-600M",
            M2M100ForConditionalGeneration,
            NllbTokenizerFast,
        ),
        "en_detoxifier": (
            "s-nlp/bart-base-detox",
            BartForConditionalGeneration,
            BartTokenizerFast,
        ),
        "ru_detoxifier": (
            "s-nlp/ruT5-base-detox",
            T5ForConditionalGeneration,
            T5TokenizerFast,
        ),
    }

    if type not in model_types:
        raise ValueError("Invalid type choice")

    model_name, ModelClass, TokenizerClass = model_types[type]

    logging.info(f"Loading {type} model: {model_name}")

    model = ModelClass.from_pretrained(model_name).eval().to(device)
    tokenizer = TokenizerClass.from_pretrained(model_name)

    return model, tokenizer


def translate_batch(
    texts: List[str],
    model: M2M100ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 32,
) -> List[str]:
    """
    Translate a batch of texts.

    Args:
        texts (List[str]): The list of texts to translate.
        model (M2M100ForConditionalGeneration): The translation model.
        tokenizer (PreTrainedTokenizerFast): The tokenizer for the translation model.
        batch_size (int, optional): The batch size for translation. Defaults to 32.

    Returns:
        List[str]: The translated texts.
    """
    translations = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i : i + batch_size]
        batch_translated = model.generate(
            **tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device),
            forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang],
        )
        translations.extend(
            tokenizer.decode(tokens, skip_special_tokens=True)
            for tokens in batch_translated
        )
    return translations


def detoxify_batch(
    texts: List[str],
    model: Union[BartForConditionalGeneration, T5ForConditionalGeneration],
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 32,
) -> List[str]:
    """
    Detoxify a batch of texts.

    Args:
        texts (List[str]): The list of texts to detoxify.
        model (Union[BartForConditionalGeneration, T5ForConditionalGeneration]): The detoxification model.
        tokenizer (PreTrainedTokenizerFast): The tokenizer for the detoxification model.
        batch_size (int, optional): The batch size for detoxification. Defaults to 32.

    Returns:
        List[str]: The detoxified texts.
    """
    detoxified = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Detoxifying"):
        batch = texts[i : i + batch_size]
        batch_detoxified = model.generate(
            **tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(
                model.device
            )
        )
        detoxified.extend(
            tokenizer.decode(tokens, skip_special_tokens=True)
            for tokens in batch_detoxified
        )
    return detoxified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtranslation baseline for PAN 2024 text detoxification task"
        "that performs detox: translate input (toxic) text from "
        "source language into pivot language (English), detox it "
        "and then translate detoxified text back into source language"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=argparse.FileType("r", encoding="utf-8"),
        help="Input JSONL file containing text examples.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=argparse.FileType("w", encoding="utf-8"),
        help="Output JSONL file containing detoxified text examples.",
    )
    parser.add_argument(
        "--language",
        required=True,
        type=str,
        choices=["am", "es", "ru", "uk", "en", "zh", "ar", "hi", "de"],
        help="Language of the input data. Should be one of"
        "['am', 'es', 'ru', 'uk', 'en', 'zh', 'ar', 'hi', 'de']",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for translation and detoxification.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    lang_id_mapping = {
        "ru": "rus_Cyrl",
        "en": "eng_Latn",
        "am": "amh_Ethi",
        "es": "spa_Latn",
        "uk": "ukr_Cyrl",
        "zh": "zho_Hans",
        "ar": "arb_Arab",
        "hi": "hin_Deva",
        "de": "deu_Latn",
    }

    inputs = [json.loads(line) for line in args.input]
    texts = [entry["text"] for entry in inputs]
    doc_ids = [entry["id"] for entry in inputs]

    if args.language == "en":
        model, tokenizer = get_model("en_detoxifier")
        detoxified_texts = detoxify_batch(texts, model, tokenizer, args.batch_size)
    elif args.language == "ru":
        model, tokenizer = get_model("ru_detoxifier")
        detoxified_texts = detoxify_batch(texts, model, tokenizer, args.batch_size)
    else:
        model, tokenizer = get_model("translator")

        tokenizer.src_lang = lang_id_mapping[args.language]
        tokenizer.tgt_lang = lang_id_mapping["en"]
        texts = translate_batch(texts, model, tokenizer, args.batch_size)

        model, tokenizer = get_model("en_detoxifier")
        detoxified_texts = detoxify_batch(texts, model, tokenizer, args.batch_size)

        model, tokenizer = get_model("translator")
        tokenizer.tgt_lang = lang_id_mapping[args.language]
        tokenizer.src_lang = lang_id_mapping["en"]
        detoxified_texts = translate_batch(texts, model, tokenizer, args.batch_size)

    for doc_id, text in zip(doc_ids, detoxified_texts):
        args.output.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False))
        args.output.write("\n")


if __name__ == "__main__":
    main()
