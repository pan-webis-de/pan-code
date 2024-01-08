#!/usr/bin/env python3
from transformers import (
    NllbTokenizerFast,
    M2M100ForConditionalGeneration,
    BartTokenizerFast,
    BartForConditionalGeneration,
)
import argparse
import logging
from typing import List, Union
import torch
import json
from tqdm import trange


def get_model(type: str):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if type == "translator":
        logging.info("Loading translation model: \
                     'facebook/nllb-200-distilled-600M'")

        model = (
            M2M100ForConditionalGeneration.from_pretrained(
                "facebook/nllb-200-distilled-600M"
            )
            .eval()
            .to(device)
        )
        tokenizer = NllbTokenizerFast.from_pretrained(
            "facebook/nllb-200-distilled-600M"
        )
    elif type == "paraphraser":
        logging.info("Loading detoxification model: \
                     's-nlp/bart-base-detox'")
        model = (
            BartForConditionalGeneration.from_pretrained("s-nlp/bart-base-detox")
            .eval()
            .to(device)
        )
        tokenizer = BartTokenizerFast.from_pretrained("s-nlp/bart-base-detox")
    else:
        raise ValueError("Invalid type choice")
    return model, tokenizer


def translate(
    inputs: List[str],
    model: M2M100ForConditionalGeneration,
    tokenizer: NllbTokenizerFast,
    batch_size: int = 32,
    src_lang_id: str = "rus_Cyrl",
    tgt_lang_id: str = "eng_Latn",
) -> List[str]:
    tokenizer.src_lang = src_lang_id
    tokenizer.tgt_lang = tgt_lang_id

    translated_outputs = []
    logging.info(
        f"Translating from {tokenizer.src_lang} to {tokenizer.tgt_lang}."
    )
    for i in trange(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        inputs_tokenized = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs_tokenized)
        translated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated_outputs.extend(translated_batch)
    return translated_outputs


def paraphrase_batch(
    texts: List[str],
    model,
    tokenizer,
    n: Union[None, int] = None,
    max_length: str = "auto",
    beams: int = 5,
):
    logging.info("Detoxifying texts")
    batch_size = 32

    paraphrased_texts = []
    for i in trange(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)["input_ids"].to(
            model.device
        )

        if max_length == "auto":
            max_length = inputs.shape[1] + 10

        result = model.generate(
            inputs,
            num_return_sequences=n or 1,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=10.0,
            max_length=max_length,
            min_length=int(0.5 * max_length),
            num_beams=beams,
        )
        paraphrased_texts.extend(
            tokenizer.decode(r, skip_special_tokens=True) for r in result
        )
    if not n and isinstance(texts, str):
        return paraphrased_texts[0]
    return paraphrased_texts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backtranslation baseline for PAN 2024 text detox task "
            "that performs detox: translate input (toxic) text from "
            "source language into pivot language (English), detox it "
            "and then translate detoxified text back into source language"
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=argparse.FileType("rb"),
        help="The input file, expected to be a `.jsonl` file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=argparse.FileType("w", encoding="utf-8"),
        help="The output file, will create a `.jsonl` file.",
    )
    parser.add_argument(
        "--src_lang_id",
        required=True,
        choices=["am", "es", "ru", "uk", "en", "zh", "ar", "hi", "de"],
        help="Language id (should be one of 'am', 'es', 'ru', 'uk', 'en', 'zh', 'ar', 'hi', 'de')",
    )
    parser.add_argument(
        "--batch_size",
        required=False,
        default=32,
        type=int,
        help="Batch size for translation and detoxification (default is 32).",
    )

    args = parser.parse_args()

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

    logging.basicConfig(level=logging.INFO)

    tr_model, tr_tokenizer = get_model(type="translator")

    sources = [json.loads(line)["text"] for line in args.input]

    print(args)
    # Check if the source language is English
    if args.src_lang_id == "en":
        logging.info("Source language is English. Performing only detox.")

        tst_model, tst_tokenizer = get_model(type="paraphraser")
        detoxified = paraphrase_batch(
            texts=sources, model=tst_model, tokenizer=tst_tokenizer
        )

        for line in detoxified:
            args.output.write(json.dumps(line, ensure_ascii=False))
            args.output.write("\n")
    else:
        logging.info("Running")
        translated_sources = translate(
            inputs=sources,
            model=tr_model,
            tokenizer=tr_tokenizer,
            batch_size=args.batch_size,
            src_lang_id=lang_id_mapping[args.src_lang_id],
            tgt_lang_id=lang_id_mapping["en"],
        )

        del tr_model
        del tr_tokenizer

        tst_model, tst_tokenizer = get_model(type="paraphraser")

        detoxified = paraphrase_batch(
            texts=translated_sources, model=tst_model, tokenizer=tst_tokenizer
        )

        del tst_tokenizer
        del tst_model

        tr_model, tr_tokenizer = get_model(type="translator")

        backtranslated_sources = translate(
            inputs=detoxified,
            model=tr_model,
            tokenizer=tr_tokenizer,
            batch_size=args.batch_size,
            src_lang_id=lang_id_mapping["en"],
            tgt_lang_id=lang_id_mapping[args.src_lang_id],
        )

        for line in backtranslated_sources:
            args.output.write(json.dumps(line, ensure_ascii=False))
            args.output.write("\n")

if __name__ == "__main__":
    main()