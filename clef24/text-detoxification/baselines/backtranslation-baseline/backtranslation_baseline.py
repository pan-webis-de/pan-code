#!/usr/bin/env python3
from transformers import (
    NllbTokenizerFast,
    M2M100ForConditionalGeneration,
    BartTokenizerFast,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
import argparse
import logging
from typing import List, Union, Dict
import torch
import json
from tqdm import trange, tqdm
from collections import defaultdict


def group_by_language(
    inputs: List[Dict],
) -> Dict[str, List[Dict[str, Union[str, int]]]]:
    """
    Groups examples by language.

    Args:
        inputs (List[Dict]): List of examples.

    Returns:
        Dict[str, List[Dict[str, Union[str, int]]]]: Dict where keys are lang codes and values are lists of dicts with text, language, and original_index.
    """

    grouped_data = defaultdict(list)
    for i, example in enumerate(inputs):
        lang = example.get("language", "unknown")
        grouped_data[lang].append(
            {
                "text": example["text"],
                "language": example.get("language", "unknown"),
                "original_index": i,
            }
        )
    return grouped_data


def get_model(
    type: str,
) -> (PreTrainedModel, PreTrainedTokenizerFast):
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
    if type == "translator":
        logging.info(
            "Loading translation model: \
                     'facebook/nllb-200-distilled-600M'"
        )

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
    elif type == "en_detoxifier":
        logging.info(
            "Loading detoxification model: \
                     's-nlp/bart-base-detox'"
        )
        model = (
            BartForConditionalGeneration.from_pretrained("s-nlp/bart-base-detox")
            .eval()
            .to(device)
        )
        tokenizer = BartTokenizerFast.from_pretrained("s-nlp/bart-base-detox")

    elif type == "ru_detoxifier":
        logging.info(
            "Loading detoxification model: \
                     's-nlp/bart-base-detox'"
        )
        model = (
            T5ForConditionalGeneration.from_pretrained("s-nlp/ruT5-base-detox")
            .eval()
            .to(device)
        )
        tokenizer = T5TokenizerFast.from_pretrained("s-nlp/ruT5-base-detox")
    else:
        raise ValueError("Invalid type choice")
    return model, tokenizer


def translate(
    inputs: List[str],
    model: M2M100ForConditionalGeneration,
    tokenizer: NllbTokenizerFast,
    # batch_size: int = 32,
    src_lang_ids: List[str] = ["rus_Cyrl"],
    tgt_lang_ids: List[str] = ["eng_Latn"],
) -> List[str]:
    """
    Translates a list of input sentences from the src_lang_id
    to the tgt_lang_id using (model, tokenizer).

    Args:
        inputs (List[str]): The list of input sentences to be translated.
        model (M2M100ForConditionalGeneration): The pretrained translation model.
        tokenizer (NllbTokenizerFast): The tokenizer used for tokenizing the input sentences.
        batch_size (int, optional): The batch size for translation. Defaults to 32.
        src_lang_id (str, optional): The source language ID. Defaults to "rus_Cyrl".
        tgt_lang_id (str, optional): The target language ID. Defaults to "eng_Latn".

    Returns:
        List[str]: The list of translated sentences.

    Examples:
        inputs = ["Привет, как дела?", "Спасибо за помощь!"]
        translated = translate(inputs, model, tokenizer, 1, "rus_Cyrl", "eng_Latn")
        print(translated)
        # Output: ["Hello, how are you?", "Thanks for the help!"]
    """

    translated_outputs = []
    logging.info(
        f"Translating from {tokenizer.src_lang} to \
                  {tokenizer.tgt_lang}."
    )
    for text, src_id, tgt_id in tqdm(
        zip(inputs, src_lang_ids, tgt_lang_ids), desc="Translating"
    ):
        tokenizer.src_lang = src_id
        tokenizer.tgt_lang = tgt_id

        inputs_tokenized = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs_tokenized)
        translated = tokenizer.decode(outputs, skip_special_tokens=True)
        translated_outputs.append(translated)
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
    # parser.add_argument(
    #     "--batch_size",
    #     required=False,
    #     default=32,
    #     type=int,
    #     help="Batch size for translation and detoxification (default is 32).",
    # )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    inputs = [json.loads(line) for line in args.input]
    sources = [i["text"] for i in inputs]
    doc_ids = [i["id"] for i in inputs]
    lang_ids = [i["language"] for i in inputs]

    if all(x in ("en", "english") for x in lang_ids):
        logging.info(
            "Source language is English. Performing only detox with \
                's-nlp/bart-base-detox'"
        )

        tst_model, tst_tokenizer = get_model(type="en_detoxifier")
        detoxified = paraphrase_batch(
            texts=sources, model=tst_model, tokenizer=tst_tokenizer
        )

        for doc_id, text in zip(doc_ids, detoxified):
            args.output.write(
                json.dumps({"id": doc_id, "text": text}, ensure_ascii=False)
            )
            args.output.write("\n")

    elif all(x in ("russian", "ru") for x in lang_ids):
        logging.info(
            "Source language is Russian. Performing only detox with \
                's-nlp/ruT5-base-detox'"
        )

        tst_model, tst_tokenizer = get_model(type="ru_detoxifier")
        detoxified = paraphrase_batch(
            texts=sources, model=tst_model, tokenizer=tst_tokenizer
        )

        for doc_id, text in zip(doc_ids, detoxified):
            args.output.write(
                json.dumps({"id": doc_id, "text": text}, ensure_ascii=False)
            )
            args.output.write("\n")
    else:
        logging.info("Running backtranslation baseline")
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

        grouped_data = group_by_language(inputs=inputs)

        lang_ids = [lang_id_mapping[x["language"]] for x in grouped_data]

        tr_model, tr_tokenizer = get_model(type="translator")

        translated_sources = translate(
            inputs=sources,
            model=tr_model,
            tokenizer=tr_tokenizer,
            batch_size=args.batch_size,
            src_lang_ids=lang_ids,
            tgt_lang_ids=lang_id_mapping["en"] * len(lang_ids),
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
            src_lang_ids=lang_id_mapping["en"] * len(lang_ids),
            tgt_lang_ids=lang_ids,
        )

        for doc_id, text in zip(doc_ids, backtranslated_sources):
            args.output.write(
                json.dumps({"id": doc_id, "text": text}, ensure_ascii=False)
            )
            args.output.write("\n")


if __name__ == "__main__":
    main()
