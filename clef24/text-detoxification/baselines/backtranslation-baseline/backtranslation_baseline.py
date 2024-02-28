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


def get_batches_by_language(inputs: List[Dict]) -> List[List]:
    """
    Splits a list of dictionaries into a list of lists, where each sublist contains
    dictionaries with the same language.

    Args:
        inputs (List[Dict]): The list of dictionaries to be split.

    Returns:
        List[List]: The list of lists containing dictionaries grouped by language.
    """

    # Create a defaultdict to store dictionaries grouped by language
    batches_by_language = defaultdict(list)

    # Group dictionaries by language
    for entry in inputs:
        language = entry["language"]
        batches_by_language[language].append(entry)

    return list(batches_by_language.values())


def group_by_language(
    inputs: List[Dict],
) -> Dict[str, List[Dict[str, Union[str, int]]]]:
    """
    Groups examples by language.

    Args:
        inputs (List[Dict]): List of examples.

    Returns:
        Dict[str, List[Dict[str, Union[str, int]]]]: \
            Dict where keys are lang \
                codes and values are lists of dicts with\
                  text, language, and original_index.
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
    batch_size: int = 32,
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
    for i in trange(0, len(inputs), batch_size):
        batch_text = inputs[i : i + batch_size]
        batch_src_ids = src_lang_ids[i : i + batch_size]
        batch_tgt_ids = tgt_lang_ids[i : i + batch_size]

        tokenizer.src_lang = batch_src_ids[0]
        tokenizer.tgt_lang = batch_tgt_ids[0]

        inputs_tokenized = tokenizer(
            batch_text, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs_tokenized)
            translated = [
                tokenizer.decode(
                    token_ids=x,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                for x in outputs.detach().cpu()
            ]
            translated_outputs.extend(translated)

    assert all(len(x) for x in translated_outputs) > 0
    assert len(translated_outputs) == len(inputs)
    return [x[:512] for x in translated_outputs]


def paraphrase_batch(
    texts: List[str],
    model,
    tokenizer,
    n: Union[None, int] = None,
    max_length: str = "auto",
    beams: int = 1,
    batch_size=32,
):
    """
    Detoxify a batch of texts by generating paraphrases using a given model and tokenizer.

    Args:
        texts (List[str]): The list of texts to be detoxified.
        model: The model used for generating paraphrases.
        tokenizer: The tokenizer used for tokenizing the texts.
        n (Union[None, int], optional): The number of paraphrases to generate for each input text. Defaults to None.
        max_length (str, optional): The maximum length of the generated paraphrases. Defaults to "auto".
        beams (int, optional): The number of beams to use during generation. Defaults to 5.

    Returns:
        List[str]: The list of detoxified paraphrases.

    Examples:
        >>> texts = ["I love coding", "Python is awesome"]
        >>> model = MyModel()
        >>> tokenizer = MyTokenizer()
        >>> paraphrase_batch(texts, model, tokenizer)
        ['I enjoy coding', 'Python is amazing']
    """
    logging.info("Detoxifying texts")

    paraphrased_texts = []
    for i in trange(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True,
        )["input_ids"].to(model.device)
        # Adjust max_length based on the length of the input sequence
        if max_length == "auto":
            max_length = inputs.shape[1] + 10
        else:
            max_length = min(max_length, inputs.shape[1] + 10)

        with torch.no_grad():
            result = model.generate(
                inputs,
                num_return_sequences=1,
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
        "--batch_size",
        required=False,
        type=int,
        default=32,
        help="Batch size (default is 32)",
    )

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

        batches_by_language = get_batches_by_language(inputs)

        lang_ids = []
        for batch in batches_by_language:
            lang_ids.extend(lang_id_mapping[x["language"]] for x in batch)

        tr_model, tr_tokenizer = get_model(type="translator")

        translated_data = []
        for language_batch in tqdm(
            batches_by_language, desc="Processing each language"
        ):
            translated_data.extend(
                translate(
                    inputs=[x["text"] for x in language_batch],
                    model=tr_model,
                    tokenizer=tr_tokenizer,
                    batch_size=args.batch_size,
                    src_lang_ids=[
                        lang_id_mapping[x["language"]] for x in language_batch
                    ],
                    tgt_lang_ids=["eng_Latn"] * len(language_batch),
                )
            )

        del tr_model
        del tr_tokenizer

        tst_model, tst_tokenizer = get_model(type="en_detoxifier")

        detoxified = paraphrase_batch(
            texts=translated_data, model=tst_model, tokenizer=tst_tokenizer
        )

        del tst_tokenizer
        del tst_model

        tr_model, tr_tokenizer = get_model(type="translator")

        backtranslated_sources = []
        for i in trange(0, len(detoxified), 1000):
            backtranslated_sources.extend(
                translate(
                    inputs=detoxified[i : i + 1000],
                    model=tr_model,
                    tokenizer=tr_tokenizer,
                    batch_size=args.batch_size,
                    src_lang_ids=["eng_Latn"] * len(language_batch),
                    tgt_lang_ids=lang_ids[i : i + 1000],
                )
            )

        for doc_id, text in zip(doc_ids, backtranslated_sources):
            args.output.write(
                json.dumps({"id": doc_id, "text": text}, ensure_ascii=False)
            )
            args.output.write("\n")


if __name__ == "__main__":
    main()
