#!/usr/bin/env python3

import argparse
import json
import logging
import re
from typing import List
from datasets import load_dataset
from tqdm import tqdm

SPACES = re.compile(r"\s+")


def detoxify(
    text: str,
    stopwords: List[str],
    remove_all_terms: bool = True,
    remove_no_terms: bool = False,
) -> str:

    if remove_no_terms:
        return text
    if remove_all_terms:
        return ""

    tokens = [
        token
        for token in SPACES.split(text)
        if not stopwords or token.lower().strip() not in stopwords
    ]
    return " ".join(tokens)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="The trivial baseline for the PAN 2024 text detoxification task"
        "that removes all/none/or specified stopwrods from given text for "
        "detoxification."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=argparse.FileType("rb"),
        help="The input file, expected a jsonl file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=argparse.FileType("w", encoding="UTF-8"),
        help="The output file, will create a jsonl file.",
    )
    parser.add_argument(
        "--language",
        required=False,
        type=str,
        default=None,
        help="Specify language. Should be one of "
        "['am', 'es', 'ru', 'uk', 'en', 'zh', 'ar', 'hi', 'de']."
        " Without specification will load all stopwords.",
        choices=["am", "es", "ru", "uk", "en", "zh", "ar", "hi", "de"],
    )
    parser.add_argument(
        "--remove-all-terms",
        required=False,
        default=False,
        type=bool,
        help="Generate the empty string.",
    )
    parser.add_argument(
        "--remove-no-terms",
        required=False,
        default=False,
        type=bool,
        help="Output the text without modification.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Specify logging level (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    if args.language is not None:
        logging.info(f"Loading stopwords for {args.language}")
        stopwords = load_dataset("textdetox/multilingual_toxic_lexicon")[args.language][
            "text"
        ]
        stopwords = set(stopwords)
    else:
        logging.info(
            "No specific language for stopwords provided. Loading all stopwords"
        )
        stopwords = load_dataset("textdetox/multilingual_toxic_lexicon")

        words = []

        for language in stopwords.keys():
            words.extend(stopwords[language]["text"])

        stopwords = set(words)

    logging.info("Started processing texts.")
    for line in tqdm(args.input, desc="Processing 'input.jsonl'"):
        instance = json.loads(line)
        instance["text"] = detoxify(
            instance["text"], stopwords, args.remove_all_terms, args.remove_no_terms
        )

        args.output.write(json.dumps(instance, ensure_ascii=False))
        args.output.write("\n")

    logging.info(f"All done. Outputs are written to {args.output.name}")


if __name__ == "__main__":
    main()
