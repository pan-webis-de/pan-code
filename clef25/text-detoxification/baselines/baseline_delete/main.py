import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, Set

import jieba
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

FILE_PATH: str = Path(__file__).resolve()
INPUT_DATA_PATH: str = Path(FILE_PATH.parent.parent.parent, "input_data")
OUTPUT_DATA_PATH: str = Path(FILE_PATH.parent.parent.parent, "output_data")


os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)


class DetoxificationBaseline:
    """
    A class for implementing a baseline toxic text detoxification approach.
    The baseline removes toxic terms identified in a multilingual lexicon.

    Attributes:
        stopwords (Set[str]): A set of toxic terms to remove
        spaces_re (re.Pattern): Compiled regex for whitespace matching
    """

    def __init__(self, toxic_lexicon_path: Optional[str] = None):
        """
        Initialize the detoxification baseline.

        Args:
            toxic_lexicon_path: Optional path to load custom toxic lexicon.
                              If None, uses default multilingual lexicon.
        """
        self.spaces_re = re.compile(r"\s+")
        self.stopwords = self._load_toxic_lexicon(toxic_lexicon_path)

    def _load_toxic_lexicon(
        self,
        path: Optional[str] = None,
        hf_dataset_name: str = "textdetox/multilingual_toxic_lexicon",
    ) -> Set[str]:
        """
        Load toxic lexicon from HuggingFace datasets or local path.

        Args:
            path: Optional path to local lexicon file

        Returns:
            Set of toxic terms
        """
        if path:
            with open(path) as f:
                return set(json.load(f))

        stopwords_dataset = load_dataset(hf_dataset_name)
        words = []

        for language in stopwords_dataset.keys():
            words.extend(stopwords_dataset[language]["text"])

        return set(words)

    def detoxify(
        self,
        text: str,
        language: str = "en",
        remove_all_terms: bool = False,
        remove_no_terms: bool = False,
    ) -> str:
        """
        Remove toxic terms from input text based on language.

        Args:
            text: Input text to detoxify
            language: Language code ('zh' for Chinese, others for space-separated)
            remove_all_terms: If True, returns empty string (for testing)
            remove_no_terms: If True, returns original text (for testing)

        Returns:
            Detoxified text
        """
        if remove_no_terms:
            return text
        if remove_all_terms:
            return ""

        if language != "zh":
            tokens = [
                token
                for token in self.spaces_re.split(text)
                if token.lower().strip() not in self.stopwords
            ]
            return " ".join(tokens)
        else:
            return "".join([x for x in jieba.cut(text) if x not in self.stopwords])

    def process_dataframe(
        self,
        input_path: Path,
        output_path: Path,
        text_col: str = "toxic_sentence",
        lang_col: str = "lang",
    ) -> None:
        """
        Process a dataframe of toxic sentences, saving detoxified versions.

        Args:
            input_path: Path to input TSV file
            output_path: Path to save output TSV file
            text_col: Name of column containing toxic text
            lang_col: Name of column containing language codes
        """
        df = pd.read_csv(input_path, sep="\t")

        tqdm.pandas(desc="Detoxifying sentences")
        df["neutral_sentence"] = df.progress_apply(
            lambda row: self.detoxify(
                row[text_col], language=row[lang_col], remove_all_terms=False
            ),
            axis=1,
        )

        df.to_csv(output_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline detoxification by removing toxic terms"
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path(INPUT_DATA_PATH, "dev_inputs.tsv"),
        help="Path to input TSV file",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path(OUTPUT_DATA_PATH, "baseline_delete_dev.tsv"),
        help="Path to output TSV file",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=None,
        help="Optional path to custom toxic lexicon JSON file",
    )
    args = parser.parse_args()

    detox = DetoxificationBaseline(args.lexicon)
    detox.process_dataframe(args.input_path, args.output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
