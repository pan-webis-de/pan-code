import argparse
import os
from pathlib import Path
from typing import Any, List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizerFast,
)

FILE_PATH: str = Path(__file__).resolve()
INPUT_DATA_PATH: str = Path(FILE_PATH.parent.parent.parent, "input_data")
OUTPUT_DATA_PATH: str = Path(FILE_PATH.parent.parent.parent, "output_data")


os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)


class BackTranslationBaseline:
    """
    A class for performing back-translation based text detoxification.

    Attributes:
        lang_id_mapping (Dict[str, str]): Mapping from language codes to NLLB language codes
        translation_model (PreTrainedModel): Model for translation
        translation_tokenizer (PreTrainedTokenizer): Tokenizer for translation
        detox_model (PreTrainedModel): Model for detoxification
        detox_tokenizer (PreTrainedTokenizer): Tokenizer for detoxification
        hin_model (PreTrainedModel): Model for Hinglish translation
        hin_tokenizer (PreTrainedTokenizer): Tokenizer for Hinglish translation
    """

    def __init__(
        self,
        device: str = "cuda",
        translation_model_name: str = "facebook/nllb-200-3.3B",
        detox_model_name: str = "s-nlp/bart-base-detox",
        hinglish_model_name: str = "rudrashah/RLM-hinglish-translator",
    ):
        """
        Initialize models and tokenizers.

        Args:
            device: The device to load models on ('cuda' or 'cpu')
            translation_model_name: Name/path of the translation model
            detox_model_name: Name/path of the detoxification model
            hinglish_model_name: Name/path of the Hinglish translation model
        """
        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.translation_model_name = translation_model_name
        self.detox_model_name = detox_model_name
        self.hinglish_model_name = hinglish_model_name

        self.lang_id_mapping = {
            "ru": "rus_Cyrl",
            "en": "eng_Latn",
            "am": "amh_Ethi",
            "es": "spa_Latn",
            "uk": "ukr_Cyrl",
            "zh": "zho_Hans",
            "ar": "arb_Arab",
            "hi": "hin_Deva",
            "de": "deu_Latn",
            "tt": "tat_Cyrl",
            "fr": "fra_Latn",
            "it": "ita_Latn",
            "he": "heb_Hebr",
            "ja": "jpn_Jpan",
            "hin": "hin_Deva",
        }

        # Initialize translation model
        print(f"Loading translation model ({self.translation_model_name})...")
        self.translation_tokenizer = AutoTokenizer.from_pretrained(
            self.translation_model_name
        )
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.translation_model_name
        )
        self.translation_model = self.translation_model.eval().to(self.device)

        # Initialize detoxification model
        print(f"Loading detoxification model ({self.detox_model_name})...")
        self.detox_model = BartForConditionalGeneration.from_pretrained(
            self.detox_model_name
        )
        self.detox_model = self.detox_model.eval().to(self.device)
        self.detox_tokenizer = BartTokenizerFast.from_pretrained(self.detox_model_name)

        # Initialize Hinglish model
        print(f"Loading Hinglish translation model ({self.hinglish_model_name})...")
        self.hin_model = AutoModelForCausalLM.from_pretrained(self.hinglish_model_name)
        self.hin_model = self.hin_model.eval().to(self.device)
        self.hin_tokenizer = AutoTokenizer.from_pretrained(self.hinglish_model_name)
        self.hin_template = "Hinglish:\n{hi_en}\n\nEnglish:\n{en}"
        self.en_to_hin_template = "English:\n{en}\n\nHinglish:\n{hi_en}"

    def translate_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 32,
        max_length: int = 128,
        verbose: bool = True,
    ) -> List[str]:
        """
        Translate a batch of texts from source to target language.

        Args:
            texts: List of texts to translate
            src_lang: Source language code
            tgt_lang: Target language code
            batch_size: Batch size for translation
            verbose: Whether to show progress bar

        Returns:
            List of translated texts
        """
        self.translation_tokenizer.src_lang = self.lang_id_mapping[src_lang]
        self.translation_tokenizer.tgt_lang = self.lang_id_mapping[tgt_lang]

        translations = []
        iterator = range(0, len(texts), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc=f"Translating {src_lang}→{tgt_lang}")

        for i in iterator:
            batch = texts[i : i + batch_size]
            tokenized = self.translation_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            outputs = self.translation_model.generate(
                **tokenized,
                forced_bos_token_id=self.translation_tokenizer.lang_code_to_id[
                    self.translation_tokenizer.tgt_lang
                ],
            )
            translations.extend(
                self.translation_tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
            )
        return translations

    def detoxify_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        verbose: bool = True,
    ) -> List[str]:
        """
        Detoxify a batch of texts.

        Args:
            texts: List of texts to detoxify
            batch_size: Batch size for detoxification
            verbose: Whether to show progress bar

        Returns:
            List of detoxified texts
        """
        detoxified = []
        iterator = range(0, len(texts), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Detoxifying")

        for i in iterator:
            batch = texts[i : i + batch_size]
            tokenized = self.detox_tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            outputs = self.detox_model.generate(**tokenized)
            detoxified.extend(
                self.detox_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            )
        return detoxified

    def translate_hinglish_batch(
        self,
        texts: List[str],
        direction: str = "to_english",
        batch_size: int = 8,
        verbose: bool = True,
        **generation_kwargs: Any,
    ) -> List[str]:
        """
        Translate between Hinglish and English.

        Args:
            texts: List of texts to translate
            direction: Either 'to_english' or 'to_hinglish'
            batch_size: Batch size for translation
            verbose: Whether to show progress bar
            generation_kwargs: Additional generation arguments

        Returns:
            List of translated texts
        """
        template = (
            self.hin_template if direction == "to_english" else self.en_to_hin_template
        )
        dataloader = DataLoader(texts, batch_size=batch_size)
        generated = []

        iterator = dataloader
        if verbose:
            iterator = tqdm(
                iterator,
                desc=f"Hinglish {'→English' if direction == 'to_english' else '←English'}",
            )

        with torch.no_grad():
            for batch in iterator:
                filled = [template.format(hi_en=b, en="") for b in batch]
                input_ids = self.hin_tokenizer(
                    filled,
                    return_tensors="pt",
                    padding=True,
                ).input_ids.to(self.device)

                outputs = self.hin_model.generate(
                    input_ids=input_ids, **generation_kwargs
                )
                decoded = self.hin_tokenizer.batch_decode(
                    outputs[:, input_ids.shape[1] :], skip_special_tokens=True
                )
                generated.extend(decoded)
        return generated

    def process_dataset(
        self,
        input_path: str,
        output_path: str,
        batch_size: int = 64,
        max_length: int = 128,
        verbose: bool = True,
    ) -> None:
        """
        Process a dataset through the backtranslation pipeline.

        Args:
            input_path: Path to input TSV file
            output_path: Path to save output TSV file
            batch_size: Batch size for processing
            max_length: Max size of processed sequence
            verbose: Whether to show progress bars
        """
        # Load data
        if verbose:
            print(f"Loading data from {input_path}...")
        data = pd.read_csv(input_path, sep="\t")

        # Step 1: Translate all languages to English
        translations = {lang: [] for lang in self.lang_id_mapping}
        for lang in tqdm(
            self.lang_id_mapping, desc="Translating to English", disable=not verbose
        ):
            lang_data = data[data.lang == lang].toxic_sentence.tolist()

            if not lang_data:
                continue

            if lang == "en":
                translations[lang] = lang_data
            elif lang == "hin":
                translations[lang] = self.translate_hinglish_batch(
                    lang_data,
                    direction="to_english",
                    batch_size=batch_size,
                    verbose=verbose,
                    max_length=max_length,
                )
            else:
                translations[lang] = self.translate_batch(
                    lang_data,
                    src_lang=lang,
                    tgt_lang="en",
                    batch_size=batch_size,
                    verbose=verbose,
                )

        # Step 2: Detoxify all English texts
        detoxified = {lang: [] for lang in translations}
        for lang in tqdm(translations, desc="Detoxifying", disable=not verbose):
            if lang == "ru":  # Skip Russian as in original code
                continue
            if translations[lang]:
                detoxified[lang] = self.detoxify_batch(
                    translations[lang], batch_size=batch_size, verbose=verbose
                )

        # Step 3: Backtranslate to original languages
        backtranslations = {lang: [] for lang in self.lang_id_mapping}
        for lang in tqdm(
            self.lang_id_mapping, desc="Backtranslating", disable=not verbose
        ):
            if not detoxified.get(lang):
                continue

            if lang == "en":
                backtranslations[lang] = detoxified[lang]
            elif lang == "hin":
                backtranslations[lang] = self.translate_hinglish_batch(
                    detoxified[lang],
                    direction="to_hinglish",
                    batch_size=batch_size,
                    verbose=verbose,
                    max_length=128,
                )
            else:
                backtranslations[lang] = self.translate_batch(
                    detoxified[lang],
                    src_lang="en",
                    tgt_lang=lang,
                    batch_size=batch_size,
                    verbose=verbose,
                )

        # Save results
        if verbose:
            print(f"Saving results to {output_path}...")
        os.makedirs(output_path.parent, exist_ok=True)

        data["neutral_sentence"] = data.lang.map(
            lambda x: (
                backtranslations.get(x, [""] * len(data))[0]
                if backtranslations.get(x)
                else ""
            )
        )
        data.to_csv(output_path, sep="\t", index=False)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Backtranslation-based text detoxification pipeline"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default=Path(INPUT_DATA_PATH, "dev_inputs.tsv"),
        help="Path to input TSV file containing toxic sentences",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=Path(OUTPUT_DATA_PATH, "baseline_backtranslation_dev.tsv"),
        help="Path to save output TSV file with neutral sentences",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for processing (default: 64)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max size of processed sequences (default: 128)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run models on (default: cuda)",
    )
    parser.add_argument(
        "--verbose", type=bool, help="Show progress bars and verbose output"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Initializing backtranslation pipeline...")
    pipeline = BackTranslationBaseline(device=args.device)

    print("Processing dataset...")
    pipeline.process_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
