import argparse
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

FILE_PATH: str = Path(__file__).resolve()
INPUT_DATA_PATH: str = Path(FILE_PATH.parent.parent.parent, "input_data")
OUTPUT_DATA_PATH: str = Path(FILE_PATH.parent.parent.parent, "output_data")


class DetoxificationDataset(Dataset):
    """Dataset for batch processing detoxification tasks."""

    def __init__(
        self, texts: List[str], langs: List[str], lang_prompts: Dict[str, str]
    ):
        self.texts = texts
        self.langs = langs
        self.lang_prompts = lang_prompts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Tuple[str, str]:
        lang = self.langs[idx]
        prompt = self.lang_prompts.get(lang, "Detoxify: ")
        return prompt + self.texts[idx], lang


class MT0Detoxifier:
    """
    A class for detoxifying text using the MT0 model with language-specific prompts.

    Attributes:
        model (PreTrainedModel): The loaded MT0 model for sequence-to-sequence tasks.
        tokenizer (PreTrainedTokenizerBase): The tokenizer for the MT0 model.
        lang_prompts (Dict[str, str]): Language-specific prompts for detoxification.
    """

    LANGUAGES = Literal[
        "zh",
        "es",
        "ru",
        "ar",
        "hi",
        "uk",
        "de",
        "am",
        "en",
        "it",
        "ja",
        "he",
        "fr",
        "tt",
        "hin",
    ]

    def __init__(self, model_name: str = "s-nlp/mt0-xl-detox-orpo"):
        """
        Initialize the MT0Detoxifier with the specified model.

        Args:
            model_name (str): Name or path of the pretrained model to load.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.lang_prompts = {
            "zh": "排毒：",
            "es": "Desintoxicar: ",
            "ru": "Детоксифицируй: ",
            "ar": "إزالة السموم: ",
            "hi": "विषहरण: ",
            "uk": "Детоксифікуй: ",
            "de": "Entgiften: ",
            "am": "መርዝ መርዝ: ",
            "en": "Detoxify: ",
            "it": "Disintossicare: ",
            "ja": "解毒: ",
            "he": "לְסַלֵק רַעַל: ",
            "fr": "Désintoxiquer:",
            "tt": "Токсиннарны чыгару: ",
            "hin": "Detoxify: ",
        }

    def _prepare_batch(self, batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts and prepare for model input."""
        return self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

    def _generate_batch(self, encodings: Dict[str, torch.Tensor]) -> List[str]:
        """Generate detoxified text for a batch of inputs."""
        with torch.no_grad():
            outputs = self.model.generate(
                **encodings,
                max_length=128,
                num_beams=10,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                num_beam_groups=5,
                diversity_penalty=2.5,
                num_return_sequences=1,
                early_stopping=True,
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def detoxify_batch(
        self, texts: List[str], langs: List[LANGUAGES], batch_size: int = 16
    ) -> List[str]:
        """
        Detoxify a batch of texts with their corresponding languages.

        Args:
            texts (List[str]): List of toxic texts to detoxify.
            langs (List[LANGUAGES]): List of language codes corresponding to the texts.

        Returns:
            List[str]: List of detoxified texts.
        """
        # Prepare prompts
        batch_texts = [
            self.lang_prompts.get(lang, "Detoxify: ") + text
            for text, lang in zip(texts, langs)
        ]

        # Process in proper batches
        results = []

        for i in tqdm(
            range(0, len(batch_texts), batch_size), desc="Processing batches"
        ):
            current_batch = batch_texts[i : i + batch_size]
            encodings = self._prepare_batch(current_batch)
            batch_results = self._generate_batch(encodings)
            results.extend(batch_results)

        return results


def process_file(input_path: str, output_path: str, batch_size: int = 16):
    """
    Process an input TSV file, detoxify the toxic sentences, and save to output file.

    Args:
        input_path (str): Path to the input TSV file.
        output_path (str): Path to save the output TSV file.
    """
    # Initialize detoxifier
    detoxifier = MT0Detoxifier()

    # Read input data
    df = pd.read_csv(input_path, sep="\t")

    # Process in true batches
    neutral_sentences = detoxifier.detoxify_batch(
        texts=df["toxic_sentence"].tolist(),
        langs=df["lang"].tolist(),
        batch_size=batch_size,
    )

    # Add results to dataframe and save
    df["neutral_sentence"] = neutral_sentences
    df.to_csv(output_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Detoxify text using MT0 baseline model"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=Path(INPUT_DATA_PATH, "dev_inputs.tsv"),
        help="Path to input TSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=Path(OUTPUT_DATA_PATH, "baseline_mt0_dev.tsv"),
        help="Path to output TSV file",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (default: 16)",
    )

    args = parser.parse_args()

    # Validate paths
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process the file
    process_file(args.input_path, args.output_path, args.batch_size)
    print(f"Processing complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
