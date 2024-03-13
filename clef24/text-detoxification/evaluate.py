#!/usr/bin/env python3

__credits__ = ["David Dale", "Daniil Moskovskiy", "Dmitry Ustalov", "Elisei Stakovskii"]

import argparse
import sys
from functools import partial
from typing import Optional, Type, Tuple, Dict, Callable, List, Union
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sacrebleu import CHRF
from tqdm.auto import trange
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


def prepare_target_label(
    model: AutoModelForSequenceClassification, target_label: Union[int, str]
) -> int:
    """
    Prepare the target label to ensure it is valid for the given model.

    Args:
        model (AutoModelForSequenceClassification): Text classification model.
        target_label (Union[int, str]): The target label to prepare.

    Returns:
        int: The prepared target label.

    Raises:
        ValueError: If the target_label is not found in model labels or ids.
    """
    if target_label in model.config.id2label:
        pass
    elif target_label in model.config.label2id:
        target_label = model.config.label2id.get(target_label)
    elif (
        isinstance(target_label, str)
        and target_label.isnumeric()
        and int(target_label) in model.config.id2label
    ):
        target_label = int(target_label)
    else:
        raise ValueError(
            f'target_label "{target_label}" not in model labels or ids: {model.config.id2label}.'
        )
    assert isinstance(target_label, int)
    return target_label


def classify_texts(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    target_label: Union[int, str],
    second_texts: Optional[List[str]] = None,
    batch_size: int = 32,
    raw_logits: bool = False,
    desc: Optional[str] = "Calculating STA scores",
) -> npt.NDArray[np.float64]:
    """
    Classify a list of texts using the given model and tokenizer.

    Args:
        model (AutoModelForSequenceClassification): Text classification model.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        texts (List[str]): List of texts to classify.
        target_label (Union[int, str]): The target label for classification.
        second_texts (Optional[List[str]]): List of secondary texts (not needed by default).
        batch_size (int): Batch size for inference.
        raw_logits (bool): Whether to return raw logits instead of probs.
        desc (Optional[str]): Description for tqdm progress bar.

    Returns:
        npt.NDArray[np.float64]: Array of classification scores for the texts.
    """

    target_label = prepare_target_label(model, target_label)

    res = []

    for i in trange(0, len(texts), batch_size, desc=desc):
        inputs = [texts[i : i + batch_size]]

        if second_texts is not None:
            inputs.append(second_texts[i : i + batch_size])
        inputs = tokenizer(
            *inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            try:
                logits = model(**inputs).logits
                if raw_logits:
                    preds = logits[:, target_label]
                elif logits.shape[-1] > 1:
                    preds = torch.softmax(logits, -1)[:, target_label]
                else:
                    preds = torch.sigmoid(logits)[:, 0]
                preds = preds.cpu().numpy()
            except:
                print(i, i + batch_size)
                preds = [0] * len(inputs)
        res.append(preds)
    return np.concatenate(res)


def evaluate_sta(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    target_label: int = 1,  # 1 is polite, 0 is toxic
    batch_size: int = 32,
) -> npt.NDArray[np.float64]:
    """
    Evaluate the STA of a list of texts using the given model and tokenizer.

    Args:
        model (AutoModelForSequenceClassification): Text classification model.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        texts (List[str]): List of texts to evaluate.
        target_label (int): The target label for style evaluation.
        batch_size (int): Batch size for inference.

    Returns:
        npt.NDArray[np.float64]: Array of STA scores for the texts.
    """
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        model, tokenizer, texts, target_label, batch_size=batch_size, desc="Style"
    )

    return scores


def evaluate_sim(
    model: SentenceTransformer,
    original_texts: List[str],
    rewritten_texts: List[str],
    batch_size: int = 32,
    efficient_version: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Evaluate the semantic similarity between original and rewritten texts.
    Note that the subtraction is done due to the implementation of the `cosine` metric in `scipy`.
    For more details see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html

    Args:
        model (SentenceTransformer): The sentence transformer model.
        original_texts (List[str]): List of original texts.
        rewritten_texts (List[str]): List of rewritten texts.
        batch_size (int): Batch size for inference.
        efficient_version (bool): To use efficient calculation method.

    Returns:
        npt.NDArray[np.float64]: Array of semantic similarity scores between \
              original and rewritten texts.
    """
    similarities = []

    batch_size = min(batch_size, len(original_texts))
    for i in trange(0, len(original_texts), batch_size, desc="Calculating SIM scores"):
        original_batch = original_texts[i : i + batch_size]
        rewritten_batch = rewritten_texts[i : i + batch_size]

        embeddings = model.encode(original_batch + rewritten_batch)
        original_embeddings = embeddings[: len(original_batch)]
        rewritten_embeddings = embeddings[len(original_batch) :]

        if efficient_version:
            similarity_matrix = np.dot(original_embeddings, rewritten_embeddings.T)
            original_norms = np.linalg.norm(original_embeddings, axis=1)
            rewritten_norms = np.linalg.norm(rewritten_embeddings, axis=1)
            similarities.extend(
                1
                - similarity_matrix / (np.outer(original_norms, rewritten_norms) + 1e-9)
            )
        else:
            t = [
                1 - cosine(original_embedding, rewritten_embedding)
                for original_embedding, rewritten_embedding in zip(
                    original_embeddings, rewritten_embeddings
                )
            ]
            similarities.extend(t)
    return similarities


def evaluate_style_transfer(
    original_texts: List[str],
    rewritten_texts: List[str],
    style_model: AutoModelForSequenceClassification,
    style_tokenizer: AutoTokenizer,
    meaning_model: AutoModelForSequenceClassification,
    references: Optional[List[str]] = None,
    style_target_label: int = 1,
    batch_size: int = 32,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Wrapper for calculating sub-metrics and joint metric.

    Args:
        original_texts (List[str]): List of original texts.
        rewritten_texts (List[str]): List of rewritten texts.
        style_model (AutoModelForSequenceClassification): The style classification model.
        style_tokenizer (AutoTokenizer): The tokenizer corresponding to the style model.
        meaning_model (AutoModelForSequenceClassification): The meaning classification model.
        references (Optional[List[str]]): List of reference texts (if available).
        style_target_label (int): The target label for style classification.
        batch_size (int): Batch size for inference.

    Returns:
        Dict[str, npt.NDArray[np.float64]]: Dictionary containing evaluation metrics.
    """
    accuracy = evaluate_sta(
        style_model,
        style_tokenizer,
        rewritten_texts,
        target_label=style_target_label,
        batch_size=batch_size,
    )

    similarity = evaluate_sim(
        model=meaning_model,
        original_texts=original_texts,
        rewritten_texts=rewritten_texts,
        batch_size=batch_size,
    )

    result = {
        "STA": accuracy,
        "SIM": similarity,
    }

    if references is not None:

        chrf = CHRF()

        result["CHRF"] = np.array(
            [
                chrf.sentence_score(hypothesis=rewritten, references=[reference]).score
                / 100
                for rewritten, reference in zip(rewritten_texts, references)
            ],
            dtype=np.float64,
        )

        result["J"] = result["STA"] * result["SIM"] * result["CHRF"]

    return result


def load_model(
    model_name: Optional[str] = None,
    model: Optional[AutoModelForSequenceClassification] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    model_class: Type[
        AutoModelForSequenceClassification
    ] = AutoModelForSequenceClassification,
    use_cuda: bool = True,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load a pre-trained model and tokenizer from Hugging Face Hub.

    Args:
        model_name (Optional[str]): The name of the model to load.
        model (Optional[AutoModelForSequenceClassification]): A pre-loaded model instance.
        tokenizer (Optional[AutoTokenizer]): A pre-loaded tokenizer instance.
        model_class (Type[AutoModelForSequenceClassification]): The class of the model to load.
        use_cuda (bool): Whether to use CUDA for GPU acceleration.

    Returns:
        Tuple[AutoModelForSequenceClassification, AutoTokenizer]: The loaded model and tokenizer.
    """
    if model_name == "sentence-transformers/LaBSE":
        model = SentenceTransformer("sentence-transformers/LaBSE")
        return model
    if model is None:
        if model_name is None:
            raise ValueError("Either model or model_name should be provided")
        model = model_class.from_pretrained(model_name)

        if torch.cuda.is_available() and use_cuda:
            model.cuda()
    if tokenizer is None:
        if model_name is None:
            raise ValueError("Either tokenizer or model_name should be provided")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def format_prototext(measure: str, value: str) -> str:
    """
    Format evaluation metrics into prototext format.

    Args:
        measure (str): The name of the evaluation measure.
        value (str): The value of the evaluation measure.

    Returns:
        str: The formatted prototext string.
    """
    return f'measure{{\n  key: "{measure}"\n  value: "{value}"\n}}\n'


def run_evaluation(
    args: argparse.Namespace,
    evaluator: Callable[..., Dict[str, npt.NDArray[np.float64]]],
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Run evaluation on input data using the specified evaluator.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        evaluator (Callable[..., Dict[str, npt.NDArray[np.float64]]]): The evaluation function.

    Returns:
        Dict[str, npt.NDArray[np.float64]]: Dictionary containing evaluation results.
    """
    df_input = pd.read_json(args.input, convert_dates=False, lines=True)
    df_input = df_input[["id", "text"]]
    df_input.set_index("id", inplace=True)
    df_input.rename(columns={"text": "input"}, inplace=True)

    df_prediction = pd.read_json(args.prediction, convert_dates=False, lines=True)
    df_prediction = df_prediction[["id", "text"]]
    df_prediction.set_index("id", inplace=True)
    df_prediction.rename(columns={"text": "prediction"}, inplace=True)

    df = df_input.join(df_prediction)

    if args.golden:

        df_references = pd.read_json(args.golden, convert_dates=False, lines=True)
        df_references = df_references[["id", "text"]]
        df_references.set_index("id", inplace=True)
        df_references.rename(columns={"text": "reference"}, inplace=True)

        df = df.join(df_references)

        assert (
            len(df) == len(df_input) == len(df_prediction) == len(df_references)
        ), f"Dataset lengths {len(df_input)} & {len(df_prediction)} & {len(df_references)} != {len(df)}"

    assert (
        len(df) == len(df_input) == len(df_prediction)
    ), f"Dataset lengths {len(df_input)} & {len(df_prediction)} != {len(df)}"

    assert not df.isna().values.any(), "Datasets contain missing entries"

    result = evaluator(
        original_texts=df["input"].tolist(),
        rewritten_texts=df["prediction"].tolist(),
        references=df["reference"].tolist() or None,
    )

    aggregated = {measure: np.mean(values).item() for measure, values in result.items()}

    for measure, value in aggregated.items():
        args.output.write(format_prototext(measure, str(value)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("rb"),
        required=True,
        help="Initial texts before style transfer",
    )
    parser.add_argument(
        "-g",
        "--golden",
        type=argparse.FileType("rb"),
        required=False,
        help="Ground truth texts after style transfer",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w", encoding="UTF-8"),
        default=sys.stdout,
        help="Path where to write the evaluation results",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disable use of CUDA"
    )
    parser.add_argument(
        "--prediction", type=argparse.FileType("rb"), help="Your model predictions"
    )

    args = parser.parse_args()

    style_model, style_tokenizer = load_model(
        "textdetox/xlmr-large-toxicity-classifier", use_cuda=not args.no_cuda
    )
    meaning_model = load_model("sentence-transformers/LaBSE", use_cuda=not args.no_cuda)

    run_evaluation(
        args,
        evaluator=partial(
            evaluate_style_transfer,
            style_model=style_model,
            style_tokenizer=style_tokenizer,
            meaning_model=meaning_model,
            style_target_label=0,
        ),
    )


if __name__ == "__main__":
    main()
