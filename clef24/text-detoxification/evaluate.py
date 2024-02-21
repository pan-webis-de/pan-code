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
    AutoModelForMaskedLM,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


def prepare_target_label(
    model: AutoModelForSequenceClassification, target_label: Union[int, str]
) -> int:
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


def get_pppl_score(
    model: AutoModelForMaskedLM, tokenizer: AutoTokenizer, sentence: str, device: str
) -> float:
    """
    Pseudoperplexity function as realized by David Dale
    source:
    https://gist.github.com/avidale/f574c014cd686709636b89208f2259ce
    """
    tensor_input = tokenizer.encode(sentence, return_tensors="pt").to(device)
    repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1).to(device)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2].to(device)
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id).to(
        device
    )
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100).to(
        device
    )
    with torch.inference_mode():
        loss = model(masked_input, labels=labels).loss
    return np.exp(loss.item())


# def evaluate_cola(
#     input_sentences: List[str],
#     output_sentences: List[str],
#     model: AutoModelForMaskedLM,
#     tokenizer: AutoTokenizer,
#     use_cuda: bool = True,
# ) -> np.ndarray:
#     """
#     This function using pseudoperplexity performs a relative fluency estimation and outputs a list of bool values.
#     As inputs the function expects two lists of sentences. In essence, the function compares whether the pseudoperplexity score
#     of the sentence in the second list is less or equal to the score of the sentence in the first list. If it does, then the ouput is 1, i.e. fluency
#     did not become worse, otherwise the score is 0 - the sentence is not fluent, i.e. the transformation to the initial sentence made the sentence
#     ungrammatical
#     """
#     device = torch.device("cuda" if use_cuda else "cpu")
#     first_pppl_vector, second_pppl_vector, final_bools_vector = [], [], []

#     assert len(input_sentences) == len(
#         output_sentences
#     ), "Input and output sentences number mismatch!"

#     for sent in input_sentences:
#         curr_pppl_score = get_pppl_score(
#             sentence=sent, model=model, tokenizer=tokenizer, device=device
#         )
#         first_pppl_vector.append(curr_pppl_score)

#     for sent in output_sentences:
#         curr_pppl_score = get_pppl_score(
#             sentence=sent, model=model, tokenizer=tokenizer, device=device
#         )
#         second_pppl_vector.append(curr_pppl_score)

#     for i in range(len(first_pppl_vector)):
#         if second_pppl_vector[i] <= first_pppl_vector[i]:
#             final_bools_vector.append(1)
#         else:
#             final_bools_vector.append(0)

#     assert len(first_pppl_vector) == len(second_pppl_vector) == len(final_bools_vector)

#     return np.array(final_bools_vector)


def classify_texts(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    target_label: Union[int, str],
    second_texts: Optional[List[str]] = None,
    batch_size: int = 32,
    raw_logits: bool = False,
    desc: Optional[str] = None,
) -> npt.NDArray[np.float64]:
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


def evaluate_style(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    target_label: int = 1,  # 1 is formal, 0 is informal
    batch_size: int = 32,
) -> npt.NDArray[np.float64]:
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        model, tokenizer, texts, target_label, batch_size=batch_size, desc="Style"
    )
    return scores


def evaluate_meaning(
    model: SentenceTransformer,
    original_texts: List[str],
    rewritten_texts: List[str],
    batch_size: int = 32,
    efficient_version: bool = False,
) -> npt.NDArray[np.float64]:

    similarities = []

    batch_size = min(batch_size, len(original_texts))

    for i in range(0, len(original_texts), batch_size):
        original_batch = original_texts[i : i + batch_size]
        rewritten_batch = rewritten_texts[i : i + batch_size]

        embeddings = model.encode(original_batch + rewritten_batch)
        print("embeddings", embeddings, len(embeddings))
        original_embeddings = embeddings[:batch_size]
        rewritten_embeddings = embeddings[batch_size:]

        print("original embeddings", original_embeddings)
        print("rewritten embeddings", rewritten_embeddings)

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
            print("t", t)
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
    accuracy = evaluate_style(
        style_model,
        style_tokenizer,
        rewritten_texts,
        target_label=style_target_label,
        batch_size=batch_size,
    )

    similarity = evaluate_meaning(
        model=meaning_model,
        original_texts=original_texts,
        rewritten_texts=rewritten_texts,
        batch_size=batch_size,
    )

    result = {
        "accuracy": accuracy,
        "similarity": similarity,
    }

    if references is not None:
        chrf = CHRF()

        result["chrf"] = np.array(
            [
                chrf.sentence_score(hypothesis, [reference]).score
                for hypothesis, reference in zip(rewritten_texts, references)
            ],
            dtype=np.float64,
        )
    print(accuracy, similarity)
    result["joint"] = result["accuracy"] * result["similarity"] * result["chrf"]

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
    return f'measure{{\n  key: "{measure}"\n  value: "{value}"\n}}\n'


def run_evaluation(
    args: argparse.Namespace,
    evaluator: Callable[..., Dict[str, npt.NDArray[np.float64]]],
) -> Dict[str, npt.NDArray[np.float64]]:
    df_input = pd.read_json(args.input, convert_dates=False, lines=True)
    df_input = df_input[["id", "text"]]
    df_input.set_index("id", inplace=True)
    df_input.rename(columns={"text": "input"}, inplace=True)

    df_prediction = pd.read_json(args.prediction, convert_dates=False, lines=True)
    df_prediction = df_prediction[["id", "text"]]
    df_prediction.set_index("id", inplace=True)
    df_prediction.rename(columns={"text": "prediction"}, inplace=True)

    df_references = pd.read_json(args.golden, convert_dates=False, lines=True)
    df_references = df_references[["id", "text"]]
    df_references.set_index("id", inplace=True)
    df_references.rename(columns={"text": "reference"}, inplace=True)

    df = df_input.join(df_prediction).join(df_references)

    assert (
        len(df) == len(df_input) == len(df_prediction) == len(df_references)
    ), f"Dataset lengths {len(df_input)} & {len(df_prediction)} & {len(df_references)} != {len(df)}"

    assert not df.isna().values.any(), "Datasets contain missing entries"

    result = evaluator(
        original_texts=df["input"].tolist(),
        rewritten_texts=df["prediction"].tolist(),
        references=df["reference"].tolist(),
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
        required=True,
        help="Ground truth texts after style transfer",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w", encoding="UTF-8"),
        default=sys.stdout,
        help="Where to print the evaluation results",
    )
    parser.add_argument(
        "--style-model",
        type=str,
        required=True,
        help="Style evaluation model on Hugging Face Hub",
    )
    parser.add_argument(
        "--meaning-model",
        type=str,
        required=True,
        help="Meaning evaluation model on Hugging Face Hub",
    )

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disable use of CUDA"
    )
    parser.add_argument(
        "--prediction", type=argparse.FileType("rb"), help="Your model predictions"
    )

    args = parser.parse_args()

    style_model, style_tokenizer = load_model(
        args.style_model, use_cuda=not args.no_cuda
    )
    meaning_model = load_model(args.meaning_model, use_cuda=not args.no_cuda)

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
