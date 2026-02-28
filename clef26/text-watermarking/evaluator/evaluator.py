#!/usr/bin/env python3
from tira.check_format import lines_if_valid
from pathlib import Path
import click
from tira.io_utils import to_prototext
import pandas as pd


def load_data(directory):
    ret = lines_if_valid(directory, "*.jsonl")
    return {i["id"]: i for i in ret}


def calculate_bleu(orig, water):
    # TODO
    return 0.0


def calculate_bert_score(orig, water):
    # TODO
    return 0.0


@click.command()
@click.argument("watermarked_texts", type=Path)
@click.argument("original_texts", type=Path)
@click.argument("predictions", type=Path)
@click.option("--output-directory", type=Path, required=False, help="The output directory.")
def main(original_texts, watermarked_texts, predictions, output_directory):
    orig = load_data(original_texts)
    water = load_data(watermarked_texts)
    predictions = load_data(predictions)

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for text_id, truth in orig.items():
        pred = predictions[text_id]
        if truth["truth_label"] == "watermarked" and pred["label"] == 1.0:
            true_positives += 1
        elif truth["truth_label"] == "not-watermarked" and pred["label"] == 0.0:
            true_negatives += 1
        elif truth["truth_label"] == "watermarked" and pred["label"] == 0.0:
            false_negatives += 1
        elif truth["truth_label"] == "not-watermarked" and pred["label"] == 1.0:
            false_positives += 1
        else:
            raise ValueError("this should not happen...")

    evaluation = {
        "BLEU": calculate_bleu(orig, water),
        "BertScore": calculate_bert_score(orig, water),
        "true_positives": true_positives/len(orig),
        "true_negatives": true_negatives/len(orig),
        "false_positives": false_positives/len(orig),
        "false_negatives": false_negatives/len(orig),
    }
    print(evaluation)
    if output_directory:
        output_directory.mkdir(exist_ok=True, parents=True)
        with open(output_directory / "evaluation.prototext", "w") as f:
            f.write(to_prototext([evaluation]))


if __name__ == '__main__':
    main()

