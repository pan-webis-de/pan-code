#!/usr/bin/env python3
from tira.check_format import lines_if_valid
from pathlib import Path
import click
from tira.io_utils import to_prototext
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import nltk
from torchmetrics.text.bert import BERTScore


def load_data(directory):
    ret = lines_if_valid(directory, "*.jsonl")
    return {i["id"]: i for i in ret}


def calculate_balanced_accuracy(truths, pred):
    score = balanced_accuracy_score(truths, pred > 0.5, zero_division=np.nan)
    if np.isnan(score):
        return None
    return float(score)


def calculate_bleu(orig, water):
    bleu_sum = 0
    for text_id, water_text in water.items():
        orig_text = orig[text_id]
        bleu_sum += sum(nltk.translate.bleu_score.sentence_bleu([orig_text], water_text))
    return bleu_sum/len(water)


def calculate_bert_score(orig, water):
    bertscore = BERTScore(truncation=True)
    bert_sum = 0
    for text_id, water_text in water.items():
        orig_text = orig[text_id]
        bert_sum += bertscore(orig_text, water_text)['f1'].item()
    return bert_sum/len(water)


@click.command()
@click.argument("watermarked_texts", type=Path)
@click.argument("original_texts", type=Path)
@click.argument("truth_labels", type=Path)
@click.argument("predictions", type=Path)
@click.option("--output-directory", type=Path, required=False, help="The output directory.")
def main(original_texts, watermarked_texts, truth_labels, predictions, output_directory):
    orig = load_data(original_texts)
    truths = load_data(truth_labels)
    water = load_data(watermarked_texts)
    predictions = load_data(predictions)

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for text_id, truth in truths.items():
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
        "balanced_accuracy": calculate_balanced_accuracy(truths, predictions),
        "BLEU": calculate_bleu(orig, water),
        "BertScore": calculate_bert_score(orig, water),
        "true_positives": true_positives/len(truths),
        "true_negatives": true_negatives/len(truths),
        "false_positives": false_positives/len(truths),
        "false_negatives": false_negatives/len(truths),
    }
    evaluation["twf"] = max(evaluation["BLEU"], evaluation["BertScore"]) * evaluation["balanced_accuracy"]
    print(evaluation)
    if output_directory:
        output_directory.mkdir(exist_ok=True, parents=True)
        with open(output_directory / "evaluation.prototext", "w") as f:
            f.write(to_prototext([evaluation]))


if __name__ == '__main__':
    main()

