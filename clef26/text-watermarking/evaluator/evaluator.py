#!/usr/bin/env python3
from tira.check_format import lines_if_valid
from pathlib import Path
import click
from tira.io_utils import to_prototext
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from bert_score import BERTScorer
import pandas as pd
import json


def load_data(directory):
    ret = lines_if_valid(directory, "*.jsonl")
    return pd.DataFrame.from_records(ret)


def calculate_balanced_accuracy(truths, pred):
    truths.replace({"watermarked": 1, "not-watermarked": 0}, inplace=True)
    score = balanced_accuracy_score(truths.to_list(), pred.to_list())
    if np.isnan(score):
        return None
    return float(score)


def calculate_bleu(orig, water):
    bleu_sum = 0
    for i in range(len(orig)):
        text_id = orig["id"][i]
        orig_text = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(orig["text"][i])]
        water_text = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(water.loc[water["id"] == text_id]["text"].values[0])]
        
        text_bleu_sum = 0
        text_count = 0
        max_len = min(len(orig_text), len(water_text))
        for s_i in range(max_len):
            score = nltk.translate.bleu_score.sentence_bleu([orig_text[s_i]], water_text[s_i], smoothing_function=SmoothingFunction().method1)
            text_bleu_sum += score
            text_count += 1
        
        if text_count > 0:
            bleu_sum += text_bleu_sum / text_count
        else:
            bleu_sum += 0

    return bleu_sum/len(orig)


def calculate_bert_score(orig, water):
    scorer = BERTScorer(model_type="roberta-large", lang="en")

    text_scores = []
    for i in range(len(orig)):
        text_id = orig["id"][i]
        orig_text = [s for s in nltk.sent_tokenize(orig["text"][i])]
        water_text = [s for s in nltk.sent_tokenize(water.loc[water["id"] == text_id]["text"].values[0])]
        max_len = min(len(orig_text), len(water_text))

        if max_len == 0:
            continue

        orig_sents = orig_text[:max_len]
        water_sents = water_text[:max_len]

        _, _, scores = scorer.score(water_sents, orig_sents)
        text_scores.append(scores.mean().item())
    
    if not text_scores:
        return None
    
    return float(np.mean(text_scores))


@click.command()
@click.argument("watermarked_texts", type=Path)
@click.argument("original_texts", type=Path)
@click.argument("labels", type=Path)
@click.option("--output-directory", type=Path, required=False, help="The output directory.")
def main(watermarked_texts, original_texts, labels, output_directory):
    orig = load_data(original_texts)
    water = load_data(watermarked_texts)
    labels = load_data(labels)
    truths = labels["truth_label"]
    predictions = labels["label"]

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for text_id, truth in truths.items():
        pred = predictions[text_id]
        if truth == "watermarked" and pred == 1.0:
            true_positives += 1
        elif truth == "not-watermarked" and pred == 0.0:
            true_negatives += 1
        elif truth == "watermarked" and pred == 0.0:
            false_negatives += 1
        elif truth == "not-watermarked" and pred == 1.0:
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

