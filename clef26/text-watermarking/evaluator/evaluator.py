#!/usr/bin/env python3
from tira.check_format import lines_if_valid
from pathlib import Path
import click
from tira.io_utils import to_prototext
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from torchmetrics.text.bert import BERTScore
import pandas as pd
import json
from tqdm import tqdm


def load_data(directory):
    ret = []
    with open(str(directory), mode='r', encoding="utf-8") as infile:
        for line in infile:
            ret.append(json.loads(line))
    return pd.DataFrame.from_records(ret)


def calculate_balanced_accuracy(truths, pred):
    truths.replace({"watermarked": 1, "not-watermarked": 0}, inplace=True)
    score = balanced_accuracy_score(truths.to_list(), pred.to_list())
    if np.isnan(score):
        return None
    return float(score)


def calculate_bleu(orig, water):
    bleu_sum = 0
    for i in tqdm(range(len(orig)), desc="Calculating bleu..."):
        text_id = orig["id"][i]
        orig_text = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(orig["text"][i])]
        water_text = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(water.loc[water["id"] == text_id]["text"].values[0])]
        
        text_bleu_sum = 0
        text_count = 0
        for s_i in range(len(orig_text)):
            if len(water_text) <= s_i:
                break
            score = nltk.translate.bleu_score.sentence_bleu(orig_text[s_i], water_text[s_i], smoothing_function=SmoothingFunction().method1)
            text_bleu_sum += score
            text_count += 1
        bleu_sum += text_bleu_sum / text_count

    return bleu_sum/len(orig)


def calculate_bert_score(orig, water):
    bertscore = BERTScore(model_name_or_path="roberta-large", truncation=True)
    bert_sum = 0
    for i in tqdm(range(len(orig)), desc="Calculating bert..."):
        text_id = orig["id"][i]
        orig_text = [s for s in nltk.sent_tokenize(orig["text"][i])]
        water_text = [s for s in nltk.sent_tokenize(water.loc[water["id"] == text_id]["text"].values[0])]
        l = min(len(orig_text), len(water_text))
        orig_text = orig_text[:l]
        water_text = water_text[:l]
        
        score = bertscore(orig_text, water_text)
        bert_sum += sum(score['f1'])/len(score['f1']) if len(score['f1'].size()) > 0 else 0
    return bert_sum.item()/len(orig)


@click.command()
@click.argument("watermarked_texts", type=Path)
@click.argument("original_texts", type=Path)
@click.argument("labels", type=Path)
@click.option("--output-directory", type=Path, required=False, help="The output directory.")
def main(original_texts, watermarked_texts, labels, output_directory):
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

