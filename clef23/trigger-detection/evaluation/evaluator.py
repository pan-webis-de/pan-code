#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluator for the shared task on Trigger Detection at PAN@CLEF2023.

This script checks if the given input file is a valid submission to the task and then calculates the evaluation results.

You can use the evaluator in-code by calling
    `_evaluate(y_true, y_predicted)`
  where y_true and y_predicted are 2-d array-likes (i.e. the array representation of the labels of this task)

Contact: matti.wiegmann@uni-weimar.de
         or create an issue/PR on Github: https://github.com/pan-webis-de/pan-code
"""
from typing import List, Dict, Tuple
import click
from pathlib import Path
import json
from util import LABELS, to_array_representation
import numpy as np
from validator import _validate
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score


def _load(file_path: Path) -> List:
    """ Load a labels.jsonl file, convert it to array representation and return the array.
     This function assumes that test and prediction files have the same order of works.
     """
    def _loader(line):
        return to_array_representation(json.loads(line)["labels"])

    return [_loader(_) for _ in open(file_path)]


def _evaluate(y_true: List[List[int]], y_predictions: List[List[int]], extended=False):
    """ Calculate the evaluation results.
    Extended results also include classwise labels that should not be integrated into tira but are usefull for
        evaluating model performance during development.

    Truth and Predictions must be in array representation.

    Basic evaluation shows:
        - micro and macro f1

    Extended evaluation shows:
        - subset accuracy
        - roc-auc score
        - micro F1 scores for only middle-frequency labels (top 2-15)
        - micro F1 scores for only bottom-frequency labels (top 16-bottom)
        - class wise (micro) F1 scores
    """
    y_true = np.asarray(y_true)
    y_predictions = np.asarray(y_predictions)
    results = {
        "mac_f1": round(f1_score(y_true, y_predictions, average='macro'), 4),
        "mac_p": round(precision_score(y_true, y_predictions, average='macro'), 4),
        "mac_r": round(recall_score(y_true, y_predictions, average='macro'), 4),
        "mic_f1": round(f1_score(y_true, y_predictions, average='micro'), 4),
        "mic_p": round(precision_score(y_true, y_predictions, average='micro'), 4),
        "mic_r": round(recall_score(y_true, y_predictions, average='micro'), 4),
        "sub_acc": round(accuracy_score(y_true, y_predictions), 4)
    }
    if extended:
        results["roc_auc"] = round(roc_auc_score(y_true, y_predictions), 4)
        results["mid_f1"] = round(f1_score(y_true[:, 1:15], y_predictions[:, 1:15], average='micro'), 4)
        results["mid_p"] = round(precision_score(y_true[:, 1:15], y_predictions[:, 1:15], average='micro'), 4)
        results["mid_r"] = round(recall_score(y_true[:, 1:15], y_predictions[:, 1:15], average='micro'), 4)
        results["bot_f1"] = round(f1_score(y_true[:, 15:], y_predictions[:, 15:], average='micro'), 4)
        results["bot_p"] = round(precision_score(y_true[:, 15:], y_predictions[:, 15:], average='micro'), 4)
        results["bot_r"] = round(recall_score(y_true[:, 15:], y_predictions[:, 15:], average='micro'), 4)
        for idx, label in enumerate(LABELS):
            results[f"{label}_f1"] = round(f1_score(y_true[:, idx], y_predictions[:, idx], average='micro', labels=[1]), 4)
            results[f"{label}_p"] = round(precision_score(y_true[:, idx], y_predictions[:, idx], average='micro', labels=[1]), 4)
            results[f"{label}_r"] = round(recall_score(y_true[:, idx], y_predictions[:, idx], average='micro', labels=[1]), 4)

    return results


def write_evaluations(results: dict, output_directory: Path, form: str = 'protobuf'):
    """ Write the evaluation results to a file.
     @param results: A dictionary with evaluation results
     @param output_directory: A directory where to write the output file to.
     @param form: in which format to write the results, protobuf or json
     """
    output_directory.mkdir(parents=True, exist_ok=True)
    print(results)

    def _write_protobuf():
        with open(output_directory / 'evaluation.prototext', 'w') as of:
            for k, v in results.items():
                of.write("measure{{\n  key: '{}'\n  value: '{}'\n}}\n".format(k, str(v)))

    def _write_json():
        with open(output_directory / "evaluation.json", 'w') as of:
            of.write(json.dumps(results))

    if form == 'protobuf':
        _write_protobuf()
    elif form == 'json':
        _write_json()
    else:
        raise ValueError(f"`form` must be either `protobuf` or `json`, not {form}")


@click.command()
@click.option('-p', '--predictions', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Path to the labels.jsonl that contains the predictions.')
@click.option('-t', '--truth', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Path to the labels.jsonl that contains truth corresponding to the given predictions.')
@click.option('-o', '--output-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default="./", help='Path where to write the output to')
@click.option('-f', '--output-format', type=str,
              default="json", help='Path where to write the output to')
@click.option('-e', '--extended', type=bool, default=False, is_flag=True,
              help='If set, also compute the extended metrics.')
def evaluate(predictions: str, truth: str, output_dir: str, output_format: str, extended: bool):
    """ This the cli command for the evaluator.
        - check if files exist and input are valid (by calling the validator)
        - load truth and predictions, calculate results, and write in the desired output format.

        ~$ python3 evaluator.py --extended -p "<model-output/labels.jsonl>" -t "<dataset-truth/labels.jsonl>" -o "./"

    """
    truth = Path(truth)
    predictions = Path(predictions)

    assert _validate(truth, predictions)
    t = _load(truth)
    p = _load(predictions)
    results = _evaluate(t, p, extended)
    write_evaluations(results, output_directory=Path(output_dir), form=output_format)


if __name__ == '__main__':
    evaluate()

