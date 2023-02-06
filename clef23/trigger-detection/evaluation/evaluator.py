"""
Validator for the shared task on Trigger Detection at PAN@CLEF2023.

This script checks if the given input file is a valid submission to the task.

You can check you model output (on the validation date) in code via:
    `_validate(path_to_model_output, path_to_truth)`
or from the command line:
    ~$ python3 validator.py -p path-to-model-output -t path-to-truth

Contact: matti.wiegmann@uni-weimar.de
         or create an issue/PR on Github: https://github.com/pan-webis-de/pan-code
"""
from typing import List, Dict, Tuple
import click
from pathlib import Path
import json
from util import LABELS, to_array_representation
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
        - micro F1 scores for only high-frequency labels
        - micro F1 scores for only mid-frequency labels
        - micro F1 scores for only bottom-frequency labels
        - class wise (micro) F1 scores
    """
    results = {
        "mic_p": precision_score(y_true, y_predictions, average='micro'),
        "mic_r": recall_score(y_true, y_predictions, average='micro'),
        "mic_f1": f1_score(y_true, y_predictions, average='micro'),
        "mac_p": precision_score(y_true, y_predictions, average='macro'),
        "mac_r": recall_score(y_true, y_predictions, average='macro'),
        "mac_f1": f1_score(y_true, y_predictions, average='macro')
    }
    if extended:
        results["sub_acc"] = accuracy_score(y_true, y_predictions)
        results["roc_auc"] = roc_auc_score(y_true, y_predictions)
        results["roc_auc"] = None

    return results


def write_evaluations(results: dict, output_directory: Path, form: str = 'protobuf'):
    """ Write the evaluation results to a file.
     @param results: A dictionary with evaluation results
     @param output_directory: A directory where to write the output file to.
     @param form: in which format to write the results, protobuf or json
     """
    output_directory.mkdir(parents=True, exist_ok=True)

    def _write_protobuf():
        with open(output_directory / "evaluation.prototext") as of:
            for k, v in results:
                of.write('measure{{\n  key: "{}"\n  value: "{}"\n}}\n'.format(k, str(v)))

    def _write_json():
        with open(output_directory / "evaluation.json") as of:
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
@click.option('-o', '--output-dir', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default="./", help='Path where to write the output to')
@click.option('-f', '--output-format', type=str,
              default="protobuf", help='Path where to write the output to')
@click.option('-e', '--extended', type=bool, default=False, help='If set, also compute the extended metrics.')
def evaluate(predictions: str, truth: str, output_dir: str, output_format: str, extended: bool):
    """ This the cli command for the validator. The CLI checks if the passed files exists and then calls `_validate` """
    truth = Path(truth)
    predictions = Path(predictions)

    assert _validate(truth, predictions)
    t = _load(truth)
    p = _load(predictions)
    results = _evaluate(t, p, extended)
    write_evaluations(results, output_directory=Path(output_dir), form=output_format)


if __name__ == '__main__':
    evaluate()
