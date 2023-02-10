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
from typing import List, Dict
import click
from pathlib import Path
import json
from util import LABELS, to_array_representation


trigger_warnings = set(LABELS)


class ValidationError(Exception):
    pass


def _check_format(labels_file: Path) -> List[Dict[str, str | List[str]]]:
    """ check if
     - this file has a json loadable item on each line (which is utf-8 readable)
     - this has the correct keys (work_id and labels) for each example
     - the labels are in the label set and are convertable

     Note that the validator only checks the predictions file for correctness.
     """
    def __check(line):
        line = json.loads(line)
        if 'work_id' not in line:
            raise ValidationError(f"The file {labels_file} does not have the required key `work_id` on line {line}")
        if 'labels' not in line:
            raise ValidationError(f"The file {labels_file} does not have the required key `labels` on line {line}")
        if not isinstance(line['work_id'], str):
            raise ValidationError(f"The value of `work_id` should be a string in the file {labels_file} on line {line}")
        if not isinstance(line['labels'], list):
            raise ValidationError(f"The value of `labels` should be a list in the file {labels_file} on line {line}")
        for label in line['labels']:
            if not isinstance(label, str):
                raise ValidationError(f"The value of `labels` should be a list of strings "
                                      f"(check `util.py` for allowed values) "
                                      f"in {labels_file} on line {line}")
            if label not in trigger_warnings:
                raise ValidationError(f"The given file contains {label}, which is not an official label "
                                      f"(Do you use an old version of the dataset?). "
                                      f"The value of `labels` must be a list of official labels "
                                      f"(check `util.py` for a list)."
                                      f"in {labels_file} on line {line}")
        try:
            to_array_representation(line['labels'])
        except Exception as e:
            raise ValidationError(f"The value of `labels` can not be converted into its array representation", e)
        return line

    return [__check(line) for line in open(labels_file)]


def _all_keys(truth: List, predictions: List):
    """ check if
        - all works are there (same length)
        - all works are there (ID overlap)
        - same work is at the same position
    """
    if not len(truth) == len(predictions):
        raise ValidationError(f"The given prediction file contains {len(predictions)} items, but it should have {len(truth)}."
                              "Truth and predictions must contain the same works in the same order.")

    if set([_["work_id"] for _ in predictions]) != set([_["work_id"] for _ in truth]):
        raise ValidationError(f"The given prediction file has different works than the truth."
                              "Truth and predictions must contain the same works in the same order.")

    for t, p in zip(truth, predictions):
        if t['work_id'] != p['work_id']:
            raise ValidationError(f"The order of works in the predictions file are not the same as in the truth."
                                  "Truth and predictions must contain the same works in the same order.")


def _validate(truth_file: Path, predictions_file: Path) -> bool:
    """ This function checks if the predictions file is formatted correctly and gives hints if it is not. """

    if not truth_file.name == "labels.jsonl":
        raise ValidationError("The truth file should be called `labels.jsonl`.")
    if not predictions_file.name == "labels.jsonl":
        raise ValidationError("The file with the predictions should be called `labels.jsonl`.")

    truth = [json.loads(line) for line in open(truth_file)]
    predictions = _check_format(predictions_file)
    _all_keys(truth, predictions)
    return True


@click.command()
@click.option('-p', '--predictions', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Path to the labels.jsonl that contains the predictions.')
@click.option('-t', '--truth', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Path to the labels.jsonl that contains truth corresponding to the given predictions.')
def validate(predictions: str, truth: str):
    """ This the cli command for the validator. The CLI checks if the passed files exists and then calls `_validate` """
    truth = Path(truth)
    predictions = Path(predictions)

    _validate(truth, predictions)


if __name__ == '__main__':
    validate()
