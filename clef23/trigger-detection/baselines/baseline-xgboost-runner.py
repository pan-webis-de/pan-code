"""
XGBoost baseline for the shared task on Trigger Detection at PAN23.

This is the runner script. It contains code to load a pretrained model and execute it on the passed dataset.
"""
import logging
from typing import List
import click
from pathlib import Path

import joblib

from util import load_data, write_predictions

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)


def vectorize(x_text: List[str], savepoint: Path):
    logging.info("load vectorizer")
    vec = joblib.load(savepoint / "vectorizer.joblib")
    logging.info("load feature selection")
    x = vec.transform(x_text)
    if (savepoint / "feature-selector.joblib").exists():
        feature_selector = joblib.load(savepoint / "feature-selector.joblib")
        x = feature_selector.transform(x)
    return x


def run_experiment(input_dataset_dir: Path, output_dir: Path, savepoint: Path):
    logging.info("load validation data")
    work_id, x_text, _ = load_data(input_dataset_dir)
    x = vectorize(x_text, savepoint)
    clf = joblib.load(savepoint / "clf-ovr.joblib")
    y_predicted = clf.predict(x)
    write_predictions(output_dir, work_id, y_predicted)


@click.option('-i', '--input-dataset-dir', type=click.Path(exists=True, file_okay=False),
              help="Path to the works.jsonl.")
@click.option('-o', '--output-dir', type=click.Path(exists=True, file_okay=False), default="./output",
              help="Path where to write the output file")
@click.option('-s', '--savepoint', type=click.Path(exists=True, file_okay=False), default="/baseline/model",
              help="Path to the saved model. Should hava a model.joblib and a tokenizer.joblib")
@click.command()
def run(input_dataset_dir, output_dir, savepoint):
    """
    $ python3 baseline-xgboost-runner.py \
        -i "<input-data-path>" \
        -o "<output-dir>" \
        -s "<saved-model-dir>"
    """
    run_experiment(Path(input_dataset_dir), Path(output_dir), Path(savepoint))


if __name__ == "__main__":
    run()
