"""
XGBoost baseline for the shared task on Trigger Detection at PAN23.

This is the trainer script. It contains code to fit the vectorizer, train the model, and save both to disk.
The runner script will then load model and vectorizer and make predictions.

This file also contains the code for the ablation study which can be run on one dataset (excluding data balance).
"""
import logging
from typing import Tuple, Iterable, Dict, List
import click
from pathlib import Path

from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from datetime import datetime as dt
from scipy.sparse import vstack
import numpy as np
from util import _time, load_data

import joblib
import xgboost as xgb

logging.basicConfig(filename=f"logs/log-{dt.now().isoformat()}.log", encoding='utf-8', level=logging.DEBUG)

# NOTE These are the parameters for the ablation study
ABL_MODEL_PARAM = {'max_depth': [2, 4],
                   'learning_rate': [0.25, 0.5, 1],
                   # 'n_estimators': [50, 75, 100]
                   }
ABL_VEC_PARAMS = [
    {'n_gram_range': (1, 1), 'analyzer': 'word', 'f_select': 'None'},
    {'n_gram_range': (1, 1), 'analyzer': 'word', 'f_select': 'chi2'},
    {'n_gram_range': (1, 2), 'analyzer': 'word', 'f_select': 'None'},
    {'n_gram_range': (1, 2), 'analyzer': 'word', 'f_select': 'chi2'},
    {'n_gram_range': (1, 3), 'analyzer': 'word', 'f_select': 'None'},
    {'n_gram_range': (1, 3), 'analyzer': 'word', 'f_select': 'chi2'},
    {'n_gram_range': (3, 3), 'analyzer': 'char', 'f_select': 'None'},
    {'n_gram_range': (3, 3), 'analyzer': 'char', 'f_select': 'chi2'},
    {'n_gram_range': (3, 5), 'analyzer': 'char', 'f_select': 'None'},
    {'n_gram_range': (3, 5), 'analyzer': 'char', 'f_select': 'chi2'}]


def fit_vectorizer(x_text, y, savepoint: Path, fit: bool = False,
                   n_gram_range=(1, 2), min_df=5, analyzer='word', f_select='chi2') -> Tuple[Iterable, Iterable]:
    """
    :param x_text: the texts
    :param y: the labels
    :param savepoint: Path where to save the vectorizer
    :param fit: if true, we load the vectorizer from savepoint. If false, we fit a new one and save it.
    :param n_gram_range: vectorizer parameter, set here for ablation
    :param min_df: vectorizer parameter, set here for ablation
    :param analyzer: vectorizer parameter, set here for ablation
    :param f_select: feature selection to use. 'None" or 'chi2'
    :return:
    """

    if fit:
        logging.info("fit vectorizer")
        vec = TfidfVectorizer(lowercase=False, analyzer=analyzer, ngram_range=n_gram_range, min_df=min_df)
        x = vec.fit_transform(x_text)
        _time()
        joblib.dump(vec, savepoint / "vectorizer.joblib")

        if f_select is None or f_select == 'None':
            return x, y

        logging.info("fit feature selection")
        if f_select == 'chi2':
            feature_selector = SelectKBest(chi2, k=10000)
        else:
            raise AttributeError(f"f_select can not be {f_select}")
        x = feature_selector.fit_transform(x, y)
        _time()
        joblib.dump(feature_selector, savepoint / "feature-selector.joblib")
    else:
        logging.info("load vectorizer")
        vec = joblib.load(savepoint / "vectorizer.joblib")
        logging.info("load feature selection")
        x = vec.transform(x_text)
        if f_select == 'chi2':
            feature_selector = joblib.load(savepoint / "feature-selector.joblib")
            x = feature_selector.transform(x)

    return x, y


def _train_model(x_train, y_train, x_validation, y_validation, savepoint: Path, ablate=True,
                 max_depth: List[int] = None, learning_rate: List[float] = None, n_estimators: List[int] = None,
                 **kwargs) -> Tuple[Iterable, Dict]:
    """
    Train a XGB model on the given data.
    :param x_train: training features (scipy sparse matrix)
    :param y_train: training labels (numpy array)
    :param x_validation: validation features (scipy sparse matrix)
    :param y_validation: validation labels (numpy array)
    :param savepoint: where to save the trained model to
    :param ablate: run the ablation study (gridsearch over multiple values)
    :return: (y_predicted, parameters) the predicted labels on the validation split
    """
    _time(True)
    clf = xgb.XGBClassifier(tree_method="hist", n_estimators=200, max_depth=2, learning_rate=1, n_jobs=16)

    if ablate:
        split_index = [-1] * len(y_train) + [0] * len(y_validation)
        x = vstack((x_train, x_validation))
        y = np.vstack((y_train, y_validation))
        ps = PredefinedSplit(test_fold=split_index)
        gs = GridSearchCV(clf, {'max_depth': max_depth,
                                'learning_rate': learning_rate}, verbose=1, cv=ps, scoring='f1_macro')
        gs.fit(x, y, early_stopping_rounds=10, eval_set=[(x_validation, y_validation)])
        logging.info(f"Best score in grid search: {gs.best_score_}")
        logging.info(f"Best parameters in grid search: {gs.best_params_}")
        be = gs.best_estimator_
        parameters = gs.best_params_
    else:
        clf.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], early_stopping_rounds=10)
        be = clf
        parameters = {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 1}

    _time()
    logging.info(f"save model to {savepoint}")
    joblib.dump(be, savepoint / "clf-ovr.joblib")
    predictions = be.predict(x_validation)
    return predictions, parameters


def run_trainer(training_dataset_dir: Path, validation_dataset_dir: Path, savepoint: Path, ablate=False) -> None:
    """ Train the model. Here we also control the ablation.

    Ablated parameters are:
        - data sample: (fixed a-priory and passed via training_dataset_dir and validation_dataset_dir)
        - tokenizer:  n_gram_range, analyzer, f_select
        - Model: grid search over 'max_depth', 'learning_rate', and 'n_estimators'

    :param training_dataset_dir: A directory with a works.jsonl used for training
    :param validation_dataset_dir: A directory with a works.jsonl used for validation
    :param savepoint: Where to save the model and vectorizer to
    :param ablate: If True, run the ablation study.
    :return: None
    """
    logging.debug(f"Run Ablation: {ablate}")

    def _run(xt, yt, xv, yv, vectorizer_params: Dict, model_params: Dict):
        logging.warning(f"Vectorizer Parameters: {vectorizer_params}")
        logging.info("fit training vectorizer")
        x_train, y_train = fit_vectorizer(xt, yt, savepoint, fit=True, **vectorizer_params)
        logging.info("vectorize validation data")
        x_validation, y_validation = fit_vectorizer(xv, yv, savepoint, **vectorizer_params)

        logging.info("train model")
        y_predicted, parameters = _train_model(x_train, y_train, x_validation, y_validation, savepoint, ablate=ablate, **model_params)

        logging.warning(f"Model Parameters: {parameters}")
        logging.warning(
            f"Classification report on the validation data: {classification_report(y_validation, y_predicted)}"),
        logging.warning(f"trained with validation scores of {f1_score(y_validation, y_predicted, average='macro')} "
                        f"macro f1 and {f1_score(y_validation, y_predicted, average='micro')} micro f1")

    logging.info("load training data")
    _, x_train_text, y_train = load_data(training_dataset_dir)
    logging.info("load validation data")
    _, x_validation_text, y_validation = load_data(validation_dataset_dir)

    if ablate:
        for ablation_parameter in ABL_VEC_PARAMS:
            _run(x_train_text, y_train, x_validation_text, y_validation, ablation_parameter, ABL_MODEL_PARAM)
    else:
        _run(x_train_text, y_train, x_validation_text, y_validation, {}, {})


@click.option('--training', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Path to the pan23-trigger-detection-train directory (from the PAN23 distribution). It should'
                   'contain a works.jsonl with the `labels` key')
@click.option('--validation', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Path to the pan23-trigger-detection-validation (or test) directory (from the PAN23 distribution). '
                   'It should contain a works.jsonl with the `labels` key')
@click.option('-s', '--savepoint', type=click.Path(exists=False, file_okay=False),
              default="../models/xgboost-baseline-full",
              help="Path where to store the trained model. Will be overwritten if it already exists.")
@click.option('-a', '--ablate', type=bool, default=False, is_flag=True,
              help='If set, run the ablation study.')
@click.command()
def train(training, validation, savepoint, ablate):
    """
    $ python3 baseline-xgboost-trainer.py \
        --training "/home/mike4537/data/pan23-trigger-detection/pan23-trigger-detection-train" \
        --validation "/home/mike4537/data/pan23-trigger-detection/pan23-trigger-detection-validation"


    $ python3 baseline-xgboost-trainer.py -a \
        --training "/home/mike4537/data/pan23-ml/pan23-trigger-detection-train" \
        --validation "/home/mike4537/data/pan23-ml/pan23-trigger-detection-validation"

    """
    Path(savepoint).mkdir(parents=True, exist_ok=True)
    run_trainer(Path(training), Path(validation), Path(savepoint), ablate)


if __name__ == "__main__":
    train()
