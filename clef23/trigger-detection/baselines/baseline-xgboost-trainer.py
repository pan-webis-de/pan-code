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
import json

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

logging.basicConfig(filename=f"logs/log-xgboost-{dt.now().isoformat()}.log", encoding='utf-8', level=logging.WARN)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# NOTE These are the parameters for the ablation study
ABL_MODEL_PARAM = {'max_depth': [2, 3, 4],
                   'learning_rate': [0.25, 0.5, 0.75],
                   }
ABL_VEC_PARAMS = [
    # {'dir': 'word-1-all', 'n_gram_range': (1, 1), 'analyzer': 'word', 'f_select': 'None'},
    # {'dir': 'word-1-chi2', 'n_gram_range': (1, 1), 'analyzer': 'word', 'f_select': 'chi2'},
    # {'dir': 'word-2-all', 'n_gram_range': (1, 2), 'analyzer': 'word', 'f_select': 'None'},
    # {'dir': 'word-2-chi2', 'n_gram_range': (1, 2), 'analyzer': 'word', 'f_select': 'chi2'},
    # {'dir': 'word-3-all', 'n_gram_range': (1, 3), 'analyzer': 'word', 'f_select': 'None'},
    # {'dir': 'word-3-chi2', 'n_gram_range': (1, 3), 'analyzer': 'word', 'f_select': 'chi2'},
    # {'dir': 'char-3-all', 'n_gram_range': (3, 3), 'analyzer': 'char', 'f_select': 'None'},
    # {'dir': 'char-3-chi2', 'n_gram_range': (3, 3), 'analyzer': 'char', 'f_select': 'chi2'},
    {'dir': 'char-5-chi2', 'n_gram_range': (3, 5), 'analyzer': 'char', 'f_select': 'chi2'}]


def fit_vectorizer(x_text, y, savepoint: Path, fit: bool = False,
                   n_gram_range=(3, 3), min_df=5, analyzer='char', f_select='None', **kwargs) -> Tuple[Iterable, Iterable]:
    """
    :param x_text: the texts
    :param y: the labels
    :param savepoint: Path where to save the vectorizer
    :param fit: if true, we load the vectorizer from savepoint. If false, we fit a new one and save it.
    :param n_gram_range: vectorizer parameter, set here for ablation
    :param min_df: vectorizer parameter, set here for ablation
    :param analyzer: vectorizer parameter, set here for ablation
    :param f_select: feature selection to use. 'None" or 'chi2'
    :param subdir: the subdir where to save the vectorizer or None
    :return:
    """
    if fit:
        logger.debug("fit vectorizer")
        vec = TfidfVectorizer(lowercase=False, analyzer=analyzer, ngram_range=n_gram_range, min_df=min_df)
        x = vec.fit_transform(x_text)
        _time()
        joblib.dump(vec, savepoint / "vectorizer.joblib")

        if f_select is None or f_select == 'None':
            return x, y

        logger.debug("fit feature selection")
        if f_select == 'chi2':
            feature_selector = SelectKBest(chi2, k=10000)
        else:
            raise AttributeError(f"f_select can not be {f_select}")
        x = feature_selector.fit_transform(x, y)
        _time()
        joblib.dump(feature_selector, savepoint / "feature-selector.joblib")
    else:
        logger.debug("load vectorizer")
        vec = joblib.load(savepoint / "vectorizer.joblib")
        logger.debug("load feature selection")
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
    parameters = {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.25}
    clf = xgb.XGBClassifier(tree_method="hist", n_estimators=parameters['n_estimators'],
                            early_stopping_rounds=10,
                            max_depth=parameters['max_depth'], learning_rate=parameters['learning_rate'], n_jobs=16)

    if ablate:
        split_index = [-1] * len(y_train) + [0] * len(y_validation)
        x = vstack((x_train, x_validation))
        y = np.vstack((y_train, y_validation))
        ps = PredefinedSplit(test_fold=split_index)
        gs = GridSearchCV(clf, {'max_depth': max_depth,
                                'learning_rate': learning_rate}, verbose=1, cv=ps, scoring='f1_macro')
        gs.fit(x, y, eval_set=[(x_validation, y_validation)])
        logger.info(f"Best score in grid search: {gs.best_score_}")
        logger.info(f"Best parameters in grid search: {gs.best_params_}")
        be = gs.best_estimator_
        parameters = gs.best_params_
    else:
        clf.fit(x_train, y_train, eval_set=[(x_validation, y_validation)])
        be = clf

    _time()
    logger.info(f"save model to {savepoint}")
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
    logger.debug(f"Run Ablation: {ablate}")

    def _run(xt, yt, xv, yv, vectorizer_params: Dict, model_params: Dict, savepoint: Path):
        savepoint = savepoint if not vectorizer_params["dir"] else savepoint / vectorizer_params["dir"]
        savepoint.mkdir(exist_ok=True)
        logger.debug("fit training vectorizer")
        x_train, y_train = fit_vectorizer(xt, yt, savepoint, fit=True, **vectorizer_params)
        logger.debug("vectorize validation data")
        x_validation, y_validation = fit_vectorizer(xv, yv, savepoint, **vectorizer_params)

        logger.info("train model")
        y_predicted, parameters = _train_model(x_train, y_train, x_validation, y_validation, savepoint, ablate=ablate, **model_params)

        logger.info(f"Vectorizer Parameters: {vectorizer_params}")
        logger.info(f"Model Parameters: {parameters}")
        micro_f1 = f1_score(y_validation, y_predicted, average='micro')
        macro_f1 = f1_score(y_validation, y_predicted, average='macro')
        logger.info(f"trained with validation scores of {macro_f1} macro f1 and {micro_f1} micro f1")
        logger.info(f"Classification report on the validation data: {classification_report(y_validation, y_predicted)}")
        results = {**vectorizer_params, **parameters, "micro_f1": micro_f1, "macro_f1": macro_f1}
        open(savepoint / 'results.json', 'w').write(json.dumps(results))
        return results

    logger.info("load training data")
    _, x_train_text, y_train = load_data(training_dataset_dir)
    logger.info("load validation data")
    _, x_validation_text, y_validation = load_data(validation_dataset_dir)

    if ablate:
        for vectorizer_parameter in ABL_VEC_PARAMS:
            _run(x_train_text, y_train, x_validation_text, y_validation, vectorizer_parameter, ABL_MODEL_PARAM, savepoint)
    else:
        _run(x_train_text, y_train, x_validation_text, y_validation, {'dir': None}, {}, savepoint=savepoint)


@click.option('--training', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Path to the pan23-trigger-detection-train directory (from the PAN23 distribution). It should'
                   'contain a works.jsonl with the `labels` key')
@click.option('--validation', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Path to the pan23-trigger-detection-validation (or test) directory (from the PAN23 distribution). '
                   'It should contain a works.jsonl with the `labels` key')
@click.option('-s', '--savepoint', type=click.Path(exists=False, file_okay=False),
              default="./models/xgb-baseline",
              help="Path where to store the trained model. Will be overwritten if it already exists.")
@click.option('-a', '--ablate', type=bool, default=False, is_flag=True,
              help='If set, run the ablation study.')
@click.command()
def train(training, validation, savepoint, ablate):
    """
    Use the following command to train a model and save it to the default directoy.

    python3 baseline-xgboost-trainer.py \
        --training "<input-dataset-path>" \
        --validation "<validation-path>"

    """
    Path(savepoint).mkdir(parents=True, exist_ok=True)
    run_trainer(Path(training), Path(validation), Path(savepoint), ablate)


if __name__ == "__main__":
    train()
