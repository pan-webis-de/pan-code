#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an n-gram baseline for the PAN20 shared task on celebrity profiling.
For usage information, call the help:
~# python3 pan20-celebrity-profiling-ngram-baseline.py --help
"""
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
import logging
import click

# Regular expressions for preprocessing
text_re = re.compile("[^a-zA-Z\s]")
url_re = re.compile("http(s)*://[\w]+\.(\w|/)*(\s|$)")
hashtag_re = re.compile("[\W]#[\w]*[\W]")
mention_re = re.compile("(^|[\W\s])@[\w]*[\W\s]")
smile_re = re.compile("(:\)|;\)|:-\)|;-\)|:\(|:-\(|:-o|:o|<3)")
emoji_re = re.compile("(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])")
not_ascii_re = re.compile("([^\x00-\x7F]+)")
time_re = re.compile("(^|\D)[\d]+:[\d]+")
numbers_re = re.compile("(^|\D)[\d]+[.'\d]*\D")
space_collapse_re = re.compile("[\s]+")

# numerical encoding of the classes
g_dict = {'male': 0, 'female': 1}
inv_g_dict = {0: 'male', 1: 'female'}
o_dict = {"sports": 0, "performer": 1, "creator": 2, "politics": 3}
inv_o_dict = {0: "sports", 1: "performer", 2: "creator", 3: "politics"}

# Hyperparameters
N_GRAM_RANGE = (1, 2)
MAX_WORD_FEATURES = 10000
MAX_TWEETS_PER_USER = 10000
MAX_FOLLOWERS_PER_CELEBRITY = 10


def _preprocess_feed(tweet: str):
    """ takes the original tweet text and returns the preprocessed texts.
    Preprocessing includes:
        - lowercasing
        - replacing hyperlinks with <url>, mentions with <user>, time, numbers, emoticons, emojis
        - removing additional non-ascii special characters
        - collapsing spaces
    """
    t = tweet.lower()
    t = re.sub(url_re, " <URL> ", t)
    t = t.replace("\n", "")
    t = t.replace("#", " <HASHTAG> ")
    t = re.sub(mention_re, " <USER> ", t)
    t = re.sub(smile_re, " <EMOTICON> ", t)
    t = re.sub(emoji_re, " <EMOJI> ", t)
    t = re.sub(time_re, " <TIME> ", t)
    t = re.sub(numbers_re, " <NUMBER> ", t)
    t = re.sub(not_ascii_re, "", t)
    t = re.sub(space_collapse_re, " ", t)
    t = t.strip()
    return t


def _read_text_linewise(p, mode):
    """ load each celebrity, concat the first 500 tweets and add a separator token between each """
    if mode == 'celeb':
        for line in open(p, "r"):
            yield " <eotweet> ".join(json.loads(line)["text"][:MAX_TWEETS_PER_USER])
    elif mode == 'follow':
        for line in open(p, "r"):
            yield " <eofollower> ".join([" <eotweet> ".join(follower[:MAX_TWEETS_PER_USER])
                                         for follower in json.loads(line)["text"][:MAX_FOLLOWERS_PER_CELEBRITY]])


def _get_age_class(by):
    """ convert the birthyears of a certain range to the center point.
     This is to reduce the number of classes when doing a classification model over regression on age
     """
    by = int(by)
    if 1940 <= by <= 1955:
        return 1947
    elif 1956 <= by <= 1969:
        return 1963
    elif 1970 <= by <= 1980:
        return 1975
    elif 1981 <= by <= 1989:
        return 1985
    elif 1990 <= by <= 1999:
        return 1994


def load_dataset(dataset_path: str, mode: str, vectorizer_path: str):
    """
    load the dataset, preprocess the texts for ML and build a feature matrix
    :param dataset_path: path to the dataset to be loaded
    :param mode: 'celeb' or 'follow' to load the celebrity feed or the follower feeds
    :param vectorizer_path: Path to a stored vectorizer which will be loaded from there if available or created and
                            stored there.
    :return: x, - x is the feature matrix for the texts
             y_age, y_gender, y_occ, - y are the targets for each labels
             ids - the ids identifying the indices of x and y
    """
    if mode == "celeb":
        x_path = dataset_path + "/celebrity-feeds.ndjson"
    else:
        x_path = dataset_path + "/follower-feeds.ndjson"
    y_data = [json.loads(line) for line in open(dataset_path + "/labels.ndjson", "r")]

    if not Path(vectorizer_path).exists():
        logging.info("no stored vectorizer found, creating ...")
        vec = TfidfVectorizer(preprocessor=_preprocess_feed, ngram_range=N_GRAM_RANGE,
                              max_features=MAX_WORD_FEATURES, analyzer='word', min_df=3)
        vec.fit(_read_text_linewise(x_path, mode))
        joblib.dump(vec, vectorizer_path)
    else:
        logging.info("loading stored vectorizer")
        vec = joblib.load(vectorizer_path)

    # load x data
    logging.info("transforming data ...")
    x = vec.transform(_read_text_linewise(x_path, mode))

    # load Y data
    y_gender = [g_dict[l["gender"]] for l in y_data]
    y_occ = [o_dict[l["occupation"]] for l in y_data]
    y_age = [_get_age_class(l["birthyear"]) for l in y_data]
    ids = [i["id"] for i in y_data]
    return x, y_age, y_gender, y_occ, ids


@click.command()
@click.option('-m', '--mode', default='follow', help='Which tweets to learn and predict on. Use "celeb" for profiling on the celebrity tweets and "follow" for profiling on the follower tweets')
@click.option('-v', '--vectorizer', default="../data/celeb-word-vectorizer.joblib", help='Path to a stored vectorizer. Will be created if missing.')
@click.option('--training_dir', required=True, help='Path to the directory holding the respective feeds.ndjson and labels.ndjson from the training set')
@click.option('--test_dir', required=True, help='Path to the directory holding the respective feeds.ndjson and labels.ndjson from the test set')
def logreg(mode, vectorizer, training_dir, test_dir):
    """ Main method for the baselines. It wires data loading, model training, and evaluation.
    The model used is a simple, linear LogisticRegression from sklearn.
    The method predicts on the given test dataset and writes the results to a labels.ndjson.
    Use the evaluator from the celebrity profiling task at PAN2020 to evaluate the results.
    """
    # 1. load the training dataset
    logging.basicConfig(level=logging.INFO)
    logging.info("loading training dataset")
    x_train, y_train_age, y_train_gender, y_train_occ, _ = \
        load_dataset(training_dir, mode, vectorizer)

    # 2. train models
    logging.info("fitting model age")
    age_model = LogisticRegression(multi_class='multinomial', solver="newton-cg")
    age_model.fit(x_train, y_train_age)
    logging.info("fitting model gender")
    gender_model = LogisticRegression(multi_class='multinomial', solver="newton-cg")
    gender_model.fit(x_train, y_train_gender)
    logging.info("fitting model acc")
    occ_model = LogisticRegression(multi_class='multinomial', solver="newton-cg")
    occ_model.fit(x_train, y_train_occ)

    # 3. load the test dataset
    logging.info("loading test dataset ...")
    x_test, y_test_age, y_test_gender, y_test_occ, cid = \
        load_dataset(test_dir, mode, vectorizer)

    # 4. Predict and Evaluate
    logging.info("predicting")
    age_pred = age_model.predict(x_test)
    gender_pred = gender_model.predict(x_test)
    occ_pred = occ_model.predict(x_test)
    output_labels = [{"id": i, "birthyear": int(a), "gender": inv_g_dict[g], "occupation": inv_o_dict[o]}
                     for i, a, g, o in zip(cid, age_pred, gender_pred, occ_pred)]

    open("labels.ndjson", "w").writelines(
        [json.dumps(x) + "\n" for x in output_labels]
    )


if __name__ == "__main__":
    logreg()
