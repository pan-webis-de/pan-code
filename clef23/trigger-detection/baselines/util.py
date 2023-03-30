import json
import logging
from typing import Tuple, List
from tqdm import tqdm
from pathlib import Path
from datetime import datetime as dt

from resiliparse.parse.html import HTMLTree
from resiliparse.extract.html2text import extract_plain_text
import re

import numpy as np
from numpy.typing import ArrayLike

re_punct = re.compile(r'[^\w\s]')
time = dt.now()

LABELS = ["pornographic-content", "violence", "death", "sexual-assault", "abuse", "blood", "suicide",
          "pregnancy", "child-abuse", "incest", "underage", "homophobia", "self-harm", "dying", "kidnapping",
          "mental-illness", "dissection", "eating-disorders", "abduction", "body-hatred", "childbirth",
          "racism", "sexism", "miscarriages", "transphobia", "abortion", "fat-phobia", "animal-death",
          "ableism", "classism", "misogyny", "animal-cruelty"]  # 32


def _time(silent=False):
    global time
    now = dt.now()
    if not silent:
        print(f"took {now - time}")
    time = now


def _preprocess(x: str) -> str:
    """ A minimalistic preprocessor: remove all html codes, all non-ascii characters, and lowercase """
    tree = HTMLTree.parse(x)
    x = extract_plain_text(tree, preserve_formatting=False)
    x = x.lower()
    x = re.sub(re_punct, '', x)
    return x


def load_data(dataset_dir: Path, preprocess: bool = True) -> Tuple[List, List, ArrayLike]:
    """
    Load a trigger detection dataset and return the id, texts, and labels.
    :param preprocess: if the loaded texts should be preprocessed
    :param dataset_dir: Path to the dataset to load.
    :return: work_id, x, y
    """
    x_text = []
    y = []
    work_id = []
    for line in tqdm(open(dataset_dir / "works.jsonl"), desc=f"loading data from {dataset_dir}"):
        line = json.loads(line)
        x_text.append(_preprocess(line["text"]) if preprocess else line["text"])
        if 'labels' in line.keys():
            y.append(line["labels"])
        work_id.append(line["work_id"])

    return work_id, x_text, np.asarray(y)


def write_predictions(output_dir: Path, work_ids: List[str], labels: ArrayLike) -> None:
    """
    Write the model predictions to a labels.jsonl that is expected by the evaluator
    :param output_dir: Path where to write the file to
    :param work_ids: a list of length n with work_ids
    :param labels: a List-like of length n, where each element is a list of labels (array form) and corresponds to the\
                   elment with the same ID in work_ids.
    :return: None
    """
    with open(output_dir / "labels.jsonl", 'w') as of:
        for wid, label_list in zip(work_ids, labels):
            result = {"work_id": wid, "labels": [LABELS[idx] for idx, cls in enumerate(label_list) if cls == 1]}
            of.write(f"{json.dumps(result)}\n")


