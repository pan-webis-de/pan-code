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
        y.append(line["labels"])
        work_id.append(line["work_id"])

    return work_id, x_text, np.asarray(y)




