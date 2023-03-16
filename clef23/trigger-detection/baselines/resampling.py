"""
Utilities for the shared task on Trigger Detection at PAN23.

This script contains code to resample the dataset (as it is on zenodo) to modify the class balance.
"""
import json
import logging
from typing import Tuple, Union, List
from tqdm import tqdm
from pathlib import Path
# import random
import numpy as np
from numpy.random import default_rng
from numpy.typing import ArrayLike, DTypeLike
from datetime import datetime as dt
import click

from util import load_data

logging.basicConfig(filename=f'logs/log-resampling-{dt.now().isoformat()}', encoding='utf-8', level=logging.INFO)
# logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

random = default_rng(42)


def __class_count(labels: ArrayLike) -> Tuple[ArrayLike, List[int]]:
    """ Returns the number of number of examples per class in the array (in order of the input matrix).
        Also returns the class indices in descending order.
    :param labels: a (n x c) matrix of n examples and c binary classes. Entries are 0 or 1.
    :return:
    """
    class_counts = labels.sum(axis=0)
    class_order = [ind for ind, _ in sorted(list(enumerate(class_counts)), reverse=True, key=lambda x: x[1])]
    logging.debug(f"Loaded classes {class_order} and class counts {class_counts}")
    return class_counts, class_order


def __update_counts(old_sampled_class_ids, labels, all_sampled_ids):
    """
    update the given dict of label counts
    :param old_sampled_class_ids: a list of indices of examples {x: [idx, ...] for x in classes}
    :param labels:
    :param new_sample_from_class: a list of ids that should be added
    :return:
    """
    new_sampled_classes_count = {x: 0 for x in old_sampled_class_ids.keys()}
    new_sampled_class_ids = {x: [] for x in old_sampled_class_ids.keys()}

    for example_index in all_sampled_ids:
        # find all labels of the example at `example_index` in the original label matrix
        new_labels = np.asarray(labels[example_index] == 1).nonzero()[0]
        for nc in new_labels:
            new_sampled_classes_count[nc] += 1
            new_sampled_class_ids[nc].append(example_index)

    logging.debug(f"New class counts: {new_sampled_classes_count}")
    return new_sampled_classes_count, new_sampled_class_ids


def __get_sample_target(target_sample_count: Union[int | DTypeLike | list], source_class_size):
    """
    Determine the free slots for a class `cls` in the given sampling settings
    :param oversample: If oversampling is false, the target slots is at most source_class_size
    :return:
    """
    if isinstance(target_sample_count, list):
        target_slots = np.Infinity
        previous_distance = np.Infinity
        for free_slot_candidate in target_sample_count:
            if abs(free_slot_candidate - source_class_size) < previous_distance:
                previous_distance = abs(free_slot_candidate - source_class_size)
                target_slots = free_slot_candidate
    else:
        target_slots = target_sample_count
    return target_slots


def __sample(target_sample_count: Union[int | List[int]], source_labels: ArrayLike,
             class_count: List[int], classes: List[int],
             oversample: bool = True, undersample: bool = True) -> List[int]:
    """
    This method does the actual sampling.
    The oversampling only draws examples with 1 label.

    :param target_sample_count: int -> the number of examples which should be sampled
                                List -> a list of possible "steps" to which we can sample.
                                The sampler will check which of these "steps" is closest and sample that many examples.
    :param source_labels: the original label matrix
    :param classes: A list with the indices of the classes (ordered in reverse order)
    :param class_count: How many examples the class with idx has.
    :param oversample: True -> Examples will be oversampled to match `target_sample_count`.
                       False -> All examples of the rare classes will be added once
    :param undersample: True -> Examples will be randomly undersampled to match `target_sample_count`
                        False -> All examples of the frequent classes will be added once
    :return: A list of examples as indices from `source_labels`
    """
    logging.info(f"Start sampling with oversampling: {oversample} and undersampling: {undersample}")
    # classes = list(range(len(source_labels[0])))
    sampled_classes_count = {x: 0 for x in classes}  # This counts how many labels there are in total
    sampled_class_ids = {x: [] for x in classes}  # this tracks the indices of examples in the argument lists
    all_sampled_ids = []

    for cls in reversed(classes):
        cls_indices = np.asarray(source_labels[:, cls] == 1).nonzero()[0]
        logging.debug(f"Sample class {cls} ({len(cls_indices)} original examples)")

        # search the closest "step" to the current number of labels
        target_sample_size = __get_sample_target(target_sample_count, class_count[cls])
        # if we should oversample but oversample is false:
        if (target_sample_size > class_count[cls] and not oversample) or \
                (target_sample_size < class_count[cls] and not undersample):
            target_sample_size = class_count[cls]
        free_slots = target_sample_size - sampled_classes_count[cls]
        new_sample = []

        # There are fewer free slots than examples, now we undersample
        logging.debug(f"sample for class {cls} has  {free_slots} free slots.")

        if free_slots < 0:  # Problem. Happens through side effects. What to do?
            logging.warning(f"Class {cls} is too full by {free_slots}")
            continue
            # raise ValueError(f"Class {cls} is too full by {free_slots}")

        # Assign the leftovers to get the original sample, if there are free slots
        leftovers = [idx for idx in cls_indices if idx not in sampled_class_ids[cls]]
        if free_slots >= len(leftovers):
            logging.debug(f"Adding {len(leftovers)} leftover examples for class {cls} into {free_slots} free slots.")
            new_sample.extend(leftovers)
            free_slots -= len(leftovers)

        elif free_slots < len(leftovers):  # this happens by oversampling smaller classes with side effects.
            single_class_leftovers = {_ for _ in leftovers if sum(source_labels[_]) == 1}
            multi_class_leftovers = [_ for _ in leftovers if _ not in single_class_leftovers]
            logging.debug(f"Undersampling ({free_slots}) free slots from ({len(leftovers)}) leftovers: ({len(single_class_leftovers)}) single class and ({len(multi_class_leftovers)}) multi class examples.")

            # assert undersample
            if not undersample:
                logging.warning(f"Undersampling is false, but free slots ({free_slots}) is smaller than leftovers {len(leftovers)}. Undersample leftovers to fill the class.")

            if free_slots < len(single_class_leftovers) or free_slots < len(single_class_leftovers) * 3 or len(single_class_leftovers) > len(multi_class_leftovers) * 10:
                _ = random.choice(list(single_class_leftovers), free_slots)
                logging.debug(f"Added {len(_)} single label leftovers through undersamping.")
            else:
                # check how distorted the sample is
                _ = random.choice(leftovers, free_slots)
                logging.debug(f"Added {len(_)} (s/m label) leftovers through undersamping.")
            new_sample.extend(_)
            free_slots -= len(_)

        # All original classes are assigned. If there are still free slots, oversample.
        assert free_slots >= 0

        if free_slots > 0:
            single_class_examples = {_ for _ in cls_indices if sum(source_labels[_]) == 1}
            multi_class_examples = [_ for _ in cls_indices if _ not in single_class_examples]
            logging.debug(f"Oversampling ({free_slots}) free slots from {class_count[cls]} examples: ({len(single_class_examples)}) single class and ({len(multi_class_examples)}) multi class examples.")

            assert oversample  # if oversampling is false, there was an unexpected error before

            if free_slots < len(single_class_examples) or free_slots < len(single_class_examples) * 3 or len(single_class_examples) > len(multi_class_examples) * 10:
                _ = random.choice(list(single_class_examples), size=(free_slots))
                logging.debug(f"Added {len(_)} single label leftovers through undersamping.")
            else:
                _ = random.choice(cls_indices, size=(free_slots))
                logging.debug(f"Added {len(_)} (s/m label) leftovers through undersamping.")
            new_sample.extend(_)
            free_slots -= len(_)

        assert free_slots == 0
        assert sampled_classes_count[cls] + len(new_sample) == target_sample_size

        logging.debug(f"Adding {len(new_sample)} new examples to the sample.")
        all_sampled_ids.extend(new_sample)
        sampled_classes_count, sampled_class_ids = __update_counts(sampled_class_ids, source_labels, all_sampled_ids)

    logging.info(f"Sampled class counts: {sampled_classes_count}")
    return all_sampled_ids


def _resample_ruos_m(work_id: List[str], x: List[str], y: ArrayLike,
                     class_count: List[int], class_order: List[int]) -> Tuple[List[str], List[str], ArrayLike]:
    """ randomly stratify to the mean (rsm): stratify everything to the number of example that the
     example at the median index has. This means undersampling for the frequent half of labels,
     and oversampling for the rare half
     """
    median_index = class_order[round(len(class_order)/2)]  # index with the median sample count
    target_sample_count = class_count[median_index]
    logging.info(f"RUOS-M to median index ({median_index}) with {target_sample_count} examples")

    indices_of_new_data_sample = __sample(target_sample_count, y, class_count, class_order,
                                          oversample=True, undersample=True)

    return ([work_id[idx] for idx in indices_of_new_data_sample],
            [x[idx] for idx in indices_of_new_data_sample],
            np.asarray([y[idx] for idx in indices_of_new_data_sample]))


def _resample_rus_m(work_id: List[str], x: List[str], y: ArrayLike,
                    class_count: List[int], class_order: List[int]) -> Tuple[List[str], List[str], ArrayLike]:
    """ randomly stratify to the mean (rsm): stratify everything to the number of example that the
     example at the median index has. This means undersampling for the frequent half of labels,
     and oversampling for the rare half
     """
    median_index = class_order[round(len(class_order)/2)]  # index with the median sample count
    target_sample_count = class_count[median_index]
    logging.info(f"RUS-M to median index ({median_index}) with {target_sample_count} examples")

    indices_of_new_data_sample = __sample(target_sample_count, y, class_count, class_order,
                                          oversample=False, undersample=True)

    return ([work_id[idx] for idx in indices_of_new_data_sample],
            [x[idx] for idx in indices_of_new_data_sample],
            np.asarray([y[idx] for idx in indices_of_new_data_sample]))


def _resample_rus_top(work_id: List[str], x: List[str], y: ArrayLike,
                      class_count: List[int], class_order: List[int]) -> Tuple[List[str], List[str], ArrayLike]:
    """randomly undersample the top 1/4 most frequent labels (rus-top3):
    undersample the top 1/4 by frequency (i.e. cutoff). """
    target_index = class_order[round(len(class_order)*(1/4))]
    target_sample_count = class_count[target_index]
    logging.info(f"RUS-TOP to index {target_index} with {target_sample_count} examples")

    indices_of_new_data_sample = __sample(target_sample_count, y, class_count, class_order,
                                          oversample=False, undersample=True)

    return ([work_id[idx] for idx in indices_of_new_data_sample],
            [x[idx] for idx in indices_of_new_data_sample],
            np.asarray([y[idx] for idx in indices_of_new_data_sample]))


def _resample_ruos_ends(work_id: List[str], x: List[str], y: ArrayLike,
                        class_count: List[int], class_order: List[int]) -> Tuple[List[str], List[str], ArrayLike]:
    """ randomly undersample the top 1/4 most frequent labels and
        oversample everything below the 25th percentil (of classes)
    """
    undersample_index = class_order[round(len(class_order)*(1/4))]
    undersample_target_count = class_count[undersample_index]
    oversample_index = class_order[round(len(class_order)*(3/4))]
    oversample_target_count = class_count[oversample_index]
    logging.info(f"RUOS-ENDS undersample to index {undersample_index} with {undersample_target_count} examples and "
                 f"oversample to index {oversample_index} with {oversample_target_count} examples.")

    indices_of_new_data_sample = __sample(undersample_target_count, y, class_count, class_order,
                                          oversample=False, undersample=True)
    new_y = np.asarray([y[idx] for idx in indices_of_new_data_sample])
    class_count, class_order = __class_count(new_y)
    indices_of_new_data_sample2 = __sample(oversample_target_count,
                                           np.asarray([y[idx] for idx in indices_of_new_data_sample]),
                                           class_count, class_order,
                                           oversample=True,
                                           undersample=False)

    return ([work_id[idx] for idx in indices_of_new_data_sample2],
            [x[idx] for idx in indices_of_new_data_sample2],
            np.asarray([y[idx] for idx in indices_of_new_data_sample2]))


def _resample_ruos_q(work_id: List[str], x: List[str], y: ArrayLike,
                     class_count: List[int], class_order: List[int],
                     steps: int = 3) -> Tuple[List[str], List[str], ArrayLike]:
    """  randomly over and undersample to balance the quartils. This means we select the median element of each quartil
    and balance all others to this value"""
    # get order of classes
    step_halves = len(class_order) / (steps*2)
    target_indices = [class_order[round(i*step_halves)] for i in range(1, (steps*2)+1, 2)]
    target_sample_counts = [class_count[idx] for idx in target_indices]

    logging.info(f"RUOS-Q to the indices {target_indices} with {target_sample_counts} examples")

    indices_of_new_data_sample = __sample(target_sample_counts, y,
                                          class_count, class_order,
                                          oversample=True, undersample=True)

    return ([work_id[idx] for idx in indices_of_new_data_sample],
            [x[idx] for idx in indices_of_new_data_sample],
            np.asarray([y[idx] for idx in indices_of_new_data_sample]))


def _resample_rus_q(work_id: List[str], x: List[str], y: ArrayLike,
                    class_count: List[int], class_order: List[int],
                    steps: int = 3) -> Tuple[List[str], List[str], ArrayLike]:
    """ randomly undersample to the closest 3-quartil (rusq):
    undersample to 2*median, median, or 1/2*median, whatever is closer """
    step_halves = len(class_order) / (steps*2)
    target_indices = [class_order[round(i*step_halves)] for i in range(1, (steps*2)+1, 2)]
    target_sample_counts = [class_count[idx] for idx in target_indices]

    logging.info(f"RUS-Q to the indices {target_indices} with {target_sample_counts} examples")

    indices_of_new_data_sample = __sample(target_sample_counts, y,
                                          class_count, class_order,
                                          oversample=False, undersample=True)

    return ([work_id[idx] for idx in indices_of_new_data_sample],
            [x[idx] for idx in indices_of_new_data_sample],
            np.asarray([y[idx] for idx in indices_of_new_data_sample]))


def _write_dataset(work_id: List[str], x: List[str], y: ArrayLike,
                   output_dataset_dir: Path) -> None:
    """ write the dataset (in the parameters) in a works.jsonl into the output directory """
    with open(output_dataset_dir / "works.jsonl", 'w') as of:
        for w, text, labels in zip(work_id, x, y):
            data = {"work_id": w, "text": text, "labels": labels.tolist()}
            of.write(f"{json.dumps(data)}\n")


def resample(input_dataset_dir: Path, output_dataset_dir: Path, strategy: str = 'rsm') -> None:
    """ Resample the dataset to reduce class imbalance.

    Promising strategies:
    - randomly under and oversample to the mean (ruos-m): sample everything to the median index class frequency
    - randomly undersample everything above the 75th percentil (of classes) (rus-top): undersample the top 1/4 of classes
    - randomly undersample everything above the 75th percentil and oversample everything below the 25th percentil (of classes) (ruos-ends): undersample the top 1/4 of classes
    - randomly under and oversample to the closest ruos (ruos-q)
    - randomly undersample to the closest quartil (rus-q): undersample to 2*median, median, or 1/2*median, whatever is closer

    Oversampling is random, undersampling prioritizes examples with few (1) labels.

    :param input_dataset_dir: Path to a directory with a works.jsonl
    :param output_dataset_dir: Path to a directory where to write the new, resampled works.jsonl
    :param strategy: 'rsm', 'rus-top3', 'rsq', 'rusq'
    :return: None
    """
    work_id, x, y = load_data(input_dataset_dir, preprocess=False)
    class_count, class_order = __class_count(y)
    def _s(method, strategy):
        work_id_resampled, x_resampled, y_resampled = method(work_id, x, y, class_count, class_order)
        (output_dataset_dir / strategy).mkdir(exist_ok=True, parents=False)
        _write_dataset(work_id_resampled, x_resampled, y_resampled, output_dataset_dir / strategy)

    if strategy == 'ruos-m' or strategy == 'all':
        _s(_resample_ruos_m, 'ruos-m')
    if strategy == 'ruos-ends' or strategy == 'all':
        _s(_resample_ruos_ends, 'ruos-ends')
    if strategy == 'ruos-q' or strategy == 'all':
        _s(_resample_ruos_q, 'ruos-q')
    if strategy == 'rus-m' or strategy == 'all':
        _s(_resample_rus_m, 'rus-m')
    if strategy == 'rus-top' or strategy == 'all':
        _s(_resample_rus_top, 'rus-top')
    if strategy == 'rus-q' or strategy == 'all':
        _s(_resample_rus_q, 'rus-q')

    if strategy not in {'ruos-m', 'rus-top', 'ruos-ends', 'rus-m', 'ruos-q', 'rus-q', 'all'}:
        raise AttributeError(f"invalid strategy {strategy}")


@click.command()
@click.option('--works', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Path to the directory with the pan23-trigger-detection-training data (from the PAN23 distribution).')
@click.option('--output', type=click.Path(exists=False, file_okay=False, dir_okay=True),
              help='Path to a directory where to write the new <strategy>/works.jsonl')
@click.option('--strategy', type=str,
              help='The resampling strategy to use (ruos-m, rus-top, ruos-ends, ruos-q, rus-q, or all) ')
def run(works: str, output: str, strategy: str):
    """
    $ python3 resampling.py \
        --works "/home/mike4537/data/pan23-trigger-detection/pan23-trigger-detection-train" \
        --output "/home/mike4537/data/pan23-trigger-detection/samples/" \
        --strategy "all"
    """
    output = Path(output)
    output.mkdir(exist_ok=True, parents=False)
    resample(Path(works), output, strategy)


if __name__ == "__main__":
    run()
