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

logging.basicConfig(filename=f'logs/log-resampling-{dt.now().isoformat()}', encoding='utf-8', level=logging.DEBUG)
# logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

random = default_rng(42)


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


def __get_sample_target(target_sample_count: Union[int | DTypeLike | list], source_class_size, already_sampled):
    """
    Determine the free slots for a class `cls` in the given sampling settings
    :return:
    """
    if isinstance(target_sample_count, list):
        free_slots = np.Infinity
        previous_distance = np.Infinity
        for free_slot_candidate in target_sample_count:
            if abs(free_slot_candidate - source_class_size) < previous_distance:
                previous_distance = abs(free_slot_candidate - source_class_size)
                free_slots = free_slot_candidate
    else:
        free_slots = target_sample_count
    return free_slots - already_sampled


def __sample(target_sample_count: Union[int | List[int]], source_labels: ArrayLike, oversample: bool = True,
             undersample: bool = True) -> List[int]:
    """
    This method does the actual sampling.
    The oversampling only draws examples with 1 label.

    :param target_sample_count: int -> the number of examples which should be sampled
                                List -> a list of possible "steps" to which we can sample.
                                The sampler will check which of these "steps" is closest and sample that many examples.
    :param source_labels: the original label matrix
    :param oversample: True -> Examples will be oversampled to match `target_sample_count`.
                       False -> All examples of the rare classes will be added once
    :param undersample: True -> Examples will be randomly undersampled to match `target_sample_count`
                        False -> All examples of the frequent classes will be added once
    :return: A list of examples as indices from `source_labels`
    """
    logging.info(f"Start sampling with oversampling: {oversample} and undersampling: {undersample}")
    classes = list(range(len(source_labels[0])))
    source_class_sizes = [sum(source_labels[:, c]) for c in classes]
    sampled_classes_count = {x: 0 for x in classes}  # This counts how many labels there are in total
    sampled_class_ids = {x: [] for x in classes}  # this tracks the indices of examples in the argument lists
    all_sampled_ids = []

    for cls in sorted(classes, reverse=True):
        cls_indices = np.asarray(source_labels[:, cls] == 1).nonzero()[0]
        logging.info(f"Sample index {cls} ({len(cls_indices)} examples)")

        # search the closest "step" to the current number of labels
        free_slots = __get_sample_target(target_sample_count, source_class_sizes[cls], sampled_classes_count[cls])
        new_sample = []

        # There are fewer free slots than examples, now we undersample
        if free_slots <= len(cls_indices):
            # for undersampling, we only offer elements that are not already sampled:
            candidates = [idx for idx in cls_indices if idx not in sampled_class_ids[cls]]
            if len(candidates) == free_slots or not undersample:
                logging.info(f"Adding all missing examples for class {cls}")
                new_sample.extend(candidates)

            elif undersample:
                logging.info(f"Undersampling class {cls}, filling {free_slots} slots from {len(candidates)} candidate examples")
                fails = 0
                while free_slots > 0:
                    _ = random.choice(candidates)
                    if sum(source_labels[_]) == 1 or fails > 3000:
                        new_sample.append(_)
                        free_slots -= 1
                    else:
                        fails += 1
            else:
                raise ValueError(f"free_slots is {free_slots} <= {len(cls_indices)} but undersampling is not true. ")

        # oversample: add all examples + oversample new amount by random
        elif free_slots > len(cls_indices):
            # First, we add all missing examples of that class (which have not already been sampled)
            new_sample.extend([idx for idx in cls_indices if idx not in sampled_class_ids[cls]])
            if oversample:
                logging.info(f"Oversampling class {cls}, filling {free_slots} slots from {len(cls_indices)} examples")
                oversample_by = free_slots - len(cls_indices)
                logging.info(f"\tby {oversample_by}")
                _ = random.choice(cls_indices, size=(oversample_by))
                # TODO: only oversample examples without side effects?
                new_sample.extend(_)
            else:
                logging.info(f"Adding nothing to class {cls}. ({free_slots} > {len(cls_indices)}) but oversampling is False")

        # there are already more examples of this class as there should be. Handle this when it happens.
        else:
            raise ValueError(f"free_slots is invalid with {free_slots}")

        all_sampled_ids.extend(new_sample)
        sampled_classes_count, sampled_class_ids = __update_counts(sampled_class_ids, source_labels, all_sampled_ids)

    return [index for example_list in sampled_class_ids.values() for index in example_list]


def _resample_ruos_m(work_id: List[str], x: List[str],
                  y: ArrayLike) -> Tuple[List[str], List[str], ArrayLike]:
    """ randomly stratify to the mean (rsm): stratify everything to the number of example that the
     example at the median index has. This means undersampling for the frequent half of labels,
     and oversampling for the rare half
     """
    median_index = round(len(y[0])/2)
    target_sample_count = sum(y[:, median_index])
    logging.info(f"RSM to median index ({median_index}) with {target_sample_count} examples")

    indices_of_new_data_sample = __sample(target_sample_count, y, oversample=True, undersample=True)

    return ([work_id[idx] for idx in indices_of_new_data_sample],
            [x[idx] for idx in indices_of_new_data_sample],
            np.asarray([y[idx] for idx in indices_of_new_data_sample]))


def _resample_rus_top(work_id: List[str], x: List[str],
                       y: ArrayLike) -> Tuple[List[str], List[str], ArrayLike]:
    """randomly undersample the top 1/4 most frequent labels (rus-top3):
    undersample the top 1/4 by frequency (i.e. cutoff). """
    target_index = round(len(y[0])*(1/4))
    target_sample_count = sum(y[:, target_index])
    logging.info(f"RUS-TOP to index {target_index} with {target_sample_count} examples")

    indices_of_new_data_sample = __sample(target_sample_count, y, oversample=False, undersample=True)

    return ([work_id[idx] for idx in indices_of_new_data_sample],
            [x[idx] for idx in indices_of_new_data_sample],
            np.asarray([y[idx] for idx in indices_of_new_data_sample]))


def _resample_ruos_ends(work_id: List[str], x: List[str],
                       y: ArrayLike) -> Tuple[List[str], List[str], ArrayLike]:
    """ randomly undersample the top 1/4 most frequent labels and
        oversample everything below the 25th percentil (of classes)
    """
    undersample_index = round(len(y[0])*(1/4))
    undersample_target_count = sum(y[:, undersample_index])
    oversample_index = round(len(y[0])*(3/4))
    oversample_target_count = sum(y[:, oversample_index])
    logging.info(f"RUOS-ENDS undersample to index {undersample_index} with {undersample_target_count} examples and "
                 f"oversample to index {oversample_index} with {oversample_target_count} examples.")

    indices_of_new_data_sample = __sample(undersample_target_count, y, oversample=False, undersample=True)
    indices_of_new_data_sample2 = __sample(oversample_target_count,
                                           np.asarray([y[idx] for idx in indices_of_new_data_sample]),
                                           oversample=True,
                                           undersample=False)

    return ([work_id[idx] for idx in indices_of_new_data_sample2],
            [x[idx] for idx in indices_of_new_data_sample2],
            np.asarray([y[idx] for idx in indices_of_new_data_sample2]))


def _resample_ruos_q(work_id: List[str], x: List[str],
                  y: ArrayLike, steps: int = 4) -> Tuple[List[str], List[str], ArrayLike]:
    """  randomly over and undersample to balance the quartils. This means we select the median element of each quartil
    and balance all others to this value"""
    step_halves = len(y[0]) / (steps*2)
    target_indices = [round(i*step_halves) for i in range(1, (steps*2)+1, 2)]
    target_sample_counts = [sum(y[:, idx]) for idx in target_indices]

    logging.info(f"RUOS-Q to the indices {target_indices} with {target_sample_counts} examples")

    indices_of_new_data_sample = __sample(target_sample_counts, y, oversample=True, undersample=True)

    return ([work_id[idx] for idx in indices_of_new_data_sample],
            [x[idx] for idx in indices_of_new_data_sample],
            np.asarray([y[idx] for idx in indices_of_new_data_sample]))


def _resample_rus_q(work_id: List[str], x: List[str],
                   y: ArrayLike, steps: int = 4) -> Tuple[List[str], List[str], ArrayLike]:
    """ randomly undersample to the closest 3-quartil (rusq):
    undersample to 2*median, median, or 1/2*median, whatever is closer """
    step_halves = len(y[0]) / (steps*2)
    target_indices = [round(i*step_halves) for i in range(1, (steps*2)+1, 2)]
    target_sample_counts = [sum(y[:, idx]) for idx in target_indices]

    logging.info(f"RUS-Q to the indices {target_indices} with {target_sample_counts} examples")

    indices_of_new_data_sample = __sample(target_sample_counts, y, oversample=False, undersample=True)

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

    if strategy == 'ruos-m':
        work_id_resampled, x_resampled, y_resampled = _resample_ruos_m(work_id, x, y)
    elif strategy == 'rus-top':
        work_id_resampled, x_resampled, y_resampled = _resample_rus_top(work_id, x, y)
    elif strategy == 'ruos-ends':
        work_id_resampled, x_resampled, y_resampled = _resample_ruos_ends(work_id, x, y)
    elif strategy == 'ruos-q':
        work_id_resampled, x_resampled, y_resampled = _resample_ruos_q(work_id, x, y)
    elif strategy == 'rus-q':
        work_id_resampled, x_resampled, y_resampled = _resample_rus_q(work_id, x, y)
    else:
        raise AttributeError(f"invalid strategy {strategy}")

    _write_dataset(work_id_resampled, x_resampled, y_resampled, output_dataset_dir)


@click.command()
@click.option('--works', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Path to the directory with the pan23-trigger-detection-training data (from the PAN23 distribution).')
@click.option('--output', type=click.Path(exists=False, file_okay=False, dir_okay=True),
              help='Path to a directory where to write the new works.jsonl')
@click.option('--strategy', type=str,
              help='The resampling strategy to use (ruos-m, rus-top, ruos-ends, ruos-q, rus-q) ')
def run(works: str, output: str, strategy: str):
    """
    $ python3 resampling.py \
        --works "/home/mike4537/data/pan23-trigger-detection/pan23-trigger-detection-train" \
        --output "/home/mike4537/data/pan23-trigger-detection/samples/ruos-m" \
        --strategy "ruos-m"
    """
    output = Path(output)
    output.mkdir(exist_ok=True, parents=False)
    resample(Path(works), output, strategy)


if __name__ == "__main__":
    run()
