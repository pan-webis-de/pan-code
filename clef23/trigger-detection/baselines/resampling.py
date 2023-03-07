"""
Utilities for the shared task on Trigger Detection at PAN23.

This script contains code to resample the dataset (as it is on zenodo) to modify the class balance.
"""
import json
import logging
from typing import Tuple, Union, List
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
from numpy.typing import ArrayLike
import click

from util import load_data

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

random.seed(42)


def __update_counts(classes_sampled, indices_sampled, labels):
    """
    update the given dict of label counts
    :param classes_sampled: a dict with label_index: count
    :param indices_sampled: a list of indices of examples (
    :param labels:
    :return:
    """
    print(f"increase count from {classes_sampled}")
    for example_index in tqdm(indices_sampled, desc="counting new classes"):
        new_labels = np.where(labels[example_index] == 1)
        for nc in new_labels:
            classes_sampled[nc] += 1
    print(f"to {classes_sampled}")


def __get_sample_target(target_sample_count, source_class_size, already_sampled):
    """
    Determine the free slots for a class `cls` in the given sampling settings
    :return:
    """
    if not isinstance(target_sample_count, int):
        free_slots = np.Infinity
        previous_distance = np.Infinity
        for free_slot_candidate in target_sample_count:
            if abs(free_slot_candidate - source_class_size) < previous_distance:
                previous_distance = abs(free_slot_candidate - source_class_size)
                free_slots = free_slot_candidate
    else:
        free_slots = target_sample_count
    return free_slots - already_sampled


def __sample(target_sample_count: Union[int | List[int]], source_labels: ArrayLike[int], oversample: bool = True,
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
    classes = list(range(len(source_labels[0])))

    source_class_sizes = [sum(source_labels[:, c]) for c in classes]
    classes_sampled = {x: 0 for x in classes}  # This counts how many labels there are in total
    sampled_ids = {x: [] for x in classes}  # this tracks the indices of examples in the argument lists

    for cls in sorted(classes, reverse=True):
        cls_indices = np.where(source_labels == 1)[cls]
        logging.info(f"sample index {cls} with {len(cls_indices)} examples")

        # search the closest "step" to the current number of labels
        free_slots = __get_sample_target(target_sample_count, source_class_sizes[cls], classes_sampled[cls])

        # There are fewer free slots than examples, now we undersample
        if free_slots < len(cls_indices):
            new_sample = []
            if undersample:
                logging.info(f"undersampling examples for class {cls} to ")
                while free_slots > 0:
                    _ = random.choice(cls_indices)
                    if sum(source_labels[_]) == 1:
                        new_sample.append(_)
                        free_slots -= 1
            else:
                logging.info(f"adding all examples for class {cls}")
                new_sample.extend(cls_indices)

        # oversample: add all examples + oversample new amount by random
        elif free_slots >= len(cls_indices):
            new_sample = []
            new_sample.extend(cls_indices)
            if oversample:
                logging.info(f"oversampling examples for class {cls}")
                oversample_by = free_slots - len(cls_indices)
                _ = random.sample(cls_indices, oversample_by)
                new_sample.extend(_)

        # there are already more examples of this class as there should be. Handle this when it happens.
        else:
            raise ValueError(f"free_slots is invalid with {free_slots}")

        sampled_ids[cls].extend(new_sample)
        __update_counts(classes_sampled, sampled_ids[cls], source_labels)

    return [index for example_list in sampled_ids.values() for index in example_list]


def _resample_ruos_m(work_id: List[str], x: List[str],
                  y: ArrayLike[int]) -> Tuple[List[str], List[str], ArrayLike[int]]:
    """ randomly stratify to the mean (rsm): stratify everything to the number of example that the
     example at the median index has. This means undersampling for the frequent half of labels,
     and oversampling for the rare half
     """
    median_index = round(len(y[0])/2)
    target_sample_count = sum(y[:, median_index])
    logging.info(f"RSM to index {median_index} with {target_sample_count} examples")

    indices_of_new_data_sample = __sample(target_sample_count, y, oversample=True, undersample=True)

    return ([work_id[idx] for idx in indices_of_new_data_sample],
            [x[idx] for idx in indices_of_new_data_sample],
            np.asarray([y[idx] for idx in indices_of_new_data_sample]))


def _resample_rus_top(work_id: List[str], x: List[str],
                       y: ArrayLike[int]) -> Tuple[List[str], List[str], ArrayLike[int]]:
    """randomly undersample the top 1/4 most frequent labels (rus-top3):
    undersample the top 1/4 by frequency (i.e. cutoff). """
    target_index = round(len(y[0])*(3/4))
    target_sample_count = sum(y[:, target_index])
    logging.info(f"RUS-Q to index {target_index} with {target_sample_count} examples")

    indices_of_new_data_sample = __sample(target_sample_count, y, oversample=False, undersample=True)

    return ([work_id[idx] for idx in indices_of_new_data_sample],
            [x[idx] for idx in indices_of_new_data_sample],
            np.asarray([y[idx] for idx in indices_of_new_data_sample]))


def _resample_ruos_ends(work_id: List[str], x: List[str],
                       y: ArrayLike[int]) -> Tuple[List[str], List[str], ArrayLike[int]]:
    """ randomly undersample the top 1/4 most frequent labels and
        oversample everything below the 25th percentil (of classes)
    """
    undersample_index = round(len(y[0])*(3/4))
    undersample_target_count = sum(y[:, undersample_index])
    oversample_index = round(len(y[0])*(1/4))
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
                  y: ArrayLike[int], steps: int = 4) -> Tuple[List[str], List[str], ArrayLike[int]]:
    """  randomly over and undersample to balance the quartils. This means we select the median element of each quartil
    and balance all others to this value"""
    step_halves = len(y[0]) / steps*2
    target_indices = [round(i*step_halves) for i in range(1, (steps*2)+1, 2)]
    target_sample_counts = [sum(y[:, idx]) for idx in target_indices]

    logging.info(f"RUOS-Q to the indices {target_indices} with {target_sample_counts} examples")

    indices_of_new_data_sample = __sample(target_sample_counts, y, oversample=True, undersample=True)

    return ([work_id[idx] for idx in indices_of_new_data_sample],
            [x[idx] for idx in indices_of_new_data_sample],
            np.asarray([y[idx] for idx in indices_of_new_data_sample]))


def _resample_rus_q(work_id: List[str], x: List[str],
                   y: ArrayLike[int], steps: int = 4) -> Tuple[List[str], List[str], ArrayLike[int]]:
    """ randomly undersample to the closest 3-quartil (rusq):
    undersample to 2*median, median, or 1/2*median, whatever is closer """
    step_halves = len(y[0]) / steps*2
    target_indices = [round(i*step_halves) for i in range(1, (steps*2)+1, 2)]
    target_sample_counts = [sum(y[:, idx]) for idx in target_indices]

    logging.info(f"RUS-Q to the indices {target_indices} with {target_sample_counts} examples")

    indices_of_new_data_sample = __sample(target_sample_counts, y, oversample=False, undersample=True)

    return ([work_id[idx] for idx in indices_of_new_data_sample],
            [x[idx] for idx in indices_of_new_data_sample],
            np.asarray([y[idx] for idx in indices_of_new_data_sample]))


def _write_dataset(work_id: List[str], x: List[str], y: ArrayLike[int],
                   output_dataset_dir: Path) -> None:
    """ write the dataset (in the parameters) in a works.jsonl into the output directory """
    with open(output_dataset_dir / "works.jsonl", 'w') as of:
        for w, text, labels in zip(work_id, x, y):
            data = {"work_id": w, "text": text, "labels": labels}
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
@click.option('--strategy', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='The resampling strategy to use (ruos-m, rus-top, ruos-ends, ruos-q, rus-q) ')
def run(works: str, output: str, strategy: str):
    output = Path(output)
    if not output.exists():
        output.mkdir()
    resample(Path(works), output, strategy)


if __name__ == "__main__":
    run()
