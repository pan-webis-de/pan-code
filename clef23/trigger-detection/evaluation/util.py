"""
Utility functions for the shared task on Trigger Detection at PAN@CLEF2023.

Contact: matti.wiegmann@uni-weimar.de
         or create an issue/PR on Github: https://github.com/pan-webis-de/pan-code
"""
from typing import List, Iterable
import numpy as np

# These are the labels used in the PAN23 Trigger Detection dataset, in order of frequency
LABELS = ["pornographic-content", "violence", "death", "sexual-assault", "abuse", "blood", "suicide",
          "pregnancy", "child-abuse", "incest", "underage", "homophobia", "self-harm", "dying", "kidnapping",
          "mental-illness", "dissection", "eating-disorders", "abduction", "body-hatred", "childbirth",
          "racism", "sexism", "miscarriages", "transphobia", "abortion", "fat-phobia", "animal-death",
          "ableism", "classism", "misogyny", "animal-cruelty"]  # 32


def to_array_representation(labels: List[str]) -> Iterable[int]:
    """ convert a string representation of the labels (used in the labels.jsonl)
        into the array representation (used in the works.jsonl).

        The array representation is natively understood by huggingface and scikit-learn.
     """
    return [1 if label in labels else 0 for label in LABELS]


def to_string_representation(label_vector: Iterable[int]) -> Iterable[str]:
    """ convert an array representation of the labels (used in the works.jsonl)
        into the string representation (used in the labels.jsonl).

        The string representation is used in the output format (which is the input to the evaluator).
        Use this function to convert a huggingface/sci-kit output to the output format of the task.
     """
    return [LABELS[idx] for idx, cls in enumerate(label_vector) if cls == 1]


if __name__ == "__main__":
    assert (to_array_representation(LABELS) == [1] * len(LABELS))
    assert (to_array_representation([]) == [0] * len(LABELS))
    assert (to_array_representation(["death"]) == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert (np.array_equal(to_array_representation(["death"]),
                           np.asarray([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])))

    assert (to_string_representation([1] * len(LABELS)) == LABELS)
    assert (to_string_representation([0] * len(LABELS)) == [])
    assert (to_string_representation([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) == ["death"])
    assert (to_string_representation(np.asarray([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) == ["death"])
