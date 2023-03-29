import os
from glob import glob
import argparse
import json
import warnings
from enum import Enum

"""
script checks solution files for Style Change Detection Task for PAN@CLEF23
"""


class ParseError(Enum):
    """ENUM for possible errors when parsing solution json"""

    INVALID_JSON = "JSON not valid/parseable."
    INVALID_FORMAT = "Task solution has to be array of 0s and 1s."
    INVALID_LENGTH = "Task solution array length does not match number of paragraph pairs."
    MISSING_KEY = "Task solution key not contained in JSON."


class Task(Enum):
    """ENUM for task types"""

    TASK1 = "Dataset 1 (easy)"
    TASK2 = "Dataset 2 (medium)"
    TASK3 = "Dataset 3 (hard)"


def get_solution_file_check_result(solution_file: str, problem_id: str, input_folder: str, task: Task) -> object:
    """
    Checks solution file (json) for correct format
    :param solution_file: path to file to be checked
    :param problem_id: problem id for which solution is checked
    :param input_folder: path of folder holding input txt files
    :param task: task (enum) to be validated
    :return: List of errors, empty list if no error occurred.
    """

    occurred_errors = []
    try:
        with open(solution_file, "r") as fh:
            solution = json.load(fh)

    except Exception as _:
        # json not valid, return as non-correct right away
        occurred_errors.append(ParseError.INVALID_JSON)
        return occurred_errors

    if "changes" not in solution:
        occurred_errors.append(ParseError.MISSING_KEY)
    else:
        # solution has to be list
        if not isinstance(solution["changes"], list):
            occurred_errors.append(ParseError.INVALID_FORMAT)
        else:
            # list may only contain 0s or 1s
            if any((x < 0 or x > 1) for x in solution["changes"]):
                occurred_errors.append(ParseError.INVALID_FORMAT)
            else:
                # check for correct size of array for given problem
                if len(solution["changes"]) != (
                    get_chunk_count(problem_id, input_folder) - 1
                ):
                    occurred_errors.append(ParseError.INVALID_LENGTH)

    return occurred_errors


def get_chunk_count(problem_id: str, input_folder: str) -> int:
    """
    Counts input chunks (paragraphs) in given input txt-file
    :param problem_id: problem id for which paragraphs are counted
    :param input_folder: path to folder holding input files (input texts, .txt)
    """
    with open(os.path.join(input_folder, f"problem-{problem_id}.txt"), 'r', newline="") as txt_file:
        return txt_file.read().count("\n") + 1


def check_output_files(problem_ids: list, output_path: str, input_folder: str, task: Task):
    """
    loops over all problem ids and checks solution file format and content
    :param problem_ids: list of all problem ids for which output is to be checked
    :param output_path: path of folder holding solution (output) files
    :param input_folder: path of folder holding input (txt) files
    :param task: task (enum) to be evaluated
    """

    # for all problem-ids, check whether solution exists and is correctly formatted
    for problem_id in problem_ids:
        file_path = os.path.join(
            output_path, "solution-problem-" + problem_id + ".json"
        )
        if os.path.exists(file_path):
            errors = get_solution_file_check_result(
                file_path, problem_id, input_folder, task
            )

            if not errors:
                print(f"[problem {problem_id}, {task.value}]: OK")
            else:
                [print(f"[problem {problem_id}: {task.value}]: {x.value}")
                 for x in errors]
        else:
            print(
                f"[problem {problem_id}, {task.value}]: no solution file found: {file_path}")


def get_problem_ids(input_folder: str) -> list:
    """
    gathers all problem-ids of input files as list
    :param input_folder: folder holding input files (txt)
    :return: sorted list of problem-ids
    """
    problem_ids = []
    for file in glob(os.path.join(input_folder, "*.txt")):
        problem_ids.append(os.path.basename(file)[8:-4])
    print(f"Read {len(problem_ids)} problem ids from {input_folder}.")
    return sorted(problem_ids)


def main():
    parser = argparse.ArgumentParser(
        description="PAN23 Style Change Detection Task: Output Verifier"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="folder containing output/solution files (json)",
        required=True,
    )
    parser.add_argument(
        "--input",
        type=str,
        help="folder containing input files for task (txt)",
        required=True,
    )
    args = parser.parse_args()

    # task 1
    problem_ids = get_problem_ids(os.path.join(args.input, "dataset1"))
    check_output_files(problem_ids, os.path.join(
        args.output, "dataset1"), os.path.join(args.input, "dataset1"), Task.TASK1)

    # task 2
    problem_ids = get_problem_ids(os.path.join(args.input, "dataset2"))
    check_output_files(problem_ids, os.path.join(
        args.output, "dataset2"), os.path.join(args.input, "dataset2"), Task.TASK2)

    # task 3
    problem_ids = get_problem_ids(os.path.join(args.input, "dataset3"))
    check_output_files(problem_ids, os.path.join(
        args.output, "dataset3"), os.path.join(args.input, "dataset3"), Task.TASK3)


if __name__ == "__main__":
    main()
