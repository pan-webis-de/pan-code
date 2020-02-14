import os
from glob import glob
import argparse
import json
import warnings
from enum import Enum

"""
script checks solution files for Style Change Detection Task for PAN@CLEF20 
"""


class ParseError(Enum):
    """ ENUM for possible errors when parsing solution json """
    INVALID_JSON = 'JSON not valid/parseable.'
    TASK1_INVALID_FORMAT = 'Task 1 solution has to be 0 or 1.'
    TASK2_INVALID_FORMAT = 'Task 2 solution has to be array of 0s and 1s.'
    TASK2_INVALID_LENGTH = 'Task 2 solution array length does not match number of paragraph pairs.'


def get_solution_file_check_result(solution_file, problem_id, input_folder):
    """
    Checks solution file (json) for correct format
    :param solution_file: path to file to be checked
    :param problem_id: problem id for which solution is checked
    :param input_folder: path of folder holding input txt files
    :return: List of errors, empty list if no error occurred.
    """

    occurred_errors = []
    try:
        with open(solution_file, 'r') as fh:
            solution = json.load(fh)

    except Exception as _:
        # json not valid, return as non-correct right away
        occurred_errors.append(ParseError.INVALID_JSON)
        return occurred_errors

    # solutions for task 1
    if 'multi-author' not in solution:
        warnings.warn(f"[problem {problem_id}]: Task 1 solution missing (WARNING).", SyntaxWarning)
    else:
        # solution for task 1 has to be 0 or 1.
        if (not isinstance(solution['multi-author'], int)) or (not solution['multi-author'] in [0, 1]):
            occurred_errors.append(ParseError.TASK1_INVALID_FORMAT)


    # solutions for task 2
    if 'changes' not in solution:
        warnings.warn(f"[problem {problem_id}]: Task 2 solution missing (WARNING).", SyntaxWarning)
    else:
        # solution for task 2 has to be list
        if (not isinstance(solution['changes'], list)):
            occurred_errors.append(ParseError.TASK2_INVALID_FORMAT)
        else:
            # solution for task 2 list may only contain 0s or 1s
            if any((x < 0 or x > 1)  for x in solution['changes']):
                occurred_errors.append(ParseError.TASK2_INVALID_FORMAT)
            else:
                # check for correct size of array for given problem
                if len(solution['changes']) != (get_paragraph_count(problem_id, input_folder) - 1):
                    occurred_errors.append(ParseError.TASK2_INVALID_LENGTH)
    return occurred_errors


def get_paragraph_count(problem_id, input_folder):
    """
    Counts paragraphs in given input txt-file
    :param problem_id: problem id for which paragraphs are counted
    :param input_folder: path to folder holding input files (input texts, .txt)
    """
    with open(os.path.join(input_folder, f'problem-{problem_id}.txt')) as txt_file:
        return txt_file.read().count("\n\n") + 1


def check_output_files(problem_ids, output_path, input_folder):
    """
    loops over all problem ids and checks solution file format and content
    :param problem_ids: list of all problem ids for which output is to be checked
    :param output_path: path of folder holding solution (output) files
    :param input_folder: path of folder holding input (txt) files
    """

    # for all problem-ids, check whether solution exists and ist correctly formatted
    for problem_id in problem_ids:
        file_path = os.path.join(output_path, 'solution-problem-' + problem_id + '.json')
        if os.path.exists(file_path):
           errors = get_solution_file_check_result(file_path, problem_id, input_folder)
           if not errors:
                print(f"[problem {problem_id}]: OK")
           else:
                [print(f"[problem {problem_id}]: {x}") for x in errors]
        else:
            print(f"[problem {problem_id}]: no solution file found")


def get_problem_ids(input_folder):
    """
    gathers all problem-ids of input files as list
    :param input_folder: folder holding input files (txt)
    :return: sorted list of problem-ids
    """
    problem_ids = []
    for file in glob(os.path.join(input_folder, '*.txt')):
        problem_ids.append(os.path.basename(file)[8:-4])
    return sorted(problem_ids)


def main():
    parser = argparse.ArgumentParser(description='PAN20 Style Change Detection Task: Output Verifier')
    parser.add_argument('--output', type=str, help='folder containing output/solution files (json)', required=True)
    parser.add_argument('--input', type=str, help='folder containing input files for task (txt)', required=True)
    args = parser.parse_args()

    problem_ids = get_problem_ids(args.input)
    check_output_files(problem_ids, args.output, args.input)


if __name__ == "__main__":
    main()
