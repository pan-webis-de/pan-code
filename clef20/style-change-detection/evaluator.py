import argparse
import glob
import json
import os
from itertools import chain

from sklearn.metrics import f1_score


def read_solution_files(solutions_folder):
    """
    reads all solution files into dict
    :param solutions_folder: path to folder holding the solution files
    :return: dict of solution files with problem-id as key and file content as value
    """
    solutions = {}
    for solution_file in glob.glob(os.path.join(solutions_folder, 'solution-problem-*.json')):
        with open(solution_file, 'r') as fh:
            curr_solution = json.load(fh)
            solutions[os.path.basename(solution_file)[9:-5]] = curr_solution
    return solutions

def read_ground_truth_files(truth_folder):
    """
    reads ground truth files into dict
    :param truth_folder: path to folder holding ground truth files
    :return: dict of ground truth files with problem-id as key and file content as value
    """
    truth = {}
    for truth_file in glob.glob(os.path.join(truth_folder, 'truth-*.json')):
        with open(truth_file, 'r') as fh:
            curr_truth = json.load(fh)
            truth[os.path.basename(truth_file)[6:-5]] = curr_truth
    return truth

def extract_task_results(truth, solutions, task):
    """
    extracts truth and solution values for a given task
    :param truth: dict of all ground truth values with problem-id as key
    :param solutions: dict of all solution values with problem-id as key
    :param task: task for which values are extracted (string, e.g., 'multi-author' or 'changes')
    :return: list of all ground truth values, list of all solution values for given task
    """
    all_solutions = []
    all_truth = []
    for problem_id, truth_instance in truth.items():
        all_truth.append(truth_instance[task])
        try:
            all_solutions.append(solutions[problem_id][task])
        except KeyError as _:
            print(f"No solution file found for problem {problem_id}, exiting.")
            exit(0)
    return all_truth, all_solutions

def compute_task1_f1_score(truth, solutions):
    """ compute f1 score for task 1
    :param truth: list of ground truth values for all problem-ids
    :param solutions: list of solutions for all problem-ids
    :return: f1 score
    """
    task1_truth, task1_solution = extract_task_results(truth, solutions, 'multi-author')
    return f1_score(task1_truth, task1_solution, average='micro')

def compute_task2_f1_score(truth, solutions):
    """ compute f1 score for task 2
    :param truth: list of ground truth values for all problem-ids
    :param solutions: list of solutions for all problem-ids
    :return: f1 score
    """
    task2_truth, task2_solution = extract_task_results(truth, solutions, 'changes')
    # task 2 - lists have to be flattened first
    return f1_score(list(chain.from_iterable(task2_truth)), list(chain.from_iterable(task2_solution)), average='micro', labels=[0, 1])


def main():
    parser = argparse.ArgumentParser(description='PAN20 Style Change Detection Task: Output Verifier')
    parser.add_argument('--solution', type=str, help='folder containing solution files (json)', required=True)
    parser.add_argument('--truth', type=str, help='folder containing ground truth files (json)', required=True)
    args = parser.parse_args()

    solutions = read_solution_files(args.solution)
    truth = read_ground_truth_files(args.truth)

    task1_score = compute_task1_f1_score(truth, solutions)
    print(f"score task 1: {task1_score:.5f}")

    task2_score = compute_task2_f1_score(truth, solutions)
    print(f"score task 2: {task2_score:.5f}")


if __name__ == "__main__":
    main()
