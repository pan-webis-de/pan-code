import argparse
import glob
import json
import os
from itertools import chain
from sklearn.metrics import f1_score

EV_OUT = "evaluation.txt"

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
    for truth_file in glob.glob(os.path.join(truth_folder, 'truth-problem*.json')):
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
            print("No solution file found for problem %s, exiting." % problem_id)
            exit(0)
    return all_truth, all_solutions

def compute_score_single_predictions(truth, solutions, key):
    """ compute f1 score for single prediction per document
    :param truth: list of ground truth values for all problem-ids
    :param solutions: list of solutions for all problem-ids
    :param key: key of solutions to compute score for (=task)
    :return: f1 score
    """
    truth, solution = extract_task_results(truth, solutions, key)
    return f1_score(truth, solution, average='macro')

def compute_score_multiple_predictions(truth, solutions, key, labels):
    """ compute f1 score for task 2 and task 3 - list of predictions
    :param truth: list of ground truth values for all problem-ids
    :param solutions: list of solutions for all problem-ids
    :param key: key of solutions to compute score for (=task)
    :return: f1 score
    """
    task2_truth, task2_solution = extract_task_results(truth, solutions, key)
    # task 2 - lists have to be flattened first
    return f1_score(list(chain.from_iterable(task2_truth)), list(chain.from_iterable(task2_solution)), average='macro', labels=labels)

def write_output(filename, k, v):
    """
    print() and write a given measurement to the indicated output file
    :param filename: full path of the file, where to write to
    :param k: the name of the metric
    :param v: the value of the metric
    :return: None
    """
    line = 'measure{{\n  key: "{}"\n  value: "{}"\n}}\n'.format(k, str(v))
    print(line)
    open(filename, "a").write(line)


def main():
    parser = argparse.ArgumentParser(description='PAN21 Style Change Detection Task: Evaluator')
    parser.add_argument("-p", "--predictions", help="path to the dir holding the predictions", required=True)
    parser.add_argument("-t", "--truth", help="path to the dir holding the true labels", required=True)
    parser.add_argument("-o", "--output", help="path to the dir to write the results to", required=True)
    args = parser.parse_args()

    solutions= read_solution_files(args.predictions)
    truth = read_ground_truth_files(args.truth)

    task1_result = compute_score_single_predictions(truth, solutions, 'multi-author')
    task2_results = compute_score_multiple_predictions(truth, solutions, 'changes', labels=[0, 1])
    task3_results = compute_score_multiple_predictions(truth, solutions, 'paragraph-authors', labels=[1,2,3,4])

    for k, v in {
        "task1_score": task1_result,
        "task2_score": task2_results,
        "task3_score": task3_results}.items():
        write_output(os.path.join(args.output, EV_OUT), k, v),


if __name__ == "__main__":
    main()
