import argparse
import glob
import json
import os
from itertools import chain
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment
import numpy as np
from scipy.sparse import coo_matrix

EV_OUT = "evaluation.prototext"

def read_solution_files(solutions_folder: str) -> dict:
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

def read_ground_truth_files(truth_folder: str) -> dict:
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

def extract_task_results(truth: dict, solutions: dict, task: str) -> tuple:
    """
    extracts truth and solution values for a given task
    :param truth: dict of all ground truth values with problem-id as key
    :param solutions: dict of all solution values with problem-id as key
    :param task: task for which values are extracted (string, e.g., 'multi-author' or 'changes')
    :return: list of all ground truth values, list of all solution values for given task
    """
    all_solutions = []
    all_truth = []
    for problem_id, truth_instance in sorted(truth.items()):
        if len(truth_instance[task]) != len(solutions[problem_id][task]):
          print(f"Solution length for problem {problem_id} is not correct, skipping.")
          continue
        all_truth.append(truth_instance[task])
        all_solutions.append(solutions[problem_id][task])
    return all_truth, all_solutions


def compute_score_multiple_predictions(truth_values: dict, solution_values: dict, key: str, labels: list) -> float:
    """ compute f1 score for list of predictions
    :param labels: labels used for the predictions
    :param truth_values: list of ground truth values for all problem-ids
    :param solution_values: list of solutions for all problem-ids
    :param key: key of solutions to compute score for (=task)
    :return: f1 score
    """
    # extract truth and solution values in suitable format
    truth, solution = extract_task_results(truth_values, solution_values, key)
    # lists have to be flattened first
    return f1_score(list(chain.from_iterable(truth)), list(chain.from_iterable(solution)), average='macro',
labels=labels, zero_division=0)


def sort_labels(labels: list) -> np.array:
    """
    Code by Benedikt Bönninghoff <benedikt.boenninghoff@rub.de>
    Helper function to sort a label vector. For instance:
        labels = [1, 2, 2, 5, 10]
        lanels_new = [0, 1, 1, 2, 3]

    :param labels: unsorted labels
    :return: sorted labels between 0 and maximum number of authors
    """
    labels_sorted = np.zeros_like(labels)
    mapper = {}
    for l in np.unique(labels):
        if l not in mapper:
            mapper[l] = len(mapper)
        idx = np.where(labels == l)
        labels_sorted[idx] = mapper[l]
    return np.array(labels_sorted)


def contingency_matrix(ref_labels: np.array, sys_labels: np.array) -> np.array:
    """
    Code by Benedikt Bönninghoff <benedikt.boenninghoff@rub.de>
    Computes the contingency matrix between reference labels and system labels. Script is taken from:
    https://github.com/nryant/dscore

    :param ref_labels: array of size (n_samples,)
    :param sys_labels: ndarray of size (n_samples,)
    :return: ndarray, (n_ref_classes, n_sys_classes), whose (i,j)-th entry is the number of times the i-th reference
             label and j-th system label co-occur
    """
    if ref_labels.ndim != sys_labels.ndim:
        raise ValueError('reference and system labels must be 1D arrays')
    if ref_labels.shape[0] != sys_labels.shape[0]:
        raise ValueError('reference and system labels must have same size')

    ref_classes, ref_class_inds = np.unique(ref_labels, return_inverse=True)
    sys_classes, sys_class_inds = np.unique(sys_labels, return_inverse=True)
    n_frames = ref_labels.size
    cm = coo_matrix(
        (np.ones(n_frames), (ref_class_inds, sys_class_inds)),
        shape=(ref_classes.size, sys_classes.size),
        dtype=int)
    cm = cm.toarray()

    return cm



def compute_der(ref: np.array, cm: np.array, ref_idx: np.array, sys_idx: np.array, reverse: bool = False) -> float:
    """
    Code by Benedikt Bönninghoff <benedikt.boenninghoff@rub.de>
    Computes the diarization error rate:
        DER = (reference length - optimal matches) / reference length

    :param ref: reference labels
    :param cm: contigency matrix
    :param ref_idx: ref indices of optimal assignment
    :param sys_idx: corresponding sys indices giving the optimal assignment
    :param reverse: return diarization accuracy rate "optimal matches / reference length"
    :return: diarization error between 0 and 1

    """
    ref_total_length = ref.shape[0]
    optimal_match_overlap = cm[ref_idx, sys_idx].sum()
    if reverse:
        der = optimal_match_overlap / ref_total_length
    else:
        der = (ref_total_length - optimal_match_overlap) / ref_total_length

    return der


def compute_jer(ref: np.array, sys: np.array, cm: np.array, ref_idx: np.array, sys_idx: np.array, reverse: bool = False) -> np.array:
    """
    Code by Benedikt Bönninghoff <benedikt.boenninghoff@rub.de>

    Assume we have N reference authors and M system authors. An optimal mapping between authors is determined using
    the Hungarian algorithm so that each reference author is paired with at most one system author.
    Then, for each reference author (ref_a) the author-specific Jaccard error rate is computed w.r.t. the
    corresponding system author (sys_a). More precisely:

        JER_ref_a = 1.0 - intersection(ref_a, sys_a) / union(ref_a, sys_a)

    If the reference author was not paired with a system author, JER_ref_a = 1.0. The Jaccard error rate then is the
    average of the author specific Jaccard error rates.

    :param ref: reference labels
    :param sys: system labels
    :param cm: contigency matrix
    :param ref_idx: ref indices of optimal assignment
    :param sys_idx: corresponding sys indices giving the optimal assignment
    :param reverse: return Jaccard similarity "intersection(ref_a, sys_a) / union(ref_a, sys_a)"
    :return: jaccard error rate between 0 and 1
    """

    # iterate over all reference authors
    jer_list = []
    for a_ref in range(np.max(ref) + 1):

        # case: no ref author for sys author
        if a_ref >= ref_idx.shape[0]:
            if reverse:
                jer_a = 0.0
            else:
                jer_a = 1.0
        else:
            # get corresponding index for sys
            a_sys = sys_idx[a_ref]

            # compute intersection
            intersection = cm[a_ref, a_sys]
            # count occurrence of a in ref
            len_ref = np.sum(ref == a_ref)
            # count occurrence of a in sys
            len_sys = np.sum(sys == a_sys)
            # compute union
            union = len_ref + len_sys - intersection
            # compute jer for ref author
            if reverse:
                jer_a = intersection / union
            else:
                jer_a = 1.0 - (intersection / union)
        jer_list.append(jer_a)

    jer = np.mean(jer_list)

    return jer


def compute_secondary_metrics(ref_list: dict, sys_list: dict, task_key: str, reverse: bool = False) -> tuple:
    """
    Code by Benedikt Bönninghoff <benedikt.boenninghoff@rub.de>
    Computes secondary metrics (jer and der)
    :param task_key: key used in solution file to reference current task
    :param ref_list: list of arrays for reference labels
    :param sys_list: list of arrays for system labels
    :param reverse: compute "diarization accuracy rate" and "Jaccard similarity"
    :return: tuple with metrics
    """

    ref_list, sys_list = extract_task_results(ref_list, sys_list, task_key)

    jer_list, der_list = [], []

    for i in range(len(ref_list)):

        # step 1: sort labels
        ref = sort_labels(ref_list[i])
        sys = sort_labels(sys_list[i])

        # step 2: build contingency matrix between ref and sys
        cm = contingency_matrix(ref, sys)

        # step 3: apply Hungarian algorithm to find optimal match between ref and sys
        ref_idx, sys_idx = linear_sum_assignment(-cm)

        # step 4: compute metrics
        # compute JER
        jer = compute_jer(ref, sys, cm, ref_idx, sys_idx, reverse=reverse)
        jer_list.append(np.round(jer, 4))

        # compute DER
        der = compute_der(ref, cm, ref_idx, sys_idx, reverse=reverse)
        der_list.append(np.round(der, 4))


    return np.mean(jer_list), np.mean(der_list)


def write_output(filename: str, k: str, v: str):
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
    parser.add_argument("-p", "--predictions", help="path to the dir holding the predictions (in a folder for each dataset/task)", required=True)
    parser.add_argument("-t", "--truth", help="path to the dir holding the true labels (in a folder for each dataset/task)", required=True)
    parser.add_argument("-o", "--output", help="path to the dir to write the results to", required=True)
    args = parser.parse_args()

    task1_solutions = read_solution_files(os.path.join(args.predictions, 'dataset1'))
    task1_truth = read_ground_truth_files(os.path.join(args.truth, 'dataset1/test'))

    try:
       task1_f1 = compute_score_multiple_predictions(task1_truth, task1_solutions, 'changes', labels=[0, 1])
    except KeyError as _:
       task1_f1 = None
       print("No solution file found for one or more problems, please check the output. Exiting task 1.")


    task2_solutions = read_solution_files(os.path.join(args.predictions, 'dataset2'))
    task2_truth = read_ground_truth_files(os.path.join(args.truth, 'dataset2/test'))
    try:
       task2_f1 = compute_score_multiple_predictions(task2_truth, task2_solutions, 'paragraph-authors', labels=[1,2,3,4,5])
    except KeyError as _:
       task2_f1 = None
       print("No solution file found for one or more problems, please check the output. Exiting task 2.")
       
    try:
       task2_jer, task2_der = compute_secondary_metrics(ref_list=task2_truth, sys_list=task2_solutions, task_key='paragraph-authors', reverse=True)
    except KeyError as _:
       task2_jer = None
       task2_der = None
       print("No solution file found for one or more problems, please check the output. Exiting task 2.")

    task3_solutions = read_solution_files(os.path.join(args.predictions, 'dataset3'))
    task3_truth = read_ground_truth_files(os.path.join(args.truth, 'dataset3/test'))
    try:
       task3_f1 = compute_score_multiple_predictions(task3_truth, task3_solutions, 'changes', labels=[0, 1])
    except KeyError as _:
       task3_f1 = None
       print("No solution file found for one or more problems, please check the output. Exiting task 3.")

    for k, v in {
        "task1_f1_score": task1_f1,
        "task2_f1_score": task2_f1,
        "task3_f1_score": task3_f1,
        "task2_der": task2_der,
        "task2_jer": task2_jer}.items():
        write_output(os.path.join(args.output, EV_OUT), k, v),


if __name__ == "__main__":
    main()
