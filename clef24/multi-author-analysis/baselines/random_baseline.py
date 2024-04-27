#!/usr/bin/python3
import argparse
import glob
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for PAN24 Multi-Author Writing Style Analysis task.')

    parser.add_argument('--input', type=str, help='The input data for three sub tasks (expected in txt format in `easy/medium/hard` subfolders).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in json files in `easy/medium/hard` subfolders.', required=False)

    return parser.parse_args()


def read_problem_files(problems_folder: str) -> dict:
    """
    reads all problem files into dict
    :param problems_folder: path to folder holding the solution files in three subfolders `easy/medium/hard`
    :return: dict of problem files with problem-id as key and file content as value
    """
    problems = {}
    for solution_file in glob.glob(os.path.join(problems_folder, '**/problem-*.txt')):
        with open(solution_file, 'r') as fh:
            problem = fh.read()
            problems[os.path.basename(solution_file)[:-4]] = problem
    return problems


def run_baseline(problems: dict, output_path: str):
    """
    Write random predictions to solution files in the format:
    {
    "authors": 2,
    "changes": [1, 1, 0, 0, 1]
    }

    :param problems: dictionary of problem files with ids as keys
    :param output_path: output folder to write solution files
    """
    os.makedirs(output_path, exist_ok=True)

    for id, problem in problems.items():
        with open(f"{output_path}/solution-{id}.json", 'w') as out:
            authors = random.randint(1, 5)
            prediction = {'authors': authors, 'changes': [random.choice([0, 1]) for _ in range(authors)]}
            out.write(json.dumps(prediction))


if __name__ == '__main__':
    """
    Input path folder holding the solution files in three subfolders `easy/medium/hard`, thus we run baseline per each subfolder
    """
    args = parse_args()
    for subtask in ["easy", "medium", "hard"]:
        run_baseline(read_problem_files(args.input+f"/{subtask}"), args.output+f"/{subtask}")

