#!/usr/bin/env python3

import argparse
from os.path import exists
import json

def error(msg):
    print('  [\033[91mx\033[0m] ' + msg)
    exit(1)

def success(msg):
    print('  [\033[92mo\033[0m] ' + msg)

def load_json_lines(f):
    if not exists(f):
        error('The file "' + f + '" does not exist.')

    ret = []
    num = 1
    with open(f, 'r') as inp:
        for l in inp:
            try:
                ret += [json.loads(l)]
            except:
                error('Invalid line ' + str(num) + ' in "' + f + '" with content: ' + l.strip())
            num += 1

    success('The file ' + f + ' is in JSONL format.')
    return ret

def spoiler_predictions_to_map(l):
    if l is None or len(l) == 0:
        error('Spoiler predictions are empty.')
    uuids = []

    for i in l:
        if 'uuid' not in i.keys() or 'spoiler_type' not in i.keys():
            error('Spoiler predictions do not have all required fields. Expected fields "uuid" and "spoiler_type". Got: ' + str(i))
        uuids += [i['uuid']]

    if len(l) != len(set(uuids)):
            error('Spoiler predictions have dupliates. I found ' + str(len(l)) + ' entries but only ' + str(len(set(uuids))) + ' unique uuids.')

    success('Spoiler predictions have correct format. Found ' + str(len(l)))
    return {i['uuid']: i['spoiler_type'] for i in l}

def normalize_spoiler_generation(i):
    if 'uuid' not in i or ('generic_spoiler' not in i and ('phrase_spoiler' not in i or 'passage_spoiler' not in i or 'multi_spoiler' not in i)):
        error('Spoiler generation does not have all required fields. Expected fields are uuid and either phrase_spoiler, passage_spoiler, multi_spoiler, or generic_spoiler. Got: ' + str(i))

    ret = {'uuid': i['uuid']}

    for t in ['phrase_spoiler', 'passage_spoiler', 'multi_spoiler']:
        ret[t] = i.get(t, i.get('generic_spoiler', None))

    return ret

def spoiler_generations_to_map(l):
    if l is None or len(l) == 0:
        error('Spoiler predictions are empty.')
    uuids = []

    for i in l:
        i = normalize_spoiler_generation(i)
        uuids += [i['uuid']]

    if len(l) != len(set(uuids)):
            error('Spoiler generations have dupliates. I found ' + str(len(l)) + ' entries but only ' + str(len(set(uuids))) + ' unique uuids.')

    l = [normalize_spoiler_generation(i) for i in l]

    success('Spoiler generations have correct format. Found ' + str(len(l)))
    return {i['uuid']: i for i in l}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate submissions to the clickbait spoiling task.')

    parser.add_argument('--input_run', type=str, help='The input run (expected in jsonl format) produced by a system that should be evaluated.', required=True)
    parser.add_argument('--ground_truth_classes', type=str, help='The ground truth classes used to evaluate submissions to task 1 (spoiler type generation). For the evaluation of task 2 (spoiler generation), this can be different from "--ground_truth_spoilers" to evaluate the effectiveness using real spoiler predictions.', required=False)
    parser.add_argument('--ground_truth_spoilers', type=str, help='The ground truth spoilers used to evaluate submissions to task 2 (spoiler generation).', required=False)
    parser.add_argument('--task', type=str, help='The task to evaluate. Choose 1 (spoiler type classification) or 2 (spoiler generation).', choices=['1', '2'], required=True)
    parser.add_argument('--output_prototext', type=str, help='Write evalualuation results as prototext file to this location.', required=False)


    return parser.parse_args()

def eval_task_1(input_run, ground_truth_classes, output_file):
    input_run = spoiler_predictions_to_map(input_run)
    ret = None
    if ground_truth_spoilers == None:
        success('No ground-truth is passed. I tested the input run and the input run is valid.')
        ret = "measure{\n  key: \"result-size\"\n  value: \"" + str(len(input_run.keys())) + "\"\n}"
        
    else:
        error('ToDo: The evaluator currently only checks if the format is valid')

    if output_file:
        with open(output_file, 'w') as f:
            f.write(ret)

def eval_task_2(input_run, ground_truth_classes, ground_truth_spoilers):
    input_run = spoiler_generations_to_map(input_run)
    if ground_truth_spoilers == None:
        success('No ground-truth is passed. I tested the input run and the input run is valid.')
    else:
        error('ToDo: The evaluator currently only checks if the format is valid')

if __name__ == '__main__':
    args = parse_args()
    input_run = load_json_lines(args.input_run)
    ground_truth_classes = None if not args.ground_truth_classes else load_json_lines(args.ground_truth_classes)
    ground_truth_spoilers = None if not args.ground_truth_spoilers else load_json_lines(args.ground_truth_spoilers)

    if args.task == '1':
        eval_task_1(input_run, ground_truth_classes, args.output_prototext)
    elif args.task == '2':
        eval_task_2(input_run, ground_truth_classes, ground_truth_spoilers)
    else:
        error('Unknown task. Expected 1 or 2. Got: ' + str(args.task))

