#!/usr/bin/env python3
import argparse
import json

def yield_all_data_points():
    with open('/mnt/ceph/tira/data/datasets/test-datasets-truth/clickbait-spoiling/task-2-spoiler-generation-20221115-test/test.jsonl') as f:
        for i in f:
            yield json.loads(i)

def main(args):
    to_remove = fields_to_remove(args)
#    with open('input.jsonl') as f:
    for i in yield_all_data_points():
        print(i.keys())
        break

def parse_args():
    parser = argparse.ArgumentParser(description='This script transforms the raw test-data to the input for algorithms')

    parser.add_argument('--task', type=str, help='The task: either 1 or 2.', choices=["1", "2"], required=True)

    return parser.parse_args()

def fields_to_remove(args):
    ret = []
    if args.task == '1':
        ret += ['sda']
        return set(ret)
    elif args.task == '2':
        return set(ret)

    raise ValueError(f'Could not handle task {args.task}')

if __name__ == '__main__':
    main(parse_args())

