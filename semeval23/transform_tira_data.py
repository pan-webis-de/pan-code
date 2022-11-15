#!/usr/bin/env python3
import argparse
import json
from tqdm import tqdm

def yield_all_data_points():
    with open('/mnt/ceph/tira/data/datasets/test-datasets-truth/clickbait-spoiling/task-2-spoiler-generation-20221115-test/test.jsonl') as f:
        for i in f:
            yield json.loads(i)

def main(args):
    to_remove = fields_to_remove(args)
    print(f'I remove the fields {to_remove}')
    with open('input.jsonl', 'w') as f:
        keys = []
        for i in tqdm(list(yield_all_data_points())):
            for field_to_remove in to_remove:
                del i[field_to_remove]
            keys += list(i.keys())
            f.write(json.dumps(i) + '\n')

    print(f'The data has keys: {set(keys)}')


def parse_args():
    parser = argparse.ArgumentParser(description='This script transforms the raw test-data to the input for algorithms')

    parser.add_argument('--task', type=str, help='The task: either 1 or 2.', choices=["1", "2"], required=True)

    return parser.parse_args()


def fields_to_remove(args):
    ret = ['spoiler', 'spoilerPositions', 'provenance']
    if args.task == '1':
        ret += ['tags']
        return set(ret)
    elif args.task == '2':
        return set(ret)

    raise ValueError(f'Could not handle task {args.task}')


if __name__ == '__main__':
    main(parse_args())

