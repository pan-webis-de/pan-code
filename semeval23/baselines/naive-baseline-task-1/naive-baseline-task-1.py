#!/usr/bin/python3
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)

    return parser.parse_args()


def run_baseline(input_file, output_file):
    with open(input_file, 'r') as inp, open(output_file, 'w') as out:
        for i in inp:
            i = json.loads(i)

            prediction = {'uuid': i['uuid'], 'spoilerType': 'passage'}
            out.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    args = parse_args()
    run_baseline(args.input, args.output)

