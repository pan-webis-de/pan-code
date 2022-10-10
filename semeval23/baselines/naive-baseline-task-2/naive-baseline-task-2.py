#!/usr/bin/python3
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 2 that spoils each clickbait post with the title of the linked page.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The spoiled posts in jsonl format.', required=False)

    return parser.parse_args()


def predict(inputs):
    for i in inputs:
        yield {'uuid': i['uuid'], 'spoiler': i['targetTitle']}


def run_baseline(input_file, output_file):
    with open(input_file, 'r') as inp, open(output_file, 'w') as out:
        inp = [json.loads(i) for i in inp]
        for output in predict(inp):
            out.write(json.dumps(output) + '\n')


if __name__ == '__main__':
    args = parse_args()
    run_baseline(args.input, args.output)

