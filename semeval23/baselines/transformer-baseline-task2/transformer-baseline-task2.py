#!/usr/bin/env python3
import argparse
import json
import pandas as pd
from run_qa import main

DEFAULT_MODEL='/model'


def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 2 that spoils each clickbait post with the title of the linked page.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The spoiled posts in jsonl format.', required=False)
    parser.add_argument('--model', type=str, help='The path to the model.', default=DEFAULT_MODEL)

    return parser.parse_args()


def predict(inputs, model=DEFAULT_MODEL):
    return main(['--model_name_or_path', model, '--output_dir', '/tmp/ignored'], input_data_to_squad_format(inputs))


def input_data_to_squad_format(inp):
    return pd.DataFrame([{'id': i['uuid'], 'title': i['targetTitle'], 'question': ' '.join(i['postText']), 'context': i['targetTitle'] + ' - ' + (' '.join(i['targetParagraphs'])), 'answers': 'not available for predictions'} for i in inp])


def run_baseline(input_file, output_file, model):
    inp = [json.loads(i) for i in open(input_file, 'r')]
    with open(output_file, 'w') as out:
        for prediction in predict(inp):
            out.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    args = parse_args()
    run_baseline(args.input, args.output, args.model)

