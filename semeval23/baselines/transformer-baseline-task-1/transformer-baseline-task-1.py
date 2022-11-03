#!/usr/bin/env python3
import argparse
import json
import pandas as pd
import numpy as np
from simpletransformers.classification import ClassificationModel
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)

    return parser.parse_args()


def load_input(df):
    if type(df) != pd.DataFrame:
        df = pd.read_json(df, lines=True)
    
    ret = []
    for _, i in df.iterrows():
        ret += [{'text': ' '.join(i['postText']) + ' - ' + i['targetTitle'] + ' ' + ' '.join(i['targetParagraphs']), 'uuid': i['uuid']}]
    
    return pd.DataFrame(ret)


def use_cuda():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def predict(df):
    df = load_input(df)
    labels = ['phrase', 'passage', 'multi']
    model = ClassificationModel('deberta', '/model', use_cuda=use_cuda())

    uuids = list(df['uuid'])
    texts = list(df['text'])
    predictions = model.predict(texts)[1]
    
    for i in range(len(df)):
        yield {'uuid': uuids[i], 'spoilerType': labels[np.argmax(predictions[i])]}


def run_baseline(input_file, output_file):
    with open(output_file, 'w') as out:
        for prediction in predict(input_file):
            out.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    args = parse_args()
    run_baseline(args.input, args.output)

