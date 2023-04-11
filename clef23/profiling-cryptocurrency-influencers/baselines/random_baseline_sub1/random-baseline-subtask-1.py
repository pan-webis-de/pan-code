#!/usr/bin/python3
import argparse
import os
from pathlib import Path
from typing import DefaultDict
import pandas as pd
import numpy as np
import random
import json

random.seed(0)

LABELS=['nano', 'micro', 'macro', 'mega', 'no influencer']

def load_file(input_directory: Path):
    """ Load a labels.jsonl file, convert it to array representation and return the array.
     This function assumes that test and prediction files have the same order of works.
    """
    user_id=[]
    user_text=[]
    with open(input_directory, 'r') as inp:
        for i in inp:
            tmp=json.loads(i)
            user_id.append(tmp['twitter user id'])
            user_text.append(tmp['texts'])
    df_to_predict = pd.DataFrame(list(zip(user_id, user_text)), columns =['twitter user id', 'text'])
    return df_to_predict 

def run_random_baseline(df_to_predict, output_file):
    user_id=[]
    user_probs=[]
    user_label=[]  
    for user, user_texts in df_to_predict.groupby("twitter user id"):
        print (user_texts.columns)
        user_values = [random.choice(LABELS) for i, text in enumerate(user_texts['text'])]
        user_label.append(max(user_values))
        user_probs.append(1.0)
        user_id.append(user)    
    df_output = pd.DataFrame(list(zip(user_id, user_label, user_probs)), columns =['twitter user id', 'class', 'probability'])
    df_output.to_json(output_file, orient='records', lines=True)
  
    

def main():
    parser = argparse.ArgumentParser(description='This is a baseline for task 1 that predicts influencer profiling.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)
    args = parser.parse_args()
    
    df_to_predict = load_file(Path(args.input))   
    run_random_baseline(df_to_predict, args.output)


if __name__ == '__main__':
    main()
    