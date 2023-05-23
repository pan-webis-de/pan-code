#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Evaluation script for the Profiling Cryptocurrency Influencers with Few-shot Learning task @PAN2023 - Low-resource influencer profiling - subtask1.
## Measures
The following evaluation measures are provided:
    - macro F1-score
    - Accuacy
    - Precission
    - Recall
    - F1-score
## Formats
The script requires two path, one for the ground truth (gold standard)
and one for the system predictions. The directory should be have a json file
path_to_truth - test_truth.json: e.g.
``` json
    {"twitter user id":"2f5a43bbfb7fcb3623222de6305b45b9","class":"macro"}
    {"twitter user id":"5509d8b2ba81eb6bc580a8aaa89b335a","class":"macro"}
    
```
path-to-model-output - subtask1.json: e.g.
``` json
    {"twitter user id":"2f5a43bbfb7fcb3623222de6305b45b9","class":"macro", "probability": 0.99 }
    {"twitter user id":"5509d8b2ba81eb6bc580a8aaa89b335a","class":"macro", "probability": 0.65}
    
```

Only files will be considered that:
- have the `.json` extension
- are properly encoded as UTF-8.
## Dependencies:
- Python 3.7+ (we recommend the Anaconda Python distribution)
- scikit-learn
- pandas
## Usage
From the command line:
>>> python pan23-author-profiling-evaluator.py -g path_to_truth -s path-to-model-output -o OUTPUT
where
    path_to_truth is the path to the file of 'test_truth.json' file with the ground gold labels
    path-to-model-output is the path to the file of the 'subtask1.json' file for a submitted method
    OUTPUT is the path to the folder where the results of the evaluation will be saved
Example: 
>>> python pan23-author-profiling-evaluator-subtask1.py -g "subtask1/test_truth.json" \
        -s "submitted_models/subtask1.json" \
        -o "output_evaluation_subtask1"

"""

import os
from pathlib import Path
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import DefaultDict
import pandas as pd
import numpy as np

TASKS_LABELS = {"subtask1": ["nano", "mega", "micro", "macro", "no influencer"]}


class Evaluation:
    def __init__(self):
        pass

    def macro_f1_score(self, x, y):
        return f1_score(x, y, average="macro")

    def evaluate_user_level(self, class_gold, class_silver, list_labels):
        metrics = {}
        metrics["uacc"] = accuracy_score(class_gold, class_silver) * 100
        metrics["umF1"] = self.macro_f1_score(class_gold, class_silver) * 100
        report_ = classification_report(class_gold, class_silver, output_dict=True)
        for k in list_labels:
            metrics[k + "_precicion"] = report_[k]["precision"] * 100
            metrics[k + "_recall"] = report_[k]["recall"] * 100
            metrics[k + "_f1"] = report_[k]["f1-score"] * 100
        return metrics

    def evaluation(self, df_gold, df_silver, list_labels):
        """ Calculate the evaluation results.
          Extended results also include per-label that should not be integrated into TIRA but are usefull for
          evaluating the participant performance during development.
          Truth and Predictions must be in array representation.
          Principal evaluation metric:
          - macro F1
          Extended evaluation shows:
          - accuracy
          - precision
          - recall
          - F1 scores
        """
        metrics_user = self.evaluate_user_level(df_gold["class"], df_silver["class"], list_labels)
        return metrics_user


def load_file(input_directory: Path):
    """ Load a labels.jsonl file, convert it to array representation and return the array.
     This function assumes that test and prediction files have the same order of works.
    """
    try:
        df_ = pd.read_json(input_directory, lines=True)
        return df_
    except IOError:
        print("Could not read file:", input_directory)


def write_evaluations(results: dict, output_directory: Path):
    """ Write the evaluation results to a file.
     @param results: A dictionary with evaluation results
     @param output_directory: A directory where to write the output file to.
     """
    try:
        df_results = pd.DataFrame(results, index=[0])
        output_directory.mkdir(parents=True, exist_ok=True)
        ##Write TIRA format
        with open(output_directory.joinpath("evaluation.prototext"), 'w+') as of:
            of.write('measure{{\n  key: "{}"\n  value: "{}"\n}}\n'.format('subtask1', str(results['umF1'])))
        ##Write json
        df_results.to_json(output_directory.joinpath("evaluation.json"), orient='records', lines=True)
    except IOError:
        print("Could not read file:", output_directory)

def check_user_id(df_truth, df_prediction):
    user_id_prediction_check=[]
    class_prediction_check=[]
    for i in df_truth['twitter user id']:
        user_id_prediction_check.append(i)
        class_prediction_check.append(df_prediction[df_prediction['twitter user id']==i]['class'].tolist()[0])
    dict_ = {"twitter user id": user_id_prediction_check, 'class': class_prediction_check} 
    df_prediction_check=pd.DataFrame(dict_)
    return (df_prediction_check)
        
def main():
    parser = argparse.ArgumentParser(description='Evaluation script Author Profiling@PAN2023 - subtask1')
    parser.add_argument('-g', type=Path,
                        help='Path to test_truth.json that contains truth')
    parser.add_argument('-s', type=Path,
                        help='Path to subtask1.json that contains system predictions')
    parser.add_argument('-o', type=Path, help='Path to output files', default="./output_evaluation_subtask1")
    args = parser.parse_args()

    output_path = args.o
    # check load files
    if not args.g:
        raise ValueError('The path to truth file is required')
    if not args.s:
        raise ValueError('The path to model prediction  is required')
    if not args.o:
        raise ValueError('The output path is required')

    # loads files
    df_truth = load_file(args.g)
    df_prediction = load_file(args.s)
    
    #Check user id
    df_prediction_check = check_user_id(df_truth, df_prediction)
    
    # evaluation

    list_labels = (list(set(df_truth["class"])))
    eval = Evaluation()
    metrics_user = eval.evaluation(df_truth, df_prediction_check, list_labels)
    print(metrics_user)

    write_evaluations(metrics_user, output_path)


if __name__ == '__main__':
    main()
