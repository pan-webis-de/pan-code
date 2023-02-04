#!/usr/bin/env python3

import argparse
from os.path import exists
from glob import glob
from os.path import isdir
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from bert_score import score
import subprocess
import tempfile
from copy import deepcopy


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
    
    if isdir(f):
        f = glob(f + '/*.json*')
        
        if len(f) != 1:
            error('The input is an directory that contains multiple json files. Please create only a single json file. Got ' + str(f))
        
        f = f[0]
    
    with open(f, 'r') as inp:
        for l in inp:
            try:
                ret += [json.loads(l)]
            except:
                error('Invalid line ' + str(num) + ' in "' + f + '" with content: ' + l.strip())
            num += 1

    success('The file ' + f + ' is in JSONL format.')
    return ret

def spoiler_predictions_to_map(l, error=error, field='spoilerType'):
    if l is None or len(l) == 0:
        error('Spoiler predictions are empty.')
    uuids = []

    for i in l:
        if 'uuid' not in i.keys() or field not in i.keys():
            error(f'Spoiler predictions do not have all required fields. Expected fields "uuid" and "{field}". Got: ' + str(i))
            return
        uuids += [i['uuid']]

    if len(l) != len(set(uuids)):
            error('Spoiler predictions have dupliates. I found ' + str(len(l)) + ' entries but only ' + str(len(set(uuids))) + ' unique uuids.')
            return

    success('Spoiler predictions have correct format. Found ' + str(len(l)))
    return {i['uuid']: i[field] if type(i[field]) is not list else i[field][0] for i in l}

def normalize_spoiler_generation(i, error, expected_spoiler_type=None):
    if 'uuid' not in i or 'spoiler' not in i:
        error('Spoiler generation does not have all required fields. Expected fields are uuid and spoiler. Got: ' + str(i))
        return

    if expected_spoiler_type and expected_spoiler_type not in i['tags']:
    	return True

    return {i['uuid']: i['spoiler']}

def spoiler_generations_to_map(l, error=error, expected_spoiler_type=None):
    if l is None or len(l) == 0:
        error('Spoiler predictions are empty.')
    uuids = []

    for i in deepcopy(l):
        i = normalize_spoiler_generation(i, error, expected_spoiler_type)
        if not i:
            return
        elif i is True:
            continue
        uuids += list(i.keys())

    if not expected_spoiler_type and len(l) != len(set(uuids)):
            error('Spoiler generations have dupliates. I found ' + str(len(l)) + ' entries but only ' + str(len(set(uuids))) + ' unique uuids.')

    l = [normalize_spoiler_generation(i, error, expected_spoiler_type) for i in l]
    l = [i for i in l if i and i is not True]

    success('Spoiler generations have correct format. Found ' + str(len(l)))
    ret = {}
    for i in l:
        for k, v in i.items():
            assert k not in ret
            ret[k] = v
    
    return ret


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate submissions to the clickbait spoiling task.')

    parser.add_argument('--input_run', type=str, help='The input run (expected in jsonl format) produced by a system that should be evaluated.', required=True)
    parser.add_argument('--ground_truth_classes', type=str, help='The ground truth classes used to evaluate submissions to task 1 (spoiler type generation). For the evaluation of task 2 (spoiler generation), this can be different from "--ground_truth_spoilers" to evaluate the effectiveness using real spoiler predictions.', required=False)
    parser.add_argument('--ground_truth_spoilers', type=str, help='The ground truth spoilers used to evaluate submissions to task 2 (spoiler generation).', required=False)
    parser.add_argument('--task', type=str, help='The task to evaluate. Choose 1 (spoiler type classification) or 2 (spoiler generation).', choices=['1', '2'], required=True)
    parser.add_argument('--output_prototext', type=str, help='Write evalualuation results as prototext file to this location.', required=False)


    return parser.parse_args()

def to_prototext(d):
    ret = ''
    
    for k, v in d.items():
        ret += 'measure{\n  key: "' + str(k) + '"\n  value: "' + str(v) + '"\n}\n'
    
    return ret.strip()

def filter_to(y_true, y_pred, filter_value):
    y_true_filtered, y_pred_filtered = [], []
    for i in range(len(y_true)):
        if y_true[i] == filter_value or y_pred[i] == filter_value:
            y_true_filtered += [1 if y_true[i] == filter_value else 0]
            y_pred_filtered += [1 if y_pred[i] == filter_value else 0]
    
    return (y_true_filtered, y_pred_filtered)

def precision_on(y_true, y_pred, filter_value):
    y_true_filtered, y_pred_filtered = filter_to(y_true, y_pred, filter_value)

    return precision_score(y_true_filtered, y_pred_filtered)

def recall_on(y_true, y_pred, filter_value):
    y_true_filtered, y_pred_filtered = filter_to(y_true, y_pred, filter_value)

    return recall_score(y_true_filtered, y_pred_filtered)


def f1_on(y_true, y_pred, filter_value):
    y_true_filtered, y_pred_filtered = filter_to(y_true, y_pred, filter_value)

    return f1_score(y_true_filtered, y_pred_filtered)


def create_protobuf_for_task_1(actual, expected):
    keys = sorted(actual.keys())
    missing_predictions = 0
    
    y_true = []
    y_pred = []
    
    for k in keys:
        y_true += [expected[k]]
        
        if k in actual:
            y_pred += [actual[k]]
        else:
            missing_predictions += 1
            y_pred += ['']

    return {
        "result-size": len(keys),
        'balanced-accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision-for-phrase-spoilers': precision_on(y_true, y_pred, 'phrase'),
        'recall-for-phrase-spoilers': recall_on(y_true, y_pred, 'phrase'),
        'f1-for-phrase-spoilers': f1_on(y_true, y_pred, 'phrase'),
        'precision-for-passage-spoilers': precision_on(y_true, y_pred, 'passage'),
        'recall-for-passage-spoilers': recall_on(y_true, y_pred, 'passage'),
        'f1-for-passage-spoilers': f1_on(y_true, y_pred, 'passage'),
        'precision-for-multi-spoilers': precision_on(y_true, y_pred, 'multi'),
        'recall-for-multi-spoilers': recall_on(y_true, y_pred, 'multi'),
        'f1-for-multi-spoilers': f1_on(y_true, y_pred, 'multi'),
        'missing-predictions': missing_predictions
    }

def eval_task_1(input_run, ground_truth_classes, output_file):
    input_run = spoiler_predictions_to_map(input_run)
    ret = None
    if ground_truth_classes == None:
        success('No ground-truth is passed. I tested the input run and the input run is valid.')
        ret = to_prototext({"result-size": len(input_run.keys())})
        
    else:
        ground_truth_classes = spoiler_predictions_to_map(ground_truth_classes, field='tags')
        ret = to_prototext(create_protobuf_for_task_1(input_run, ground_truth_classes))

    if output_file:
        with open(output_file, 'w') as f:
            f.write(ret)

def bleu_score(truth, prediction):
    """
    From: https://github.com/webis-de/acl22-clickbait-spoiling/blob/470f488bd532da1e75812de6a94458ec80fdb2b9/evaluation/meteor-metric.py#L72
    """

    def stopfilter(tokens):
        tmp = [token for token in tokens if token not in stopwords.words('english')]
        res = [token.lower() for token in tmp if token not in string.punctuation]
        return res

    def make_score(trut, predi):
        if len(trut) > 3 and len(predi) > 3:
            weights = (1./4., 1./4., 1./4., 1./4.)
        elif len(trut) > 2 and len(predi) > 2:
            weights = (1./3., 1./3., 1./3.)
        elif len(trut) > 1 and len(predi) > 1:
            weights = (1./2., 1./2.)
        else:
            weights = (1., 0.)

        if (len(weights) == 4) and (len(trut) < 4 or len(predi) < 4):
            print(trut)
            print(predi)
            print(weights)
            print('\n')

        return sentence_bleu([trut], predi, weights=weights)

    score = 0.
    lem_score = 0.

    write_dict = {'single_scores': {}, 'scores': {}}

    for i in range(len(truth)):
        real_answer = truth[i]
        if type(real_answer) is list:
            real_answer = ' '.join(real_answer)

        pred_answer = prediction[i]
        if type(pred_answer) is list:
            pred_answer = ' '.join(pred_answer)

        lem_truth_tokens = stopfilter(word_tokenize(real_answer.replace('\n', '')))
        lem_prediction_tokens = stopfilter(word_tokenize(pred_answer.replace('\n', '')))
        i_lem_score = make_score(lem_truth_tokens, lem_prediction_tokens)
        lem_score += i_lem_score

    return lem_score / len(truth)


def bert_score(truth, prediction):
    assert len(truth) == len(prediction)
    prec, rec, f1 = score(prediction, truth, lang="en")
    
    return float(f1.mean())


def meteor_score(truth, prediction):
    with tempfile.TemporaryDirectory() as tmpdirname:
        assert len(truth) == len(prediction)
        
        with open(tmpdirname + '/truths.txt', 'w') as truths, open(tmpdirname + '/preds.txt', 'w') as preds:
            for t in truth:
                truths.write(t + '\n')
            
            for p in prediction:
                preds.write(p + '\n')

        cmd = ['java', '-Xmx2G', '-jar', '/meteor-1.5.jar', tmpdirname + '/truths.txt', tmpdirname + '/preds.txt', '-l', 'en', '-norm', '-t', 'adq']
        meteor_output = subprocess.check_output(cmd).decode('utf-8')
        try:
            return float(meteor_output.split('\n\nFinal score:')[1].strip())
        except:
            raise ValueError('Could not extract the final score out of "' + meteor_output + '".')


def create_protobuf_for_task_2(actual, expected):
    keys = sorted(expected.keys())
    missing_predictions = 0
    
    y_true = []
    y_pred = []
    
    for k in keys:
        exp = expected[k]
        if type(exp) is list:
            exp = ' '.join(exp)
        
        y_true += [exp.replace('\n', ' ').strip()]
        
        if k in actual:
            act = actual[k]
            if type(act) is list:
                act = ' '.join(act)
            
            y_pred += [act.replace('\n', ' ').strip()]
        else:
            missing_predictions += 1
            y_pred += ['']

    return {
        "result-size": len(keys),
        'bleu-score': bleu_score(y_true, y_pred),
        'bert-score': bert_score(y_true, y_pred),
        'meteor-score': meteor_score(y_true, y_pred),
        'missing-predictions': missing_predictions
    }

def eval_task_2(input_run, ground_truth_classes, ground_truth_spoilers, output_file):
    input_run = spoiler_generations_to_map(input_run)
    if ground_truth_spoilers == None:
        ret = to_prototext({"result-size": len(input_run.keys())})
        success('No ground-truth is passed. I tested the input run and the input run is valid.')
    else:
        ret = {}
        for (display_name, tag_name) in [('all-spoilers', None), ('phrase-spoilers', 'phrase'), ('passage-spoilers', 'passage'), ('multi-spoilers', 'multi')]:
            print('Run evaluation for ' + display_name)
            filtered_ground_truth_spoilers = spoiler_generations_to_map(deepcopy(ground_truth_spoilers), expected_spoiler_type=tag_name)

            for k,v in create_protobuf_for_task_2(input_run, filtered_ground_truth_spoilers).items():
                ret[k + '-' + display_name] = v

        ret = to_prototext(ret)

	
    if output_file:
        with open(output_file, 'w') as f:
            f.write(ret)

if __name__ == '__main__':
    args = parse_args()
    input_run = load_json_lines(args.input_run)
    ground_truth_classes = None if not args.ground_truth_classes else load_json_lines(args.ground_truth_classes)
    ground_truth_spoilers = None if not args.ground_truth_spoilers else load_json_lines(args.ground_truth_spoilers)

    if args.task == '1':
        eval_task_1(input_run, ground_truth_classes, args.output_prototext)
    elif args.task == '2':
        eval_task_2(input_run, ground_truth_classes, ground_truth_spoilers, args.output_prototext)
    else:
        error('Unknown task. Expected 1 or 2. Got: ' + str(args.task))

