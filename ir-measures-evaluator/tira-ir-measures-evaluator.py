#!/usr/bin/env python3

import argparse
from os.path import exists
import ir_measures

def error(msg):
    print('  [\033[91mx\033[0m] ' + msg)
    exit(1)

def success(msg):
    print('  [\033[92mo\033[0m] ' + msg)

def load_qrels(f):
    if not exists(f):
        error('The file "' + f + '" does not exist.')
    
    qrels = ir_measures.read_trec_qrels(f)

def load_run(f):
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



def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate submissions to TIRA with ir_measures.')

    parser.add_argument('--input_run', type=str, help='The input run (expected in TREC format) produced by a system that should be evaluated.', required=True)
    parser.add_argument('--qrel_file', type=str, help='The ground truth qrel file used to evaluate submissions.', required=False)
    parser.add_argument('--measures', type=str, help='The measures that should be evaluated. (e.g., NumRet, nDCG@10, ...)')
    parser.add_argument('--output_prototext', type=str, help='Write evalualuation results as prototext file to this location.', required=False)

    return parser.parse_args()

def eval_task_on_measures(run, qrels, measures, output_file, digits=3):
    input_run = spoiler_predictions_to_map(input_run)
    eval_results = ir_measures.calc_aggregate(measures, qrels, run)
    
    
    value_format = "%0." + str(digits) + "f"
    measure_strings = [ "measure{\n  key: \"%s\"\n  value: \"%s\"\n}" % (name, value_format % value) for (name, value) in eval_results.items()]
    
    return "\n".join(measure_strings)
    ret 
    if ground_truth_spoilers == None:
        success('No ground-truth is passed. I tested the input run and the input run is valid.')
        ret = "measure{\n  key: \"result-size\"\n  value: \"" + str(len(input_run.keys())) + "\"\n}"
        
    else:
        error('ToDo: The evaluator currently only checks if the format is valid')

    if output_file:
        with open(output_file, 'w') as f:
            f.write(ret)

if __name__ == '__main__':
    args = parse_args()
    
    qrels = load_qrels(args.qrel_file)
    measures = [eval(measure) for measure in args.measures]
    run = load_run(args.input_run)
    
    eval_task_on_measures(run, qrels, measures)

