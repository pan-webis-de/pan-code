import json
import os

from tqdm import tqdm
from evaluator import process as eval

def ret_mapper(ret):
    return {
        "Micro Plagdet": ret['micro_plagdet'], 
        "Micro Recall": ret['micro_recall'], 
        "Micro Precision": ret['micro_precision'], 
        "Macro Plagdet": ret['macro_plagdet'], 
        "Macro Recall": ret['macro_recall'], 
        "Macro Precision": ret['macro_precision'], 
        "Granularity": ret['granularity']
    }

runs = []
with open('runs.jsonl', 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        # print(line)
        try:
            runs.append(json.loads(line.strip()))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {line_num}: {line.strip()}")
            raise e

with open('runs_new.jsonl', 'w') as f:
    for run in tqdm(list(runs)):
        print(f"Start run {run['run_id']} by {run['team']}")
        run_path = f"submissions/runs/{run['run_id']}/output/"
        if not os.path.exists(run_path):
            print(f"Run path {run_path} not found.")
            continue

        plag_tag_name = "plagiarism"
        det_tag_name = "detected-plagiarism"

        for llm in ['DeepSeek-R1', 'Llama-3', 'Mistral']:
            for obfuscation in ['simple', 'medium', 'hard']:
                ret = eval(True, run['dataset'], plag_tag_name, run_path, det_tag_name, ".", llm, obfuscation)
                run[f'evaluation-{llm}-{obfuscation}'] = ret_mapper(ret)

        f.write( json.dumps(run) + "\n" )
