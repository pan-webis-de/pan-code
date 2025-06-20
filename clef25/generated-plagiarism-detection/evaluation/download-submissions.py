#!/usr/bin/env python3
import click
from tira.rest_api_client import Client
from pathlib import Path
from tqdm import tqdm
from tira.io_utils import parse_prototext_key_values
import shutil
import json
import yaml
import os
import zipfile

def zip_directory_contents(dir_path, zip_name, dataset):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                abs_file_path = os.path.join(root, file)
                # Relative to the parent of dir_path
                arcname = dataset + "/" + os.path.relpath(abs_file_path, start=dir_path)
                zipf.write(abs_file_path, arcname)

def load_ir_metadata(directory):
    try:
        res = yaml.safe_load(open(f"{directory}/tira-ir-metadata.yml"))["resources"]
    except:
        return {}

    ret = {
        "wallcock": res["runtime"]["wallclock"],
        "CPU (Max)": res["cpu"]["used process"]["max"],
        "RAM (Max)": res["ram"]["used process"]["max"]
    }
    
    if "used system" in res["gpu"] and "vram used system" in res["gpu"]:
        ret["GPU Utilization (Max)"] = res["gpu"]["used system"]["max"]
        ret["GPU VRAM (Max)"] = res["gpu"]["vram used system"]["max"]
    
    return ret


@click.command()
@click.option("--task", type=click.Choice(["pan25-generated-plagiarism-detection"]), required=True, help="The task id in tira. See https://archive.tira.io/datasets?query=plagiarism")
@click.option("--datasets", type=click.Choice(["pan12-text-alignment-partition-01-20250609-test", "llm-plagiarism-detection-partition-01-20250527-test", "llm-plagiarism-detection-partition-02-20250527-test"]), multiple=True, help="The dataset id in tira. See https://archive.tira.io/datasets?query=plagiarism")
@click.option("--output", default="runs.jsonl")
def main(task, datasets, output):
    tira = Client()
    evaluation_results = {}
    with open(output, "w") as f:
        for dataset in datasets:
            for _, submission in tqdm(list(tira.submissions(task, dataset).iterrows())):
                run_directory = tira.download_zip_to_cache_directory(task=task, dataset=dataset, team=submission["team"], run_id=submission["run_id"])
                if submission["is_evaluation"]:
                     try:
                         evaluation_results[submission["input_run_id"]] = {i["key"]: i["value"] for i in parse_prototext_key_values(run_directory + "/evaluation.prototext")}
                     except:
                         pass

            for _, submission in tira.submissions(task, dataset).iterrows():
                run_directory = tira.download_zip_to_cache_directory(task=task, dataset=dataset, team=submission["team"], run_id=submission["run_id"])
                if submission["is_evaluation"]:
                     continue
                i = submission.to_dict().copy()
                i["evaluation"] = {} if i["run_id"] not in evaluation_results else evaluation_results[i["run_id"]]
                i["used_resources"] = load_ir_metadata(run_directory)
                f.write(json.dumps(i) + "\n")
                shutil.copytree(Path(run_directory).parent, Path(Path(output).parent) / "runs" / dataset / Path(run_directory).parent.name)
            
            dataset_inputs = tira.download_dataset(task, dataset, False)
            zip_directory_contents(Path(dataset_inputs), Path(Path(output).parent) / f"{dataset}-inputs.zip", dataset)
            dataset_truths = tira.download_dataset(task, dataset, True)
            zip_directory_contents(Path(dataset_truths), Path(Path(output).parent) / f"{dataset}-truths.zip", dataset)

if __name__ == '__main__':
    main()
