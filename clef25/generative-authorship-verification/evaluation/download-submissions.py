#!/usr/bin/env python3
import click
from tira.rest_api_client import Client
from pathlib import Path
from tqdm import tqdm
import shutil
import json
import yaml

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
@click.option("--task", type=click.Choice(["generative-ai-authorship-verification-panclef-2025"]), required=True, help="The task id in tira. See https://archive.tira.io/datasets?query=ai-authorship-verification")
@click.option("--datasets", type=click.Choice(["pan25-generative-ai-detection-eloquent-20250605-test", "pan25-generative-ai-detection-eloquent-20250610-test", "pan25-generative-ai-detection-20250604-test"]), multiple=True, help="The dataset id in tira. See https://archive.tira.io/datasets?query=ai-authorship-verification")
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
                         evaluation_results[submission["input_run_id"]] = json.load(open(run_directory + "/evaluation.json"))
                     except:
                         pass

            for _, submission in tira.submissions(task, dataset).iterrows():
                run_directory = tira.download_zip_to_cache_directory(task=task, dataset=dataset, team=submission["team"], run_id=submission["run_id"])
                if not submission["is_evaluation"]:
                     i = submission.to_dict().copy()
                     if i["run_id"] not in evaluation_results:
                         continue
                     i["evaluation"] = evaluation_results[i["run_id"]]
                     i["used_resources"] = load_ir_metadata(run_directory)
                     f.write(json.dumps(i) + "\n")
                     shutil.copytree(Path(run_directory).parent, Path(output).parent / Path(run_directory).parent.name)

if __name__ == '__main__':
    main()
