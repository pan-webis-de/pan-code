#!/usr/bin/env python3
import json
from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import click

def run_baseline(problems: "pd.DataFrame", output_path: Path, pred: int):
    """
    Write predictions to solution files in the format:
    {
    "changes": [0, 0, 0, 1]
    }

    :param problems: dictionary of problem files with ids as keys
    :param output_path: output folder to write solution files
    :param pred: the static prediction
    """
    print(f'Write outputs {len(problems)} problems to to {output_path}.')
    for _, i in problems.iterrows():
        output_file = output_path / i["file"].replace("/problem-", "/solution-problem-").replace(".txt", ".json").replace("/train/", "/").replace("/test/", "/").replace("/validation/", "/")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as out:
            paragraphs = i["paragraphs"]
            prediction = {'changes': [pred for _ in range(len(paragraphs) -1)]}
            out.write(json.dumps(prediction))

@click.command()
@click.option('--dataset', default='multi-author-writing-style-analysis-2025/multi-author-writing-spot-check-20250503-training', help='The dataset to run predictions on (can point to a local directory).')
@click.option('--output', default=Path(get_output_directory(str(Path(__file__).parent))), help='The file where predictions should be written to.')
@click.option('--predict', default=0,  help='The prediction to make.')
def main(dataset, output, predict):
    tira = Client()
    # alternatively, you can still load the data programmatically, i.e., without tira.pd.inputs.
    # See https://github.com/pan-webis-de/pan-code/blob/master/clef24/multi-author-analysis/baselines/baseline-static-prediction.py
    input_df = tira.pd.inputs(dataset, formats=["multi-author-writing-style-analysis-problems"])

    for subtask in ["easy", "medium", "hard"]:
        run_baseline(input_df[input_df["task"] == subtask], Path(output), predict)

if __name__ == '__main__':
    main()

