#!/usr/bin/env python3
from pathlib import Path

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import click

@click.command()
@click.option('--dataset', default='generative-ai-authorship-verification-panclef-2025/pan25-generative-ai-detection-smoke-test-20250428-training', help='The dataset to run predictions on (can point to a local directory).')
@click.option('--output', default=Path(get_output_directory(str(Path(__file__).parent))) / "predictions.jsonl", help='The file where predictions should be written to.')
def main(dataset, output):
    # Load the data
    tira = Client()
    df = tira.pd.inputs(dataset)

    # Make predictions
    df['label'] = 0.5

    # Save the predictions
    df[["id", "label"]].to_json(output, orient="records", lines=True)

if __name__ == "__main__":
    main()
