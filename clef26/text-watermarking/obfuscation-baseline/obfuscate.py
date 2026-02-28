#!/usr/bin/env python3
from tira.check_format import lines_if_valid
from pathlib import Path
import click
import pandas as pd


@click.command()
@click.argument("watermarked_texts", type=Path)
@click.argument("original_texts", type=Path)
@click.argument("output_directory", type=Path)
def main(watermarked_texts, original_texts, output_directory):
    watermarked_data = lines_if_valid(watermarked_texts, "*.jsonl")

    for i in range(len(watermarked_data)):
        watermarked_data[i]["truth_label"] = "watermarked"

    # for the first 30 entries we remove the watermark
    for i in range(30):
        watermarked_data[i]["text"] = watermarked_data[i]["text"].replace("y12", "")

    data = {i["id"]: i for i in watermarked_data}

    original_data = lines_if_valid(original_texts, "*.jsonl")
    # we add some more original documents that are not watermarked
    for i in range(30):
        original_data[i]["truth_label"] = "not-watermarked"
        data[original_data[i]["id"]] = original_data[i]

    # attention: the field truth_label is removed from the tira workflow for system inputs, but not for the truth data
    output_directory.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(data.values()).to_json(output_directory / "texts.jsonl", lines=True, orient="records")


if __name__ == '__main__':
    main()

