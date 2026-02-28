#!/usr/bin/env python3
from tira.check_format import lines_if_valid
from pathlib import Path
import click
import pandas as pd


def load_data(directory):
    ret = lines_if_valid(directory, "*.jsonl")
    return pd.DataFrame(ret)


@click.argument("input_directory", type=Path)
@click.argument("output_directory", type=Path)
def watermark(input_directory, output_directory):
    data = load_data(input_directory)

    # we just add "_xy123_" as watermark
    data["text"] = data["text"].apply(lambda i: i + " _xy123_")
    output_directory.mkdir(exist_ok=True, parents=True)

    data.to_json(output_directory / "watermarked-text.jsonl", lines=True, orient="records")


@click.argument("input_directory", type=Path)
@click.argument("output_directory", type=Path)
def detect(input_directory, output_directory):
    data = load_data(input_directory)
    
    # label should be 1.0 if our watermark "_xy123_" is in the text
    data["label"] = data["text"].apply(lambda i: 1.0 if "_xy123_" in i else 0.0)
    del data["text"]
    output_directory.mkdir(exist_ok=True, parents=True)
    data.to_json(output_directory / "detected-text.jsonl", lines=True, orient="records")


@click.group()
def main():
    pass


main.command()(watermark)
main.command()(detect)

if __name__ == '__main__':
    main()

