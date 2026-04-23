#!/usr/bin/env python3
from tira.check_format import lines_if_valid
from pathlib import Path
import click
import pandas as pd
import nltk
import random


# Sample attack
def sentence_shuffle_attack(text):
    sentences = nltk.sent_tokenize(text)
    random.shuffle(sentences)
    shuffled_text = " ".join(sentences)
    return shuffled_text


@click.command()
@click.argument("watermarked_texts", type=Path)
@click.argument("original_texts", type=Path)
@click.argument("output_directory", type=Path)
def main(watermarked_texts, original_texts, output_directory):
    watermarked_data = lines_if_valid(watermarked_texts, "*.jsonl")
    watermarked_data = pd.DataFrame.from_records(watermarked_data)

    truth_labels = []
    for i in range(len(watermarked_data)):
        truth_labels.append("watermarked")
    
    watermarked_data["truth_label"] = truth_labels

    original_data = lines_if_valid(original_texts, "*.jsonl")
    original_data = pd.DataFrame.from_records(original_data)
    rows = []
    for i in range(len(watermarked_data)):
        # We add the non-obfuscated watermarked text
        rows.append({"id": watermarked_data["id"][i], "text": watermarked_data["text"][i], "truth_label": "watermarked", "type": f"unchanged"})
        # We add the non-obfuscated original (non-watermarked) text
        rows.append({"id": original_data["id"][i], "text": original_data["text"][i], "truth_label": "not-watermarked", "type": f"unchanged"})
        # We attack the watermarked text
        rows.append({"id": watermarked_data["id"][i], "text": sentence_shuffle_attack(watermarked_data["text"][i]), "truth_label": "watermarked", "type": f"sentence-shuffle"})
        # We attack the original non-watermarked text
        rows.append({"id": watermarked_data["id"][i], "text": sentence_shuffle_attack(original_data["text"][i]), "truth_label": "not-watermarked", "type": f"sentence-shuffle"})

    data = pd.DataFrame(rows)

    # attention: the field truth_label is removed from the tira workflow for system inputs, but not for the truth data
    output_directory.mkdir(exist_ok=True, parents=True)
    data.to_json(output_directory / "texts.jsonl", lines=True, orient="records")


if __name__ == '__main__':
    main()

