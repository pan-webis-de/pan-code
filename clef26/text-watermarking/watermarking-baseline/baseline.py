#!/usr/bin/env python3
from tira.check_format import lines_if_valid
from pathlib import Path
import click
import pandas as pd
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkDetector, WatermarkingConfig
import unicodedata


# Some Helper Functions
def load_data(directory):
    ret = []
    with open(str(directory), mode='r', encoding="utf-8") as infile:
        for line in infile:
            ret.append(json.loads(line))
    return pd.DataFrame.from_records(ret)


def normalize_text(text):
    return unicodedata.normalize("NFKC", text)


def clean_llm_output(decoded_output):
    clean_outputs = []
    for text in decoded_output:
        if "[/INST]" in text:
            clean_outputs.append(text.split("[/INST]", 1)[1].strip())
        else:
            clean_outputs.append(text.strip())
    return clean_outputs


# Dummy Watermark Functions
@click.argument("input_directory", type=Path)
@click.argument("output_directory", type=Path)
def watermark_dummy(input_directory, output_directory):
    data = load_data(input_directory)

    # we just add "_xy123_" as watermark
    data["text"] = data["text"].apply(lambda i: i + " _xy123_")
    output_directory.mkdir(exist_ok=True, parents=True)

    data.to_json(output_directory / "watermarked-text.jsonl", lines=True, orient="records")


@click.argument("input_directory", type=Path)
@click.argument("output_directory", type=Path)
def detect_dummy(input_directory, output_directory):
    data = load_data(input_directory)
    
    # label should be 1.0 if our watermark "_xy123_" is in the text
    data["label"] = data["text"].apply(lambda i: 1.0 if "_xy123_" in i else 0.0)
    del data["text"]
    output_directory.mkdir(exist_ok=True, parents=True)
    data.to_json(output_directory / "detected-text.jsonl", lines=True, orient="records")


# Watermark Functions
@click.argument("input_directory", type=Path)
@click.argument("output_directory", type=Path)
def watermark(
    input_directory, 
    output_directory,
    model_name_or_path='mistralai/Mistral-7B-Instruct-v0.2',#"Qwen/Qwen2.5-1.5B-Instruct",
    greenlist_ratio=0.25,
    bias=2.5,
    hashing_key=15485863,
    seeding_scheme="lefthash",
    **model_args
):
    data = load_data(input_directory)

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", **model_args)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    watermarking_config = WatermarkingConfig(
        greenlist_ratio=greenlist_ratio, 
        bias=bias, 
        hashing_key=hashing_key, 
        seeding_scheme=seeding_scheme
    )

    # Prepare prompts
    prompt = "Paraphrase the following text in clear, formal prose. " \
            "Preserve the meaning, arguments, and structure. " \
            "Do not shorten or lengthen it significantly. " \
            "Do not use bullet points, markdown, or special characters. " \
            "Return exactly one paraphrased version and nothing else.\n\n"
    input_list = ["<s>[INST] "
                f"{prompt}"
                f"{normalize_text(t)} [/INST]"
                for t in data['text']]
    
    decoded_cleaned = []
    for text in tqdm(input_list):
        inputs = tok([text], padding=True, return_tensors="pt").to(device)
        max_new_tokens = int(inputs["input_ids"].shape[1] * 1.2)

        # Generate Watermark
        out_watermarked = model.generate(
            **inputs, 
            watermarking_config=watermarking_config, 
            do_sample=True, 
            top_p=0.9, 
            temperature=0.6, 
            max_new_tokens=max_new_tokens, 
            no_repeat_ngram_size=2, 
            eos_token_id=tok.eos_token_id
        )

        # Decode Watermarked Text
        decoded = tok.batch_decode(
            out_watermarked,
            skip_special_tokens=True
        )
        decoded_cleaned += clean_llm_output(decoded)
    data["text"] = decoded_cleaned

    # Save Output File
    with open(str(output_directory / "watermarked-text.jsonl"), 'w', encoding="utf-8") as outfile:
        for _, entry in data.iterrows():
            json.dump(entry.to_dict(), outfile, ensure_ascii=False)
            outfile.write('\n')


@click.argument("input_directory", type=Path)
@click.argument("output_directory", type=Path)
def detect(
    input_directory, 
    output_directory,
    model_name_or_path='mistralai/Mistral-7B-Instruct-v0.2',
    greenlist_ratio=0.25,
    bias=2.5,
    hashing_key=15485863,
    seeding_scheme="lefthash",
    **model_args
):
    data = load_data(input_directory)
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_args)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    watermarking_config = WatermarkingConfig(greenlist_ratio=greenlist_ratio, bias=bias, hashing_key=hashing_key, seeding_scheme=seeding_scheme)
    detector = WatermarkDetector(
        model_config=model.config, 
        device=device, 
        watermarking_config=watermarking_config
    )

    # Detect Watermarks
    inputs = tok(data["text"].to_list(), padding=True, return_tensors="pt").to(device)
    detection_out_watermarked = detector(inputs["input_ids"], return_dict=True)
    data["label"] = detection_out_watermarked.prediction.astype(int)
    del data["text"]
    
    # Save Output File
    output_directory.mkdir(exist_ok=True, parents=True)
    with open(str(output_directory / "detected-text.jsonl"), 'w', encoding="utf-8") as outfile:
        for _, entry in data.iterrows():
            json.dump(entry.to_dict(), outfile, ensure_ascii=False)
            outfile.write('\n')


@click.group()
def main():
    pass


main.command()(watermark)
main.command()(detect)

if __name__ == '__main__':
    main()

