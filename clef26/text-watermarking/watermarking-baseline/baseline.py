#!/usr/bin/env python3
from tira.check_format import lines_if_valid
from pathlib import Path
import click
import pandas as pd
import random
import os
import numpy as np
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkDetector, WatermarkingConfig
import unicodedata


# Some Helper Functions
def load_data(directory):
    ret = lines_if_valid(directory, "*.jsonl")
    return pd.DataFrame.from_records(ret)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def normalize_text(text):
    return unicodedata.normalize("NFKC", text)


def build_prompt(tok, text):
    system_msg = "You are a text editor. You only output the final paraphrased text. No explanations, no extra words. The output must be in English."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": (
            "Paraphrase the following text in clear, formal prose. "
            "Preserve meaning and structure. "
            "Output only the rewritten text in English. \n\n"
            + normalize_text(text)
        )}
    ]
    
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


# Watermark Functions
@click.argument("input_directory", type=Path)
@click.argument("output_directory", type=Path)
def watermark(
    input_directory, 
    output_directory,
    model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    greenlist_ratio=0.5,
    bias=2.5,
    hashing_key=15485863,
    seeding_scheme="lefthash",
    random_seed=42,
    **model_args
):
    seed_everything(random_seed)
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
    input_list = [build_prompt(tok, t) for t in data["text"]]
    
    decoded_list = []
    batch_size = 1
    for i in tqdm(range(0, len(input_list), batch_size)):
        batch = input_list[i:i+batch_size]
        inputs = tok(batch, padding=True, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]
        max_new_tokens = int(input_len)

        # Generate Watermark
        out_watermarked = model.generate(
            **inputs, 
            watermarking_config=watermarking_config, 
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            max_new_tokens=max_new_tokens, 
            no_repeat_ngram_size=3, 
            eos_token_id=tok.eos_token_id
        )

        generated_tokens = out_watermarked[:, input_len:]

        # Decode Watermarked Text
        decoded = tok.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )
        decoded_list += decoded
    data["text"] = decoded_list

    # Save Output File
    output_directory.mkdir(exist_ok=True, parents=True)
    with open(str(output_directory / "watermarked-text.jsonl"), 'w', encoding="utf-8") as outfile:
        for _, entry in data.iterrows():
            json.dump(entry.to_dict(), outfile, ensure_ascii=False)
            outfile.write('\n')


@click.argument("input_directory", type=Path)
@click.argument("output_directory", type=Path)
def detect(
    input_directory, 
    output_directory,
    model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    greenlist_ratio=0.5,
    bias=2.5,
    hashing_key=15485863,
    seeding_scheme="lefthash",
    random_seed=42,
    **model_args
):
    seed_everything(random_seed)
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
    results = []
    for text in data["text"]:
        inputs = tok(text, return_tensors="pt").to(device)
        detection_out_watermarked = detector(inputs["input_ids"], return_dict=True)
        results.extend(detection_out_watermarked.prediction.astype(int).tolist())
    data["label"] = results
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

