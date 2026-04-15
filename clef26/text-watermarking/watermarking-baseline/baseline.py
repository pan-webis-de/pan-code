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


def build_prompt(tok, text):
    system_msg = "You are a text editor. You only output the final paraphrased text. No explanations, no extra words."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": (
            "Paraphrase the following text in clear, formal prose. "
            "Preserve meaning and structure. "
            "Output only the rewritten text. \n\n"
            + normalize_text(text)
        )}
    ]
    
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


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
    model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    greenlist_ratio=0.5,
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
            do_sample=False, 
            max_new_tokens=max_new_tokens, 
            no_repeat_ngram_size=2, 
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
    results = []
    for text in data["text"]:
        inputs = tok(text, return_tensors="pt").to(device)
        detection_out_watermarked = detector(inputs["input_ids"], return_dict=True)
        results.extend(detection_out_watermarked.prediction.tolist())
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

