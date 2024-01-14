# Backtranslation Baseline for the PAN 2024 Text Detoxification Task

This directory contains a backtranslation baseline that performs text detoxification using a sequence of translation, detoxification, and backtranslation processes.

To run the baseline, ensure you have Python >= 3.7, Docker, and tira installed on your machine (`pip3 install tira`).

## Running The Baseline

Run the backtranslation baseline on a tiny dataset:

```bash
tira-run \
    --input-dataset pan23-text-detoxification/english-tiny-20231112-training \
    --image webis/clef24-text-detoxification-baseline-backtranslation:0.0.1 \
    --command '/backtranslation_baseline.py --input ${inputDataset}/input.jsonl --output ${outputDir}/references.jsonl --src_lang_id en'
```

## Development

Build the docker image using:

```bash
docker build -t webis/clef24-text-detoxification-baseline-backtranslation:0.0.1 .
```
