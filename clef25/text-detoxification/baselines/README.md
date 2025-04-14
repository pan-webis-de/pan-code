# How to run baselines

## Requirements
Works for:
- `python 3.11.*`
- `torch==2.1.1+cu121`

## Backtranslation
- `pip install -r baseline_backtranslation/requirements.txt`
- `python baseline_backtranslation/main.py --input_path data/dev_inputs.tsv --output_path results/output.tsv  --device cuda`


## mT0
- `pip install -r baseline_mt0/requirements.txt`
- `python baseline_mt0/main.py --input_path data/dev_inputs.tsv --output_path results/output.tsv --batch_size 128`

## Delete (stop words)
- `pip install -r baseline_delete/requirements.txt`
- `python baseline_delete/main.py --input_path data/dev_inputs.tsv --output_path results/output.tsv --lexicon None`

## How to fine-tune your seq2seq model

We also provide a quick starter norebook with the example on how to fine-tune your own text detoxification *seq2seq* generation model!
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Wd_32qGpED5M3cfmDapKqOGplgLQ39xP?usp=sharing)
