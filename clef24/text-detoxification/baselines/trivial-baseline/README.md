# Trivial Baseline for the PAN 2024 Text Detoxification Task

This directory contains a set of trivial baselines that remove all, none, or [specific stopwords](https://huggingface.co/datasets/textdetox/multilingual_toxic_lexicon) from the text.

To run the baselines as [TIRA](http://tira.io) would execute them, please install Python >= 3.7, Docker, and tira on your machine (`pip3 install tira`).

## Running The Baselines

A simple baseline that removes toxic words from the text can be executed via:

```
tira-run \
    --input-directory example-input-en \
    --image webis/clef24-text-detoxification-baseline:0.0.1 \
    --command '/trivial_baseline.py --input ${inputDataset}/input.jsonl --output ${outputDir}/predictions.jsonl'
```

The predictions can be found in the directory `tira-output/predictions.jsonl`. You can select the desired language to load toxic words, options are `['am', 'es', 'ru', 'uk', 'en', 'zh', 'ar', 'hi', 'de']`. Without specification, it will load toxic words for all the languages. 

---

A simple baseline that removes all terms:
```
tira-run \
    --input-directory example-input-en \
    --image webis/clef24-text-detoxification-baseline:0.0.1 \
    --command '/trivial_baseline.py --input ${inputDataset}/input.jsonl --output ${outputDir}/predictions.jsonl --remove-all-terms true'
```

The predictions can be found in the directory `tira-output/delete_all_baseline_en.jsonl`

---

A simple baseline that returns the text without modification (on a tiny dataset to ensure everything works):

```
tira-run \
    --input-directory example-input-en \
    --image webis/clef24-text-detoxification-baseline:0.0.1 \
    --command '/trivial_baseline.py --input ${inputDataset}/input.jsonl --output ${outputDir}/predictions.jsonl --remove-no-terms true'
```

The predictions can be found in the directory `tira-output/delete_none_baseline_en.jsonl`


## Development

Build the docker image via:

```
docker build -t webis/clef24-text-detoxification-baseline:0.0.1 .
```

