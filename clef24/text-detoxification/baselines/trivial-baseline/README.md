# Trivial Baseline for the PAN 2024 Text Detoxification Task

This directory contains a set of trivial baselines that remove all, none, or [specific stopwords](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) from the text.

To run the baselines as [TIRA](http://tira.io) would execute them, please install Python >= 3.7, Docker, and tira on your machine (`pip3 install tira`).

## Running The Baselines

A simple baseline that removes stopwords from the text can be executed via (on a tiny dataset to ensure everything works):

```
tira-run \
	--input-dataset pan23-text-detoxification/english-tiny-20231112-training \
	--image webis/clef24-text-detoxification-baseline:0.0.1 \
	--command '/trivial-baseline.py --input ${inputDataset}/input.jsonl --output ${outputDir}/references.jsonl --stopword-directory /lodnoobw'
```

The predictions can be found in the directory `tira-output/references.jsonl`

---

A simple baseline that removes all terms (on a tiny dataset to ensure everything works):
```
tira-run \
	--input-dataset pan23-text-detoxification/english-tiny-20231112-training \
	--image webis/clef24-text-detoxification-baseline:0.0.1 \
	--command '/trivial-baseline.py --input ${inputDataset}/input.jsonl --output ${outputDir}/references.jsonl --remove-all-terms true'
```

The predictions can be found in the directory `tira-output/references.jsonl`

---

A simple baseline that returns the text without modification (on a tiny dataset to ensure everything works):

```
tira-run \
	--input-dataset pan23-text-detoxification/english-tiny-20231112-training \
	--image webis/clef24-text-detoxification-baseline:0.0.1 \
	--command '/trivial-baseline.py --input ${inputDataset}/input.jsonl --output ${outputDir}/references.jsonl --remove-no-terms true'
```

The predictions can be found in the directory `tira-output/references.jsonl`


## Development

Build the docker image via:

```
docker build -t webis/clef24-text-detoxification-baseline:0.0.1 .
```

