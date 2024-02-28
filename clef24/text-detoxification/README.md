# Text Detoxification

The code in this directory is used to evaluate the predictions of the [text detoxification shared task at CLEF 2024](https://pan.webis.de/clef24/pan24-web/text-detoxification.html).

We provide the evaluation code in docker images to simplify the usage, together with baselines that you can try out.

The output format is identical to the input format. Please submit your results in JSON Lines format producing one output line for each input instance.

Each line should have the following format:

```
{"id": "<ID>", "text": "<TEXT>"}
```

where `id` is the id of the instance (must be passed without modification and `text` is the to-be-detoxified text.

## Baselines

We provide a set of baselines together with instructions on how you can run them on your machine:

- [baselines/trivial-baseline](baselines/trivial-baseline): A set of trivial baselines that remove all, none, or [specific stopwords](https://huggingface.co/datasets/textdetox/multilingual_toxic_lexicon) from the text.

## Evaluation

Usage:

```shell
./evaluate.py \
	--input=sample/russian/input.jsonl \
	--golden=sample/russian/references.jsonl \
	--prediction=sample/russian/references.jsonl
```

## Docker Images for Evaluation

```shell
make docker-evaluate  # => webis/clef24-text-detoxification-evaluator:0.0.1 .
```

### Build the Docker file

```
docker build -t webis/clef24-text-detoxification-evaluator:0.0.1 .
```
