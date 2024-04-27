# Text Detoxification

The code in this directory is used to evaluate the predictions of the [text detoxification shared task at CLEF 2024](https://pan.webis.de/clef24/pan24-web/text-detoxification.html).

We provide the evaluation code in docker images to simplify the usage, together with baselines that you can try out.

The output format is identical to the input format. Please submit your results in JSON Lines format producing one output line for each input instance.

Each line should have the following format:

```
{"id": "<ID>", "text": "<TEXT>"}
```

where `id` is the id of the instance (must be passed without modification and `text` is the to-be-detoxified text.

# CodaLab baseline submission

If you are participating through [CodaLab platform](https://codalab.lisn.upsaclay.fr/competitions/18243), please refer to `data` folder. There you may find the example submissions.

## Baselines

We provide a set of baselines together with instructions on how you can run them on your machine:

- [baselines/trivial-baseline](baselines/trivial-baseline): A set of trivial baselines that remove all, none, or [specific stopwords](https://huggingface.co/datasets/textdetox/multilingual_toxic_lexicon) from the text.

## Evaluation

To evaluate the performance of the method on a concrete data, run the following:

```shell
./evaluate.py \
	--input=path/to/your/input.jsonl \
	--golden=path/to/your/references.jsonl \ # not available by default
	--prediction=path/to/your/predictions.jsonl
```

Set `--input` as a path to the `input.jsonl`, `--predictions` as a path to `predictions.jsonl`. 

