# Text Detoxification

The code in this directory is used to evaluate the predictions of the [text detoxification shared task at CLEF 2025](https://pan.webis.de/clef25/pan25-web/text-detoxification.html).

We provide code for several baselines.

# CodaLab baseline submission

If you are participating through [CodaLab platform](https://codalab.lisn.upsaclay.fr/competitions/22396), please refer to [data](sample_submissions/) folder to find the submission examples of the duplicated baseline.

## Baselines

We provide a set of baselines:

- [baseline-delete](baselines/baseline_delete/)
- [baseline-backtranslation](baselines/baseline_backtranslation/)
- [baseline-mt0](baselines/baseline_mt0/)

## Evaluation

To evaluate the performance of the method on a concrete data, run the following:

```shell
./evaluate.py \
	--input=path/to/your/input.jsonl \
	--golden=path/to/your/references.jsonl \ # not available by default
	--prediction=path/to/your/predictions.jsonl
```
