# Naive Baseline for PAN'25 Voight-Kampff Generative AI Detection

This directory contains a naive baseline for [Subtask 1 of the Voight-Kampff Generative AI Detection 2025 task](https://pan.webis.de/clef25/pan25-web/generated-content-analysis.html) that always predicts that it is undecidable if a text was human-authored or machine-written (i.e., a score of 0.5).

## Development

To generate responses from a dataset, run:

```
./predict.py --dataset generative-ai-authorship-verification-panclef-2025/pan25-generative-ai-detection-smoke-test-20250428-training --output predictions.jsonl
```

The `--dataset` either must point to a local directory or must be the ID of a dataset in TIRA ([tira.io/datasets?query=generative-ai-authorship-verification-panclef-2025](https://archive.tira.io/datasets?query=generative-ai-authorship-verification-panclef-2025) shows an overview of available datasets).

## Submit to TIRA

First, please ensure that your have a valid tira client installed via:

```
tira-cli verify-installation
```

First, please test that your approach works on the smoke-test dataset as expected (more details are available in the [documentation](https://docs.tira.io/participants/participate.html#submitting-your-submission)):

```
tira-cli code-submission --dry-run --path . --task generative-ai-authorship-verification-panclef-2025 --dataset pan25-generative-ai-detection-smoke-test-20250428-training --command '/predict.py'
```

If this works as expected, you can omit the `--dry-run` argument to submit this baseline to TIRA, please run:

```
tira-cli code-submission --path . --task generative-ai-authorship-verification-panclef-2025 --dataset pan25-generative-ai-detection-smoke-test-20250428-training --command '/predict.py'
```
