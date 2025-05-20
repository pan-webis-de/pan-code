# PAN'25 "Voight-Kampff" Generative AI Authorship Verification Baselines

LLM detection baselines for the "Voight-Kampff" Generative AI Authorship Verification Baselines at PAN 2025.

## Installation

The baselines can be installed locally in a venv using `pip`:

```
python3 -m venv venv
source venv/bin/activate
pip install .
```

Alternatively, you can run them in a Docker image (see below).

## Usage

If installed via ``pip``, you can just run the baselines with

```
pan25-baseline BASELINENAME INPUT_FILE OUTPUT_DIRECTORY
```

Use `--help` on any subcommand for more information:

```console
$ pan25-baseline --help
Usage: pan25-baseline [OPTIONS] COMMAND [ARGS]...

  PAN'25 Generative AI Authorship Verification baselines.

Options:
  --help  Show this message and exit.

Commands:
  binoculars  PAN'25 baseline: Binoculars.
  ppmd        PAN'25 baseline: Compression-based cosine.
  tfidf       PAN'25 baseline: TF-IDF SVM.
```

If you want to run the baselines via Docker, use:

```
docker run --rm --gpus=all -v INPUT_FILE:/val.jsonl -v OUTPUT_DIRECTORY:/out \
    ghcr.io/pan-webis-de/pan25-generative-authorship-baselines \
    BASELINENAME /val.jsonl /out
```

`INPUT_FILE` is the test / validation input data (JSONL format). `OUTPUT_DIRECTORY` is the output
directory for the predictions.

Concrete example:

```
docker run --rm --gpus=all -v $(pwd)/val.jsonl:/val.jsonl -v $(pwd):/out \
    ghcr.io/pan-webis-de/pan25-generative-authorship-baselines \
    tfidf /val.jsonl /out
```

The option ``--gpus=all`` is needed only for Binoculars.

## Submit to TIRA

First, please ensure that your have a valid tira client installed via:

```
tira-cli verify-installation
```

First, please test that your approach works on the smoke-test dataset as expected (more details are available in the [documentation](https://docs.tira.io/participants/participate.html#submitting-your-submission)):

```
tira-cli code-submission --dry-run --path . --task generative-ai-authorship-verification-panclef-2025 --dataset pan25-generative-ai-detection-smoke-test-20250428-training --command '/usr/local/bin/pan25-baseline tfidf $inputDataset/dataset.jsonl $outputDir'
```

If this works as expected, you can omit the `--dry-run` argument to submit this baseline to TIRA, please run:

```
tira-cli code-submission --path . --task generative-ai-authorship-verification-panclef-2025 --dataset pan25-generative-ai-detection-smoke-test-20250428-training --command '/usr/local/bin/pan25-baseline tfidf $inputDataset/dataset.jsonl $outputDir'
```


## Submit LLM Approaches to TIRA

As soon as it is verified that a software works (you can use tiny-llama to do that), you can submit an submission (here an example for Llama-3.1-8B and Llama-3.1-8B-Instruct via:

```
tira-cli code-submission --mount-hf-model meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B --path . --task generative-ai-authorship-verification-panclef-2025 --dataset pan25-generative-ai-detection-smoke-test-20250428-training --command '/usr/local/bin/pan25-baseline binoculars --observer meta-llama/Llama-3.1-8B --performer meta-llama/Llama-3.1-8B-Instruct $inputDataset/dataset.jsonl $outputDir'
```

