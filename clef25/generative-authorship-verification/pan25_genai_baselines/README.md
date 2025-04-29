# PAN'25 "Voight-Kampff" Generative AI Authorship Verification Baselines

LLM detection baselines for the "Voight-Kampff" Generative AI Authorship Verification Baselines at PAN 2025.

## Installation

The baselines can be installed locally in a venv using `pip`:

```console
python3 -m venv venv
source venv/bin/activate
pip install .
```

Alternatively, you can run them in a Docker image (see below).

## Usage

If installed via ``pip``, you can just run the baselines with

```console
pan25-baseline BASELINENAME INPUT_FILE OUTPUT_DIRECTORY
```

Use `--help` on any subcommand for more information:

```console
pan25-baseline --help
Usage: pan25-baseline [OPTIONS] COMMAND [ARGS]...

  PAN'25 Generative Authorship Detection baselines.

Options:
  --help  Show this message and exit.

Commands:
  binoculars  PAN'25 baseline: Binoculars.
  ppmd        PAN'25 baseline: Compression-based cosine.
  tfidf       PAN'25 baseline: TF-IDF SVM.
```

If you want to run the baselines via Docker, use:

```console
docker run --rm --gpus=all -v INPUT_FILE:/input.jsonl -v OUTPUT_DIRECTORY:/out \
    ghcr.io/pan-webis-de/pan25-generative-authorship-baselines \
    BASELINENAME /input.jsonl /out
```

`INPUT_FILE` is the test / validation input data (JSONL format). `OUTPUT_DIRECTORY` is the output
directory for the predictions.

Concrete example:

```console
docker run --rm --gpus=all -v $(pwd)/val.jsonl:/val.jsonl -v $(pwd):/out \
    ghcr.io/pan-webis-de/pan25-generative-authorship-baselines \
    tfidf /input.jsonl /out
```

The option ``--gpus=all`` is needed only for Binoculars.
