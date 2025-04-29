# PAN'25 "Voight-Kampff" Generative AI Authorship Verification Evaluator

System evaluator for the "Voight-Kampff" Generative AI Authorship Verification Baselines at PAN 2025.

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
pan25-evaluator ANSWER_FILE TRUTH_FILE OUTPUT_DIR
```

Use `--help` on any subcommand for more information:

```console
$ pan25-evaluator --help                                                                                                                                                          USAGE   master  2✔  2✎  ⎈ webis  ceph 
Usage: pan25-evaluator [OPTIONS] ANSWER_FILE TRUTH_FILE OUTPUT_DIR

  PAN'25 Generative AI Authorship Verification evaluator.

Options:
  -o, --outfile-name TEXT  Output JSON filename  [default: evaluation.json]
  -p, --skip-prototext     Skip Tira Prototext output, only write JSON
  -s, --skip-source-eval   Skip evaluation of individual sources
  --optimize-score         Optimize score by finding optimal operating point
  --help                   Show this message and exit.

```

If you want to run the baselines via Docker, use:

```
docker run --rm -v RUN_FILE:/run.jsonl -v TRUTH_FILE:/truth.jsonl -v OUTPUT_DIRECTORY:/out \
    ghcr.io/pan-webis-de/pan25-generative-authorship-evaluator \
    /run.jsonl /truth.jsonl /out
```

`RUN_FILE` is the system's output file (JSONL format). `TRUTH_FILE` is the matching ground truth file.
`OUTPUT_DIRECTORY` is where the evaluator JSONL and Protobuf output will be written.

Concrete example:

```
docker run --rm -v $(pwd)/run.jsonl:/run.jsonl -v $(pwd)/truth.jsonl:/truth.jsonl -v $(pwd):/out \
    ghcr.io/pan-webis-de/pan25-generative-authorship-evaluator \
    /run.jsonl /truth.jsonl /out
```
