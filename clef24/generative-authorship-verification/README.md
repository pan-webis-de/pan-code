# Code for PAN'24 Generative AI Authorship Verification

Baselines and evaluator for the PAN'24 "Voight-Kampff" Generative AI Authorship Verification task.

## Run Baselines

We provide five (six) LLM detection baselines as reimplementations from the original papers:

- PPMd Compression-based Cosine [[Sculley and Brodley, 2006](https://ieeexplore.ieee.org/abstract/document/1607268)] [[Halvani et al., 2017](https://dl.acm.org/doi/abs/10.1145/3098954.3104050)]
- Authorship Unmasking [[Koppel and Schler, 2004](https://dl.acm.org/doi/abs/10.1145/1015330.1015448)] [[Bevendorff et al., 2019](https://aclanthology.org/N19-1068/)]
- Binoculars [[Hans et al., 2024](https://arxiv.org/abs/2401.12070)]
- DetectLLM LRR and NPR [[Su et al., 2023](https://arxiv.org/abs/2306.05540)]
- DetectGPT [[Mitchell et al., 2023](https://arxiv.org/abs/2301.11305)]
- Text length

With PPMd CBC and Authorship unmasking, we provide two bag-of-words authorship verification models. Binoculars, DetectLLM, and DetectGPT use large language models to measure text perplexity. The text length baseline serves mainly as a data sanity check and is designed to have random performance.

You can run the baselines locally, in a Docker container or using `tira-run`. All baselines come with a CLI and usage instructions. The general usage for any baseline is:

```console
$ baseline BASELINENAME [OPTIONS] INPUT_FILE OUTPUT_DIRECTORY
```

Use `--help` on any subcommand for more information:

```console
$ baseline --help
Usage: baseline [OPTIONS] COMMAND [ARGS]...

  PAN'24 Generative Authorship Detection baselines.

Options:
  --help  Show this message and exit.

Commands:
  binoculars  PAN'24 baseline: Binoculars.
  detectgpt   PAN'24 baseline: DetectGPT.
  detectllm   PAN'24 baseline: DetectLLM.
  length      PAN'24 baseline: Text length.
  ppmd        PAN'24 baseline: Compression-based cosine.
  unmasking   PAN'24 baseline: Authorship unmasking.
```

### Run Baselines Locally

Install requirements first, then run baseline. Use `--help` for more information.

```console
$ python3 -m venv venv && source venv/bin/activate
$ pip install -r requirements.baselines.txt
$ python3 pan24_llm_baselines/baseline.py BASELINENAME [OPTIONS] dataset.jsonl out
```

### Run in Docker

Running baselines in a Docker container requires you to mount input and output directories.

Replace `BASELINENAME` with a baseline (e.g., `binoculars`). The `baseline` prefix command is part of the container's entrypoint and must be skipped. Use `--help` for more information.

```console
$ docker run --rm \
    -v /path/to/dataset.jsonl:/dataset.jsonl \
    -v /path/to/output:/out \
    --gpus=all
    ghcr.io/pan-webis-de/pan24-generative-authorship-baselines:latest \
    BASELINENAME [OPTIONS] /dataset.jsonl /out
```

`--gpus=all` requires the [Nvidia Container Toolkit to be installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). The PPMd, Unmasking, and Text length baselines can be run without this parameter, since they don't require GPU inference.

### Run with tira-run

`tira-run` is the closest you can get to test locally how a software will be run on Tira. Replace `BASELINENAME` with a baseline (e.g., `binoculars`). Use `--help` for more information.

```console
$ tira-run \
    --gpus all \
    --image ghcr.io/pan-webis-de/pan24-generative-authorship-baselines:latest \
    --input-directory path/to/input/dir \
    --output-directory path/to/output/dir \
    --command 'baseline BASELINENAME $inputDataset/dataset.jsonl $outputDir'
```

Instead of `--input-dir`, you can also specify an online Tira dataset with `--input-dataset` (e.g. `--input-dataset generative-ai-authorship-verification-panclef-2024/pan24-generative-authorship-smoke-20240411_0-training`). The can find the correct name of a dataset in the Tira web UI after creating a new software submission for it.


## Run Evaluator

To run the evaluator locally, install the requirements first:

```console
$ python3 -m venv venv && source venv/bin/activate
$ pip install -r requirements.evaluator.txt
$ python3 evaluator/evaluator.py <predictions-file> <truth-file> <output-dir> [--outfile-name <name>]
```

You can also run the evaluator in a Docker container (requires you to mount input and output directories):

```console
$ docker run --rm \
    -v /path/to/answers.jsonl:/answers.jsonl \
    -v /path/to/truth.jsonl:/truth.jsonl \
    -v /path/to/output:/out \
    ghcr.io/pan-webis-de/pan24-generative-authorship-evaluator:latest \
    evaluator /answers.jsonl /truth.jsonl /out
```
