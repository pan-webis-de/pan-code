# Pan@CLEF25 Generative Authorship Verification

This repository contains code for the PAN@CLEF25 Generative Authorship Verification task. It implements a baseline approach using embeddings from large language models (LLMs) and cosine similarity to detect plagiarism.

## Setup

1. Clone this repository:
```bash
git clone https://github.com/jpwahle/pan_clef25_generative-authorship-verification.git
cd pan_clef25_generative-authorship-verification
```

2. Create and activate virtual environment with `uv`:
```bash
uv venv
source .venv/bin/activate
uv sync
```

## Using the Code

### 1. Generating Embeddings

> If you want to skip this step, you can download the precomputed embeddings from AWS S3 for [Llama](https://aws-static-webhost-9.s3.us-east-2.amazonaws.com/pan25/test_embeddings_meta-llama_Llama-3.3-70B-Instruct.tar.gz), [Mistral](https://aws-static-webhost-9.s3.us-east-2.amazonaws.com/pan25/test_embeddings_Linq-AI-Research_Linq-Embed-Mistral.tar.gz), and [Qwen](https://aws-static-webhost-9.s3.us-east-2.amazonaws.com/pan25/test_embeddings_Alibaba-NLP_gte-Qwen2-7B-instruct.tar.gz).

The first step is to generate embeddings for all documents using a large language model. You can use the `embeddings.py` script for example for the train directory `train_data`:

```bash
uv run python embeddings.py \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --base_path "./train_data" \
  --pairs_path "./train_data/03_train/pairs" \
  --src_path "./train_data/src" \
  --susp_path "./train_data/susp" \
  --train_or_test "train"
```

For test data, you can use a similar approach:

```bash
uv run python embeddings.py \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --base_path "./test_data" \
  --pairs_path "./test_data/03_test/pairs" \
  --src_path "./test_data/src" \   
  --susp_path "./test_data/susp" \
  --train_or_test "test"
```

This will generate embeddings for each document and save them in a directory named `train_embeddings_meta-llama_Llama-3.3-70B-Instruct` or `test_embeddings_meta-llama_Llama-3.3-70B-Instruct` respectively.

Alternatively, you can use one of the following models:
- `Linq-AI-Research/Linq-Embed-Mistral`
- `Alibaba-NLP/gte-Qwen2-7B-instruct`

For convenience, you can also use the provided bash script on SLURM:
```bash
sbatch embeddings.sh
```

### 2. Running the Cosine Similarity Baseline

#### Computing Thresholds from Training Data

To compute optimal thresholds from training data and apply them to test data:

```bash
uv run python cosine_baseline.py \
  --test_embeddings test_data/test_embeddings_meta-llama_Llama-3.3-70B-Instruct \
  --test_pairs_file test_data/03_test/pairs \
  --test_truths_path test_data/03_test_truth \
  --train_embeddings train_data/train_embeddings_meta-llama_Llama-3.3-70B-Instruct \
  --train_pairs_file train_data/03_train/pairs \
  --train_truths_path train_data/03_train_truth
```

This will:
1. Load the training data embeddings
2. Analyze plagiarism and non-plagiarism similarities in the training data
3. Compute the optimal threshold that best separates plagiarism from non-plagiarism
4. Apply this threshold to the test data
5. Generate detections in XML format in the `detections_cosine_meta-llama_Llama-3.3-70B-Instruct` directory

#### Using Pre-computed Thresholds

If you want to skip the threshold computation step and use the pre-computed thresholds, you can simply run:

```bash
uv run python cosine_baseline.py \
  --test_embeddings test_data/test_embeddings_meta-llama_Llama-3.3-70B-Instruct \
  --test_pairs_file test_data/03_test/pairs \
  --test_truths_path test_data/03_test_truth
```

The script will automatically use the following pre-computed thresholds based on the embedding model:
- `meta-llama/Llama-3.3-70B-Instruct`: 0.727
- `Linq-AI-Research/Linq-Embed-Mistral`: 0.697
- `Alibaba-NLP/gte-Qwen2-7B-instruct`: 0.596

These thresholds were optimized on the training data to maximize the separation between plagiarism and non-plagiarism pairs.

You can also use the provided bash script:
```bash
sbatch cosine_baseline.sh
```

### 3. Evaluating the Results

To evaluate the performance of the detection algorithm, use the `eval.py` script:

```bash
uv run python eval.py \
  -p test_data/03_test_truth \
  -d test_data/detections_cosine_meta-llama_Llama-3.3-70B-Instruct \
  --plag-tag=plagiarism \
  --det-tag=plagiarism
```

This will compute and display the following metrics:
- Recall
- Precision
- Granularity
- PlagDet Score (combined measure)

You can evaluate detections from multiple models at once using the provided bash script:
```bash
sbatch eval.sh
```
