#!/bin/bash -l
#SBATCH --job-name=embeddings
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --gpus=4
#SBATCH -N 1
#SBATCH --output=log_embeddings_%A_%a.log
#SBATCH --array=0-2

module load 2023a CUDA/12.4.0

# Set paths
BASE_PATH="./pan25-plag_detect_test"
PAIRS_PATH="${BASE_PATH}/03_test/pairs"
SRC_PATH="${BASE_PATH}/src"
SUSP_PATH="${BASE_PATH}/susp"
TRAIN_OR_TEST="test"

# Define array of models to test
model_array=("Linq-AI-Research/Linq-Embed-Mistral" "Alibaba-NLP/gte-Qwen2-7B-instruct" "meta-llama/Llama-3.3-70B-Instruct")

# Get the model for this array job
model=${model_array[$SLURM_ARRAY_TASK_ID]}

echo "Running embeddings with model: $model"
uv run python embeddings.py \
  --model "$model" \
  --base_path "$BASE_PATH" \
  --pairs_path "$PAIRS_PATH" \
  --src_path "$SRC_PATH" \
  --susp_path "$SUSP_PATH" \
  --train_or_test "$TRAIN_OR_TEST"

echo "Embedding generation complete!"
