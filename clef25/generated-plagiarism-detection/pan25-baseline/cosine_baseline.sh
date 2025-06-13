#!/bin/bash -l
#SBATCH --job-name=cosine
#SBATCH --time=4:00:00
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --output=log_cosine_baseline_%A_%a.log
#SBATCH --array=0-2

module load 2023a

# Define array of models to test
model_array=("Linq-AI-Research/Linq-Embed-Mistral" "Alibaba-NLP/gte-Qwen2-7B-instruct" "meta-llama/Llama-3.3-70B-Instruct")

# Get the model for this array job
model=${model_array[$SLURM_ARRAY_TASK_ID]}

test_pairs_file=pan25-plag_detect_test/03_test/pairs
test_truths_path=pan25-plag_detect_test/03_test_truth
test_embeddings=pan25-plag_detect_test/test_embeddings_${model//\//_}

echo "Running cosine baseline with embeddings from model: $model"
uv run python cosine_baseline.py --test_embeddings $test_embeddings --test_pairs_file $test_pairs_file --test_truths_path $test_truths_path
