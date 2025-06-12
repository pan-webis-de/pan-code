#!/bin/bash -l
#SBATCH --job-name=log_eval_%j.log
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --output=log_eval_%j.log

# For the random baseline
uv run python eval.py -p pan25-plag_detect_test/03_test_truth -d pan25-plag_detect_test/detections_random --plag-tag=plagiarism --det-tag=plagiarism > results/random_results.txt

# For all embedding models
embedding_model_array=("embeddings_meta-llama_Llama-3.3-70B-Instruct" "embeddings_Linq-AI-Research_Linq-Embed-Mistral" "embeddings_Alibaba-NLP_gte-Qwen2-7B-instruct")

# Run evaluation for each model
for model in "${embedding_model_array[@]}"; do
  # Extract model name from path
  model_name=$(echo "$model" | cut -d'/' -f2) 
  echo "Evaluating $model_name..."
  uv run python eval.py -p pan25-plag_detect_test/03_test_truth -d "pan25-plag_detect_test/detections_cosine_${model_name}" --plag-tag=plagiarism --det-tag=plagiarism > "results/cosine_embeddings_${model_name}_results.txt"
done