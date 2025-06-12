import os
import numpy as np
from vllm import LLM, SamplingParams
import re
from tqdm import tqdm
import time
import json
import argparse
from typing import List, Union

def get_gpu_count():
    """Get the number of GPUs allocated by SLURM."""
    gpus = os.getenv('SLURM_GPUS')
    if gpus is not None:
        return int(gpus)
    return 1  # Default to 1 if not in SLURM environment

def paragraph_chunking(text):
    """Split text into paragraphs, keeping display math with surrounding text, and discard references."""

    # Remove references section
    references_pattern = r"(?si)(?:\n\n+|^)(?:references|bibliography|reference list|works cited)(?:\n\n+.*)?$"
    text = re.sub(references_pattern, "", text.strip())

    # Remove references section
    paragraphs = re.split(r"\n\n(?!\s\n\s)", text)

    # Filter out empty paragraphs
    paragraphs = [p for p in paragraphs if p.strip()]

    return paragraphs


def compute_embeddings_with_llm(paragraphs: List[str], model: Union[str, LLM]) -> np.ndarray:
    """
    Compute embeddings using either a local LLM model or an API-based model.
    """
    embeddings = []
    
    print(f"Computing embeddings for {len(paragraphs)} paragraphs...")

    outputs = model.embed(paragraphs)
    for output in outputs:
        embeddings.append(output.outputs.embedding)

    # Normalize embeddings
    print("Embedding computation complete.")
    return np.array(embeddings)


def compute_and_save_embeddings(doc_list, doc_path, embeddings_path, llm_model):
    """
    Compute embeddings for all documents and save to disk.
    """
    print(f"Computing embeddings for {len(doc_list)} documents...")

    skipped_docs = []

    for doc in tqdm(doc_list, desc="Embedding documents"):
        doc_path_full = os.path.join(doc_path, doc)
        emb_file = os.path.join(embeddings_path, f"{doc}.npy")

        if os.path.exists(emb_file):
            print(f"Skipping {doc} because embeddings already exist")
            continue

        with open(doc_path_full, "r", encoding="utf-8") as f:
            text = f.read()
        paragraphs = paragraph_chunking(text)
        try:
            # Generate embeddings
            embeddings = compute_embeddings_with_llm(paragraphs, llm_model)
            np.save(emb_file, embeddings)
            print(f"Saved embeddings for {doc} to {emb_file}")
        except Exception as e:
            # Skip this document
            print(f"Error embedding {doc}: {e}, skipping document because of error...")
            skipped_docs.append(doc)
            continue

    print(f"Skipped {len(skipped_docs)} documents: {skipped_docs}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate embeddings for documents using LLM')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.3-70B-Instruct', 
                       help='LLM model to use for generating embeddings.')
    parser.add_argument('--pairs_path', type=str, default='./pan25-plag_detect_test/03_test/pairs',
                       help='Path to the pairs file.')
    parser.add_argument('--src_path', type=str, default='./pan25-plag_detect_test/src',
                       help='Path to the source documents directory.')
    parser.add_argument('--susp_path', type=str, default='./pan25-plag_detect_test/susp',
                       help='Path to the suspicious documents directory.')
    parser.add_argument('--base_path', type=str, default='./pan25-plag_detect_test',
                       help='Base path for the dataset.')
    parser.add_argument('--train_or_test', type=str, default='test',
                    help='Whether to use the train or test set.')
    args = parser.parse_args()


    # File paths
    pairs_file = args.pairs_path
    src_path = args.src_path
    susp_path = args.susp_path
    embeddings_path = os.path.join(
        args.base_path, args.train_or_test + "_embeddings_" + args.model.replace("/", "_")
    )
    os.makedirs(embeddings_path, exist_ok=True)

    # Load pairs
    try:
        with open(pairs_file, "r", encoding="utf-8") as f:
            all_pairs = [line.strip().split() for line in f.readlines() if line.strip()]
            all_pairs = [p for p in all_pairs if len(p) == 2]
        print(f"Loaded {len(all_pairs)} pairs from {pairs_file}")
        if not all_pairs:
            print("Error: No valid pairs found. Exiting.")
            return
    except FileNotFoundError:
        print(f"Error: Pairs file not found at {pairs_file}. Exiting.")
        return
    except Exception as e:
        print(f"Error reading pairs file {pairs_file}: {e}. Exiting.")
        return

    # Collect unique documents
    susp_docs = sorted(set(pair[0] for pair in all_pairs))
    src_docs = sorted(set(pair[1] for pair in all_pairs))
    print(f"Unique suspicious docs: {len(susp_docs)}")
    print(f"Unique source docs: {len(src_docs)}")

    # Compute and save embeddings for all documents
    start_embedding_time = time.time()
    print("\n--- Computing and Saving Embeddings ---")
    
    # Initialize model if it's a local model
    print(f"\n--- Initializing vLLM for embeddings ---")
    gpu_count = get_gpu_count()
    print(f"Using {gpu_count} GPUs for tensor parallelism")
    llm_model = LLM(model=args.model, tensor_parallel_size=gpu_count, task="embed", max_seq_len_to_capture=32768, enforce_eager=True)
    print(f"vLLM initialized with model: {args.model}")

    compute_and_save_embeddings(susp_docs, susp_path, embeddings_path, llm_model)
    compute_and_save_embeddings(src_docs, src_path, embeddings_path, llm_model)

    end_embedding_time = time.time()
    print(
        f"Embedding computation completed in {end_embedding_time - start_embedding_time:.2f} seconds"
    )

    # Save summary to a JSON file
    summary = {
        "total_susp_docs": len(susp_docs),
        "total_src_docs": len(src_docs),
        "total_time_seconds": end_embedding_time - start_embedding_time,
        "embedding_model": args.model,
        "embeddings_directory": embeddings_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_file = os.path.join(embeddings_path, "embedding_summary.json")
    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        print(f"Saved embedding summary to {summary_file}")
    except Exception as e:
        print(f"Error saving summary to {summary_file}: {e}")

    print("Embedding computation complete.")


if __name__ == "__main__":
    main()
