---
configs:
- config_name: inputs
  data_files:
  - split: train
    path: ["corpus.jsonl.gz", "queries.jsonl"]
- config_name: truths
  data_files:
  - split: train
    path: ["qrels.txt", "queries.jsonl"]

tira_configs:
  resolve_inputs_to: "."
  resolve_truths_to: "."
  baseline:
    link: https://github.com/pan-webis-de/pan-code/tree/master/clef26/generated-plagiarism-detection/baseline-pyterrier
    command: /baseline.py --dataset $inputDataset --output $outputDir --index /tmp/my-index/
    format:
      name: ["run.txt", "lightning-ir-document-embeddings", "lightning-ir-query-embeddings"]
  input_format:
    name: "lsr-benchmark-inputs"
    config:
      max_size_mb: 150
  truth_format:
    name: "qrels.txt"
  evaluator:
    measures: ["nDCG@10", "RR"]
---

# Generative Plagiarism Detection 2026: Spot Check Dataset

This dataset is intended to spot check submissions for Task-4@Pan on Generative Plagiarism Detection.

Upload this to TIRA via (remove the `--dry-run` argument after a first test):

```
tira-cli dataset-submission --path spot-check-dataset --task pan26-generated-plagiarism-detection --split train --dry-run
```
