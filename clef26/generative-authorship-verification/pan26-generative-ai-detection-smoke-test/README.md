---
configs:
- config_name: inputs
  data_files:
  - split: train
    path: ["dataset.jsonl"]
- config_name: truths
  data_files:
  - split: train
    path: ["test-truth.jsonl"]

tira_configs:
  resolve_inputs_to: "."
  resolve_truths_to: "."
  baseline:
    link: https://github.com/pan-webis-de/pan-code/tree/master/clef25/generative-authorship-verification/naive-baseline
    command: /predict.py
    format:
      name: ["*.jsonl"]
      config: {"required_fields":["id","label"],"minimum_lines":3}
  input_format:
    name: "*.jsonl"
    config:
      max_size_mb: 150
      config: {"required_fields":["id","label"],"minimum_lines":3}
  truth_format:
    name: "*.jsonl"
    config: {"required_fields":["id","label"],"minimum_lines":3}
  evaluator:
    image: mam10eks/multi-author-analysis:eval-26
    command: python3 /evaluator.py -p ${inputRun} -t ${inputDataset} -o ${outputDir}
---

# Generative AI Authorship Verification 2026: Spot Check Dataset

This dataset is intended to spot check submissions for Task-2@Pan on Voight-Kampff Generative AI Detection 2026.

Upload this to TIRA via (remove the `--dry-run` argument after a first test):

```
tira-cli dataset-submission --path smoketest --task generative-ai-authorship-verification-panclef-2026 --split train --dry-run
```

If everything works, the result should look like:


