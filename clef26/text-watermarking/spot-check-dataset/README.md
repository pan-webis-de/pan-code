---
configs:
- config_name: inputs
  data_files:
  - split: train
    path: ["train.jsonl"]
- config_name: truths
  data_files:
  - split: train
    path: ["train.jsonl"]

tira_configs:
  resolve_inputs_to: "."
  resolve_truths_to: "."
  baseline:
    link: tbd
    command: tbd
    format:
      name: ["*.jsonl"]
  input_format:
    name: "*.jsonl"
    config:
      max_size_mb: 150
  truth_format:
    name: "*.jsonl"
  evaluator:
    image: tbd
    command: tbd
---

# Text Watermarking 2026: Spot Check Dataset

This dataset is intended to spot check submissions (via the train.jsonl file from [Zenodo](https://zenodo.org/records/18620130)) for the Text Watermarking Task at [PAN 2026](https://pan.webis.de/clef26/pan26-web/text-watermarking.html).

**Attention:** The workflow engine to run the three steps for each submission (1) watermark, (2) obfuscate, (3) detect is not yet in the main branch of tira and not yet on pypi. The following describes how it works at the moment via the dev verison of TIRA, and as soon as this is finalized, we remove this note.

Upload this to TIRA via (remove the `--dry-run` argument after a first test):

```
tira-cli dataset-submission --path spot-check-dataset --task pan26-text-watermarking --split train --dry-run
```

If everything works, the result should look like:

<img width="1341" height="205" alt="Screenshot_20260227_145626" src="https://github.com/user-attachments/assets/65b8cbb6-6977-4ae8-82f0-331ceccd57dc" />

