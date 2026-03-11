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
  baseline:
    link: https://github.com/pan-webis-de/pan-code/tree/master/clef26/text-watermarking/watermarking-baseline
    workflow_configuration:
      watermark_command: '/baseline.py watermark $inputDataset $outputDir'
      detect_command: '/baseline.py detect $inputDataset $outputDir'
    format:
      name: ["pan-text-watermarking"]
  workflow:
    name: pan26-text-watermarking
    obfuscation_image: mam10eks/pan-watermarking-prototype:obfuscator-0.0.1
    obfuscation_command: '/obfuscate.py $inputDataset/01-watermarking/*.jsonl $inputDataset/original/*.jsonl $outputDir'
  input_format:
    name: "*.jsonl"
  truth_format:
    name: "*.jsonl"
  evaluator:
    image: mam10eks/pan-watermarking-prototype:eval-0.0.1
    command: '/evaluator.py --output-directory $outputDir $inputRun/01-watermarking/*.jsonl $inputDataset/*.jsonl $inputRun/02-obfuscation/*.jsonl $inputRun/03-detection/*.jsonl'
  resolve_inputs_to: "."
  resolve_truths_to: "."
---

# Text Watermarking 2026: Spot Check Dataset

This dataset is intended to spot check submissions (via the train.jsonl file from [Zenodo](https://zenodo.org/records/18620130)) for the Text Watermarking Task at [PAN 2026](https://pan.webis.de/clef26/pan26-web/text-watermarking.html).

**Attention:** The workflow engine to run the three steps for each submission (1) watermark, (2) obfuscate, (3) detect is not yet in the main branch of tira and not yet on pypi. The following describes how it works at the moment via the dev verison of TIRA, and as soon as this is finalized, we remove this note.

Upload this to TIRA via (remove the `--dry-run` argument after a first test):

```
tira-cli dataset-submission --path spot-check-dataset --task text-watermarking-panclef-2026 --split train --dry-run
```

If everything works, the result should look like:

<img width="1898" height="260" alt="Screenshot_20260307_220335" src="https://github.com/user-attachments/assets/afd75626-0286-414c-a441-f57eb893a67f" />
