---
configs:
- config_name: inputs
  data_files:
  - split: test
    path: ["*/test/*.txt"]
- config_name: truths
  data_files:
  - split: test
    path: ["*/test/*.json"]

tira_configs:
  resolve_inputs_to: "."
  resolve_truths_to: "."
  baseline:
    link: https://github.com/pan-webis-de/pan-code/tree/master/clef26/multi-author-analysis/naive-baseline
    command: /predict.py
    format:
      name: ["multi-author-writing-style-analysis-solutions"]
  input_format:
    name: "multi-author-writing-style-analysis-problems"
    config:
      max_size_mb: 150
  truth_format:
    name: "multi-author-writing-style-analysis-truths"
  evaluator:
    image: mam10eks/multi-author-analysis:eval-26
    command: python3 /evaluator.py -p ${inputRun} -t ${inputDataset} -o ${outputDir}
---

# Multi-Author Writing Style Analysis 2026: Test Dataset

This dataset is the configuration of the test dataset for Task-3@Pan on Multi-Author Writing Style Analysis.

Upload this to TIRA via (remove the `--dry-run` argument after a first test):

```
tira-cli dataset-submission --path multi-author-writing --task multi-author-writing-style-analysis-2026 --split test --dry-run
```

If everything works, the result should look like:

