# Sample Random Baseline for Multi-Author Writing Style Analysis task

This is a simple example of a random baseline that can be used as a template to prepare the solution code submission on TIRA platform.

## Build the baseline with docker

```
docker build -t pan24-multi-author .
```

## Run the baseline locally

```
tira-run \
  --input-dataset multi-author-writing-style-analysis-2024/pan24-multi-author-analysis-validation-20240219_0-training \
  --image pan24-multi-author \
  --command 'python3 /random_baseline.py --input $inputDataset --output $outputDir'
```
