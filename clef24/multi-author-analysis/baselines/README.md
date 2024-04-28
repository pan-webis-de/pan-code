# Sample Random Baseline for Multi-Author Writing Style Analysis task

This is a simple example of a random baseline that can be used as a template to prepare the solution code submission on TIRA platform. Please install Docker, python3, and the tira-cli on your machine (`pip3 install -U tira`)

## Build the baseline with docker

```
docker build -t pan24-multi-author .
```

## Run the baseline locally

First, we will run our baseline on a small dataset (with 10 problem statements) to ensure that it works:

```
tira-run \
  --input-dataset multi-author-writing-style-analysis-2024/multi-author-spot-check-20240428-training \
  --image pan24-multi-author \
  --command 'python3 /baseline-random.py --input $inputDataset --output $outputDir'
```

Now that we have ensured that everything works, we can run it on the larger validation dataset:

```
tira-run \
  --input-dataset multi-author-writing-style-analysis-2024/pan24-multi-author-analysis-validation-20240219_0-training \
  --image pan24-multi-author \
  --command 'python3 /baseline-random.py --input $inputDataset --output $outputDir'
```
