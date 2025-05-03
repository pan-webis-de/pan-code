

## Development

This directory is [configured as DevContainer](https://code.visualstudio.com/docs/devcontainers/containers), i.e., you can open this directory with VS Code or some other DevContainer compatible IDE to work directly in the Docker container with all dependencies installed.

If you want to run it locally, please install the dependencies via `pip3 install -r requirements.txt`.

To make predictions on a dataset, run:

```
./predict.py --dataset multi-author-writing-spot-check-20250503-training --output predictions
```

The `--dataset` either must point to a local directory or must be the ID of a dataset in TIRA ([tira.io/datasets?query=multi-author-writing](https://archive.tira.io/datasets?query=multi-author-writing) shows an overview of available datasets.

To evaluate your submission locally, you can run the official evaluator locally via (install the tira client via `pip3 install tira`):

```
tira-cli evaluate --predictions . --dataset multi-author-writing-spot-check-20250503-training
```

## Submit to TIRA

Detailed information on how to submit software to TIRA is available in the [documentation](https://docs.tira.io/participants/participate.html#submitting-your-submission).

To submit this baseline to TIRA, please first ensure that your TIRA client is authenticated (you can find your authentication token ):

```
tira-cli login --token YOUR-AUTHENTICATION-TOKEN
```

Next, please verify that your system has all required dependencies for a software submission to TIRA:

```
tira-cli verify-installation
```

Finally, you can upload your code submission via (add the `--dry-run` flag to test that everything works):

```
tira-cli code-submission --path . --task multi-author-writing-style-analysis-2025 --dataset multi-author-writing-spot-check-20250503-training --command '/predict.py --dataset $inputDataset --output $outputDir --predict 0'
```
