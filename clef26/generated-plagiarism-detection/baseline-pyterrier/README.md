# PyTerrier Baseline for the Generative Plagiarism Detection 2026 at PAN

This directory contains a PyTerrier baseline for the Generative Plagiarism Detection task of [PAN 2026](https://pan.webis.de/clef26/pan26-web/generated-plagiarism-detection.html).

## Development

This directory is [configured as DevContainer](https://code.visualstudio.com/docs/devcontainers/containers), i.e., you can open this directory with VS Code or some other DevContainer compatible IDE to work directly in the Docker container with all dependencies installed.

If you want to run it locally, please install the dependencies via `pip3 install -r requirements.txt`.

To create a run, please run (the final dataset is not yet ready, you can use any other dataset for the moment, e.g., `pan26-generated-plagiarism-detection/spot-check-dataset-20260227-training` is the spot-check dataset):

```
./baseline.py --dataset pan26-generated-plagiarism-detection/spot-check-dataset-20260227-training --output output --index /tmp/indexes
```

## Running on the Test Data

The test data is available in TIRA or on [Zenodo](https://zenodo.org/records/19038846). For code-submissions, you can upload your approach to TIRA and run it on the test data there, for run submissions, please download the test (please note that the qrels are hidden until after the deadline) data from Zenodo and put it to a local directory, e.g., `test-data`. Please verify that your downloaded data is valid, e.g., via `md5sum test-data/*`. The output should look like:

```
31baa52aac61d768ad555112e4521082  test-data/corpus.jsonl.gz
3eb502962505ea4d22af6546c9286042  test-data/queries.jsonl
```

When you have the test data in a directory `test-data`, you can run the baseline via a command like:

```
./baseline.py --dataset test-data --output output-test-data --index /tmp/index-test-data
```


## Code Submission to TIRA

You can make code submissions where the tira client will build a docker image of your approach from the source code and upload the image to TIRA.io so that your software can run in TIRA.io. To submit this baseline as code submission to TIRA, please run (more detailed information are available in the [documentation](https://docs.tira.io/participants/participate.html#submitting-your-submission):

```
tira-cli code-submission \
    --path . \
    --task pan26-generated-plagiarism-detection \
    --dataset spot-check-dataset-20260227-training \
    --command '/baseline.py --dataset $inputDataset --index /tmp/indexes --output $outputDir' \
    --dry-run
```

If this is successfull, please re-run with removed the `--dry-run` flag to upload the software to TIRA.
