# PyTerrier Baseline for the Generative Plagiarism Detection 2026 at PAN

This directory contains a PyTerrier baseline for the Generative Plagiarism Detection task of [PAN 2026](https://pan.webis.de/clef26/pan26-web/generated-plagiarism-detection.html).

## Development

This directory is [configured as DevContainer](https://code.visualstudio.com/docs/devcontainers/containers), i.e., you can open this directory with VS Code or some other DevContainer compatible IDE to work directly in the Docker container with all dependencies installed.

If you want to run it locally, please install the dependencies via `pip3 install -r requirements.txt`.

To create a run, please run (the final dataset is not yet ready, you can use any other dataset for the moment, e.g., cranfield):

```
./baseline.py --dataset cranfield --output output --index /tmp/indexes
```

## Code Submission to TIRA

You can make code submissions where the tira client will build a docker image of your approach from the source code and upload the image to TIRA.io so that your software can run in TIRA.io. To submit this baseline as code submission to TIRA, please run (more detailed information are available in the [documentation](https://docs.tira.io/participants/participate.html#submitting-your-submission):

```
tira-cli code-submission \
    --path . \
    --task longeval-2026 \
    --dataset task-1-spot-check-20260225-training \
    --command '/baseline.py --dataset $inputDataset --index /tmp/indexes --output $outputDir' \
    --dry-run
```

If this is successfull, please re-run with removed the `--dry-run` flag to upload the software to TIRA.
