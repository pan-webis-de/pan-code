# Evaluation of Approaches for the Clickbait Challenge at SemEval 2023 - Clickbait Spoiling

The code in this directory is used to evaluate the predictions of task 1 (spoiler type classification) and task 2 (spoiler generation) of the clickbait spoiling challenge at SemEval.

We provide the evaluation code in docker images to simplify the usage.

The output format for task 1 and task 2 is identical but other fields are mandatory.
Please submit your results in [JSON Lines format](https://jsonlines.org/) producing one output line for each input instance.

Each line should have the following format:

```
{"uuid": "<UUID>", "spoilerType": "<SPOILER-TYPE>", "spoiler": "<SPOILER>"}
```

where:

- `<UUID>` is the uuid of the input instance.
- `<SPOILER-TYPE>` Is the spoiler type (might be "phrase", "passage", or "multi") to be predicted in task 1. This field is mandatory for task 1 but optional for task 2 (to indicate that your system used some type of spoiler type classification during the spoiler generation).
- `<SPOILER>` Is the generated spoiler to be produced in task 2. This field is mandatory for task 2.

The [test-resources/](test-resources/) directory contains examples for task 1 and task 2.

Assuming you have a run `run.jsonl` in your current directory, you can verify that the run is valid for task 1 with:

```
docker run -v ${PWD}:/input --rm -ti webis/pan-clickbait-spoiling-evaluator:0.0.11 --task 1 --input_run /input/run.jsonl
```

Assuming you have a run `run.jsonl` in your current directory, you can verify that the run is valid for task 2 with:

```
docker run -v ${PWD}:/input --rm -ti webis/pan-clickbait-spoiling-evaluator:0.0.11 --task 2 --input_run /input/run.jsonl
```

## Evaluation for Task 1 and Task 2

The code in this directory is used to evaluate the predictions of spoiling approaches.
We will release the code for the evaluation soon.

The focus of evaluation is mainly on the semantical similarity of two sections of text, but syntax shouldn't be neglected too much.
The metrics we chose are BLEU-4, METEOR 1.5 and BERTScore.

## Development

- Run the unit tests via `make tests`.
- Build the docker image via `make build-docker-image`
- Publish the docker image via `make publish-docker-image`

## Integration in TIRA

Add this to TIRA via the image `webis/pan-clickbait-spoiling-evaluator:0.0.3` and the command `bash -c '/clickbait-spoiling-eval.py --task 1 --ground_truth_classes $inputDataset --input_run $inputRun --output_prototext ${outputDir}/evaluation.prototext'`.

