# Evaluation of Approaches for the Clickbait Challenge at SemEval 2023 - Clickbait Spoiling

The code in this directory is used to evaluate the predictions of task 1 (spoiler type classification) and task 2 (spoiler generation) of the clickbait spoiling challenge at SemEval.

We provide the evaluation code in docker images to simplify the usage.

Assuming you have a run `run.jsonl` in your current directory, you can verify that the run is valid for task 1 with:

```
docker run -v ${PWD}:/input --rm -ti webis/pan-clickbait-spoiling-evaluator:0.0.1 --task 1 --input_run /input/run.jsonl
```

Assuming you have a run `run.jsonl` in your current directory, you can verify that the run is valid for task 2 with:

```
docker run -v ${PWD}:/input --rm -ti webis/pan-clickbait-spoiling-evaluator:0.0.1 --task 2 --input_run /input/run.jsonl
```

## Evaluation for 


The code in this directory is used to evaluate the predictions of spoiling approaches.

The focus of evaluation is mainly on the semantical similarity of two sections of text, but syntax shouldn't be neglected too much.
The metrics we chose are BLEU-4, METEOR 1.5 and BERTScore.

## Development



