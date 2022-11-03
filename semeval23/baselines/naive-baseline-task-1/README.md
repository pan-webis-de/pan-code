# Naive Baseline for Task 1 of Clickbait Spoiling on Spoiler Type Classification

This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.

You can run it directly via: `docker run webis/pan-clickbait-spoiling-baselines:task1-naive-0.0.1 --help`.
To use this baseline in TIRA, you can upload this image (you have to tag it accordingly, e.g., `docker tag webis/pan-clickbait-spoiling-baselines:task2-naive-0.0.1  registry.webis.de/code-research/tira/<YOUR-USER-NAME>/naive-baseline-task1:0.0.1`, and push it with `docker push registry.webis.de/code-research/tira/<YOUR-USER-NAME>/naive-baseline-task1:0.0.1`) and insert it with the command `/naive-baseline-task-1.py --input $inputDataset/input.jsonl --output $outputDir/run.jsonl`.

## Development

You can build the Docker image via: `docker build -t webis/pan-clickbait-spoiling-baselines:task1-naive-0.0.1 .`

To publish the image to dockerhub, run: `docker push webis/pan-clickbait-spoiling-baselines:task1-naive-0.0.1`

