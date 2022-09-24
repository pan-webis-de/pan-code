# Naive Baseline for Task 2 of Clickbait Spoiling on Spoiler Type Classification

This is a baseline for task 2 that spoils each clickbait post with the title of the linked page.

You can run it directly via: `docker run webis/pan-clickbait-spoiling-baselines:task2-naive-0.0.1 --help`.
To use this baseline in TIRA, you can insert it with the command `/naive-baseline-task-2.py --input $inputDataset/input.jsonl --output $outputDir/run.jsonl`.

## Development

You can build the Docker image via: `docker build -t webis/pan-clickbait-spoiling-baselines:task2-naive-0.0.1 .`

To publish the image to dockerhub, run: `docker push webis/pan-clickbait-spoiling-baselines:task2-naive-0.0.1`

