# Transformer Baseline for Task 2 of Clickbait Spoiling on Spoiler Generation

This is a baseline for task 2 that spoils each clickbait post using a DeBERTa question answering mdel.

You can run it directly via: `docker run webis/pan-clickbait-spoiling-baselines:task2-transformer-0.0.2 --help`.
To use this baseline in TIRA, you can upload it (you have to tag it accordingly, e.g., `docker tag webis/pan-clickbait-spoiling-baselines:task2-transformer-0.0.2  registry.webis.de/code-research/tira/<YOUR-USER-NAME>/transformer-baseline-task2:0.0.2`, and push it with `docker push registry.webis.de/code-research/tira/<YOUR-USER-NAME>/transformer-baseline-task2:0.0.2`) and use the command `/transformer-baseline-task-2.py --input $inputDataset/input.jsonl --output $outputDir/run.jsonl`.

To get an interactive development environment where you can try out things, you can start a jupyter notebook with:

```
docker run \
	--net=host --rm -ti \
	-v ${PWD}:/repo -v /var/run/docker.sock:/var/run/docker.sock -v /tmp:/tmp \
	--workdir=/repo \
	--entrypoint jupyter-lab \
	webis/pan-clickbait-spoiling-baselines:task2-transformer-0.0.2 --ip 0.0.0.0 --allow-root
```

The notebook [example-usage.ipynb](example-usage.ipynb) provides some examples.

## Development

You can build the Docker image via: `docker build -t webis/pan-clickbait-spoiling-baselines:task2-transformer-0.0.2 .`

To publish the image to dockerhub, run: `docker push webis/pan-clickbait-spoiling-baselines:task2-transformer-0.0.2`

