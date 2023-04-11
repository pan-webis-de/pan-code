# Random Baseline for subtask2 of Profiling Cryptocurrency Influencers with Few-shot Learning
This is a baseline for influencer interest identification (subtask2).

You can run it directly via: `docker run pan-profiling-cryptocurrency-baselines:subtask2-random-0.0.1 --help`. To use this baseline in TIRA, you can upload this image (you have to tag it accordingly, e.g., `docker tag pan-profiling-cryptocurrency-baselines:subtask2-random-0.0.1  registry.webis.de/code-research/tira/<YOUR-USER-NAME>/random-baseline-subtask2:0.0.1`, and push it with `docker push registry.webis.de/code-research/tira/<YOUR-USER-NAME>/random-baseline-subtask2:0.0.1)`


## Development
You can build the Docker image via: docker build -t pan-profiling-cryptocurrency-baselines:subtask2-random-0.0.1 .

To publish the image to dockerhub, run: docker push pan-profiling-cryptocurrency-baselines:subtask2-random-0.0.1

## TIRA Command
Insert it with the command 

`python3 /random-baseline-subtask-2.py --input $inputDataset/train_text.json --output $outputDir/subtask2.json` - (training dataset)
`python3 /random-baseline-subtask-2.py --input $inputDataset/test_text.json --output $outputDir/subtask2.json` - (test dataset)
