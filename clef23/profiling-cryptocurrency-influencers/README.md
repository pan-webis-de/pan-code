# Code for PAN23 Profiling Cryptocurrency Influencers
## Running the evaluator

Run the evaluator locally (check the requirements.txt)

	evaluation/~$ python3 evaluation_script_subtask1.py -s <predictions-dir>/subtask1.json -g <truth-dir>/test_truth.json -o <output-dir>

<!-- Run command for tira: 
	python3 evaluation_script_subtask1.py -s $inputRun/predictions.json -g $inputDataset/test_truth.json -o ${outputDir}
-->

Build the container for the evaluator:

    evaluation/~$ docker build -t <dockerhub-user>/pan23-profiling-cryptocurrency-influencers-evaluator:latest -f Dockerfile .


## TIRA baselines

TIRA baselines are available in the baselines directory. You have available a random baseline model for each subtask. Check out the Readme files in the baseline directories.

