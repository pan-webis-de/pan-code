# Code for PAN23 Trigger Detection

## Running the evaluator

Run the evaluator locally (check the requirements.txt)

	~$ python3 evaluation/evaluator.py --predictions <predictions-dir>/labels.jsonl --truth <truth-dir>/labels.jsonl --output-dir <output-dir>

<!-- Run command for tira: 
	python3 evaluator.py --output-format protobuf --predictions ${inputRun}/labels.jsonl --truth ${inputDataset}/labels.jsonl --output-dir $outputDir
-->

Build the container for the evaluator:

    evaluation/~$ docker build -t <dockerhub-user>/pan23-trigger-detection-evaluator:latest -f Dockerfile .

## Running the baselines

Check out the Readme files in the baseline directories. 
