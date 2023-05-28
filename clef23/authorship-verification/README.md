# Code for PAN23 Authorship Verification

## Running the evaluator

Run the evaluator locally (check the requirements.txt). Evaluator expects the file with predictions in `<predictions-dir>/answers.jsonl` and the file with ground truth in `<truth-dir>/truth.jsonl`. The output will be written to `<output-dir>`.

	~$ python3 evaluator/evaluator.py -a <predictions-dir> -i <truth-dir> -o <output-dir>

<!-- Run command for tira: 
	python3 evaluator.py -a ${inputRun} -i ${inputDataset} -o ${outputDir}
-->

Build the container for the evaluator:

    $ docker build -t <dockerhub-user>/pan23-authorship-verification:latest -f Dockerfile .

## Running the baselines

tbd. 
