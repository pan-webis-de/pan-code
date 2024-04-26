# Code for PAN24 Multi-Author Analysis

## Running the evaluator

Run the evaluator locally (check the requirements.txt)

	~$ python3 evaluator/evaluator.py --predictions <predictions-dir> --truth <truth-dir> --output <output-dir>

Run command for tira: 
	~$ python3 evaluator.py -p ${inputRun} -t ${inputDataset} -o ${outputDir}


Build the container for the evaluator:

    $ docker build -t <dockerhub-user>/pan24-multi-author-analysis-evaluator:latest -f Dockerfile .

## Running the baselines

tbd. 
