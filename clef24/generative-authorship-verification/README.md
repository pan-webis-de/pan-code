# Code for PAN24 Generative AI Authorship Verification

Run the evaluator locally (check requirements):

	~$ python3 evaluation/evaluator.py <predictions-file> <truth-file> <output-dir> [--outfile-name <name>]

Run in Docker container:

	~$ docker run --rm \
		-v /path/to/answers.jsonl:/answers.jsonl \
        -v /path/to/truth.jsonl:/truth.jsonl \
        -v /path/to/output:/out \
        ghcr.io/pan-webis-de/pan24-generative-authorship-evaluator \
        evaluator /answers.jsonl /truth.jsonl /out
