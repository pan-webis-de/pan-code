MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

DOCKER := $(if $(shell which podman),podman,docker)

all: mypy

mypy:
	mypy .

docker-evaluate:
	$(DOCKER) build -f Dockerfile --tag "webis/clef24-text-detoxification-evaluator" .

models: models-download

models-download:
	huggingface-cli download textdetox/xlmr-large-toxicity-classifier 
	huggingface-cli download sentence-transformers/LaBSE

evaluate:
	./evaluate.py \
	--input=sample/english-tiny/input.jsonl \
	--golden=sample/english-tiny/references.jsonl \
	--prediction=sample/english-tiny/references.jsonl
	# --style-model=textdetox/xlmr-large-toxicity-classifier \
	# --meaning-model=sentence-transformers/LaBSE \