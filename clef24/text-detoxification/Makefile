MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

DOCKER := $(if $(shell which podman),podman,docker)

all: mypy

mypy:
	mypy .

docker-evaluate:
	$(DOCKER) build -f Dockerfile --tag "webis/clef24-text-detoxification-evaluator" .

evaluate:
	./evaluate.py \
	--input=sample/english-tiny/input.jsonl \
	--golden=sample/english-tiny/references.jsonl \
	--style-model=textdetox/xlmr-large-toxicity-classifier \
	--meaning-model=sentence-transformers/LaBSE \
	--prediction=sample/english-tiny/references.jsonl