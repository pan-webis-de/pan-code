MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

DOCKER := $(if $(shell which podman),podman,docker)

all: mypy

mypy:
	mypy .

docker-%:
	$(DOCKER) build -f Dockerfile --build-arg "LANGUAGE=$*" --tag "clef24-text-detoxification-evaluator:$*"

models: models-english models-russian

models-english:
	huggingface-cli download s-nlp/roberta_toxicity_classifier
	huggingface-cli download Elron/bleurt-large-128
	huggingface-cli download cointegrated/roberta-large-cola-krishna2020

models-russian:
	huggingface-cli download IlyaGusev/rubertconv_toxic_clf
	huggingface-cli download s-nlp/rubert-base-cased-conversational-paraphrase-v1
	huggingface-cli download SkolkovoInstitute/ruRoberta-large-RuCoLa-v1

evaluate-english:
	./evaluate.py \
	--input=sample/english/original.txt \
	--golden=sample/english/references.txt \
	--style-model=s-nlp/roberta_toxicity_classifier \
	--meaning-model=Elron/bleurt-large-128 \
	--fluency-model=cointegrated/roberta-large-cola-krishna2020 \
	sample/english/references.txt

evaluate-russian:
	./evaluate.py \
	--input=sample/russian/original.txt \
	--golden=sample/russian/references.txt \
	--style-model=IlyaGusev/rubertconv_toxic_clf \
	--meaning-model=s-nlp/rubert-base-cased-conversational-paraphrase-v1 \
	--fluency-model=SkolkovoInstitute/ruRoberta-large-RuCoLa-v1 \
	sample/russian/references.txt
