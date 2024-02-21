MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

DOCKER := $(if $(shell which podman),podman,docker)

all: mypy

mypy:
	mypy .

docker-evaluate:
	$(DOCKER) build -f Dockerfile --tag "webis/clef24-text-detoxification-evaluator" .

# models: models-multilingual

# models-multilingual:
# 	huggingface-cli login 
# 	huggingface-cli download textdetox/xlmr-large-toxicity-classifier
# 	huggingface-cli download sentence-transformers/LaBSE

evaluate:
	./evaluate.py \
	--input=sample/english-tiny/input.jsonl \
	--golden=sample/english-tiny/references.jsonl \
	--style-model=textdetox/xlmr-large-toxicity-classifier \
	--meaning-model=sentence-transformers/LaBSE \
	--prediction=sample/english-tiny/references.jsonl


# models-russian:
# 	huggingface-cli download IlyaGusev/rubertconv_toxic_clf
# 	huggingface-cli download s-nlp/rubert-base-cased-conversational-paraphrase-v1
# 	huggingface-cli download SkolkovoInstitute/ruRoberta-large-RuCoLa-v1



# evaluate-english:
# 	./evaluate.py \
# 	--input=sample/english/input.jsonl \
# 	--golden=sample/english/references.jsonl \
# 	--style-model=s-nlp/roberta_toxicity_classifier \
# 	--meaning-model=Elron/bleurt-large-128 \
# 	--fluency-model=cointegrated/roberta-large-cola-krishna2020 \
# 	sample/english/references.jsonl

# evaluate-russian:
# 	./evaluate.py \
# 	--input=sample/russian/input.jsonl \
# 	--golden=sample/russian/references.jsonl \
# 	--style-model=IlyaGusev/rubertconv_toxic_clf \
# 	--meaning-model=s-nlp/rubert-base-cased-conversational-paraphrase-v1 \
# 	--fluency-model=SkolkovoInstitute/ruRoberta-large-RuCoLa-v1 \
# 	sample/russian/references.jsonl
