MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

all: mypy

mypy:
	mypy .

models:
	huggingface-cli download s-nlp/roberta_toxicity_classifier
	huggingface-cli download Elron/bleurt-large-128
	huggingface-cli download cointegrated/roberta-large-cola-krishna2020
	huggingface-cli download IlyaGusev/rubertconv_toxic_clf
	huggingface-cli download s-nlp/rubert-base-cased-conversational-paraphrase-v1
	huggingface-cli download SkolkovoInstitute/ruRoberta-large-RuCoLa-v1

english:
	./evaluate.py \
	--input=sample/english/original.txt \
	--golden=sample/english/references.txt \
	--style-model=s-nlp/roberta_toxicity_classifier \
	--meaning-model=Elron/bleurt-large-128 \
	--fluency-model=cointegrated/roberta-large-cola-krishna2020 \
	sample/english/references.txt

russian:
	./evaluate.py \
	--input=sample/russian/original.txt \
	--golden=sample/russian/references.txt \
	--style-model=IlyaGusev/rubertconv_toxic_clf \
	--meaning-model=s-nlp/rubert-base-cased-conversational-paraphrase-v1 \
	--fluency-model=SkolkovoInstitute/ruRoberta-large-RuCoLa-v1 \
	sample/russian/references.txt
