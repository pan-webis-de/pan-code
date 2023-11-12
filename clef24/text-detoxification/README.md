# Text Detoxification

```shell
./evaluate.py \
	--input=sample/english/input.jsonl \
	--golden=sample/english/references.jsonl \
	--style-model=s-nlp/roberta_toxicity_classifier \
	--meaning-model=Elron/bleurt-large-128 \
	--fluency-model=cointegrated/roberta-large-cola-krishna2020 \
	sample/english/references.jsonl
```

```shell
./evaluate.py \
	--input=sample/russian/input.jsonl \
	--golden=sample/russian/references.jsonl \
	--style-model=IlyaGusev/rubertconv_toxic_clf \
	--meaning-model=s-nlp/rubert-base-cased-conversational-paraphrase-v1 \
	--fluency-model=SkolkovoInstitute/ruRoberta-large-RuCoLa-v1 \
	sample/russian/references.jsonl
```

## Docker Images for Evaluation

```shell
make docker-english  # => webis/clef24-text-detoxification-evaluator:english
make docker-russian  # => webis/clef24-text-detoxification-evaluator:russian
```

### Build the Docker file

```
docker build -t webis/clef24-text-detoxification-evaluator:0.0.1 .
```
