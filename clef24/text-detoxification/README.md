# Text Detoxification

```shell
./evaluate.py \
	--input=sample/english/original.txt \
	--golden=sample/english/references.txt \
	--style-model=s-nlp/roberta_toxicity_classifier \
	--meaning-model=Elron/bleurt-large-128 \
	--fluency-model=cointegrated/roberta-large-cola-krishna2020 \
	sample/english/references.txt
```

```shell
./evaluate.py \
	--input=sample/russian/original.txt \
	--golden=sample/russian/references.txt \
	--style-model=IlyaGusev/rubertconv_toxic_clf \
	--meaning-model=s-nlp/rubert-base-cased-conversational-paraphrase-v1 \
	--fluency-model=SkolkovoInstitute/ruRoberta-large-RuCoLa-v1 \
	sample/russian/references.txt
```
