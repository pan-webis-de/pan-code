FROM python:3.11

WORKDIR /evaluator
COPY . /evaluator
RUN \
pip3 install -r requirements.txt && \
huggingface-cli download s-nlp/roberta_toxicity_classifier && \
huggingface-cli download Elron/bleurt-large-128 && \
huggingface-cli download cointegrated/roberta-large-cola-krishna2020 && \
huggingface-cli download IlyaGusev/rubertconv_toxic_clf && \
huggingface-cli download s-nlp/rubert-base-cased-conversational-paraphrase-v1 && \
huggingface-cli download SkolkovoInstitute/ruRoberta-large-RuCoLa-v1

CMD evaluate.py
