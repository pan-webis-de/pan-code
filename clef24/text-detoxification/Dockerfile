FROM python:3.11

WORKDIR /evaluator

COPY . /evaluator

RUN \
pip3 install -r requirements.txt && \
make models

CMD evaluate.py
