# docker build -t ghcr.io/pan-webis-de/pan25-generative-authorship-evaluator:latest .

FROM python:3.11

COPY . /opt/pan25_genai_evaluator
WORKDIR /opt/pan25_genai_evaluator
RUN set -x \
    && pip install --no-cache . \
    && rm -rf ./build ./*.egg-info

VOLUME /answers.jsonl
VOLUME /truth.jsonl
VOLUME /out

CMD pan25-evaluator
