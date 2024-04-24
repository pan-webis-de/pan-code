# docker build -t ghcr.io/pan-webis-de/pan24-generative-authorship-evaluator:latest .

FROM python:3.11

COPY . /opt/pan24_llm_evaluator
WORKDIR /opt/pan24_llm_evaluator
RUN set -x \
    && pip install --no-cache . \
    && rm -rf ./build ./*.egg-info

CMD evaluator
