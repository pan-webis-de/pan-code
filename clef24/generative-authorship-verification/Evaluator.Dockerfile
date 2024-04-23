# docker build -t ghcr.io/pan-webis-de/pan24-generative-authorship-evaluator:latest -f Evaluator.Dockerfile .

FROM python:3.11

COPY requirements.evaluator.txt /opt/pan24_llm_evaluator.reqirements.txt
RUN set -x \
    && pip install --no-cache -r /opt/pan24_llm_evaluator.reqirements.txt

COPY pan24_llm_evaluator /opt/pan24_llm_evaluator
WORKDIR /opt/pan24_llm_evaluator

RUN set -x \
    && chmod +x /opt/pan24_llm_evaluator/evaluator.py \
    && ln -s /opt/pan24_llm_evaluator/evaluator.py /usr/local/bin/evaluator

CMD /usr/local/bin/evaluator
