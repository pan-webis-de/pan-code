#docker build -t ghcr.io/pan-webis-de/pan24-generative-authorship-evaluator:latest .
FROM python:3.11

RUN set -x \
    && pip install click numpy scikit-learn tira

COPY evaluator /opt/pan24_llm_evaluator
WORKDIR /opt/pan24_llm_evaluator

RUN set -x \
    && chmod +x /opt/pan24_llm_evaluator/evaluator.py \
    && ln -s /opt/pan24_llm_evaluator/evaluator.py /usr/local/bin/evaluator

CMD /usr/local/bin/evaluator
