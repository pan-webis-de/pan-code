FROM python:3.11

RUN set -x \
    && pip install click numpy scikit-learn

COPY evaluator /opt/pan24_llm_evaluator
WORKDIR /opt/pan24_llm_evaluator

RUN set -x \
    && chmod +x /opt/pan24_llm_evaluator/evaluator.py \
    && ln -s /opt/pan24_llm_evaluator/evaluator.py /usr/local/bin/evaluator

USER 1000
CMD /usr/local/bin/evaluator
