FROM python:3.10

RUN set -x \
    && pip install scikit-learn tira

COPY evaluator /opt/pan24_oppositional_evaluator
WORKDIR /opt/pan24_oppositional_evaluator

RUN set -x \
    && chmod +x /opt/pan24_oppositional_evaluator/evaluator.py \
    && ln -s /opt/pan24_oppositional_evaluator/evaluator.py /usr/local/bin/evaluator

CMD /usr/local/bin/evaluator
