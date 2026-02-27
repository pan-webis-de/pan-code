# See .devcontainer/Dockerfile.dev.
FROM webis/longeval-pyterrier-baseline:dev-0.0.1

ADD requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

ADD baseline.py /

