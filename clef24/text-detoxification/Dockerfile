FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /evaluator

COPY . /evaluator

RUN \
apt-get update && \
apt-get install -y make && \
rm -rf /var/lib/apt/lists/* && \
pip3 install -r requirements.txt

CMD evaluate.py

ARG LANGUAGE

LABEL language=$LANGUAGE

RUN \
make "models-$LANGUAGE" && \
python3 -c "import transformers; transformers.utils.move_cache()"
