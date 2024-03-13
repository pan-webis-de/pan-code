FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

WORKDIR /evaluator

COPY . /evaluator

RUN \
    apt-get update && \
    apt-get install -y make && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install -r requirements.txt

RUN \
    make "models" && \
    python3 -c "import transformers; transformers.utils.move_cache()"