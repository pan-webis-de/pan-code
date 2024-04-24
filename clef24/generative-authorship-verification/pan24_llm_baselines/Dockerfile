# docker build -t ghcr.io/pan-webis-de/pan24-generative-authorship-baselines:latest .

FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04

RUN set -x \
    && apt update \
    && apt install -y git python3 python3-packaging python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/pan24_llm_baselines
WORKDIR /opt/pan24_llm_baselines
RUN set -x \
    && pip install --no-cache --upgrade pip \
    && pip install --no-cache . \
    && pip install --no-cache .[flash-attn] \
    && rm -rf ./build ./*.egg-info

VOLUME /dataset.jsonl
VOLUME /out

ENTRYPOINT ["baseline"]
