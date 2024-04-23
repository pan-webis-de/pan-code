# docker build -t ghcr.io/pan-webis-de/pan24-generative-authorship-baselines:latest -f Baselines.Dockerfile .

FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04

RUN set -x \
    && apt update \
    && apt install -y git python3 python3-packaging python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.baselines.txt /opt/pan24_llm_baselines.reqirements.txt
RUN set -x \
    && pip install --no-cache -r /opt/pan24_llm_baselines.reqirements.txt \
    && pip install --no-cache --no-build-isolation flash-attn

COPY pan24_llm_baselines /opt/pan24_llm_baselines/pan24_llm_baselines
WORKDIR /opt/pan24_llm_baselines

RUN set -x \
    && chmod +x /opt/pan24_llm_baselines/pan24_llm_baselines/baseline.py \
    && ln -s /opt/pan24_llm_baselines/pan24_llm_baselines/baseline.py /usr/local/bin/baseline

ENV PYTHONPATH "/opt/pan24_llm_baselines"

ENTRYPOINT ["python3", "/usr/local/bin/baseline"]
