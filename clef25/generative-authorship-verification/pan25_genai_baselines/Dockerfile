FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN set -x \
    && apt update \
    && apt install -y git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/pan25_genai_baselines
WORKDIR /opt/pan25_genai_baselines
RUN set -x \
    && python3 -m pip config set global.break-system-packages true \
    && python3 -m pip install --no-cache . \
    && python3 -m pip install --no-cache --no-build-isolation flash-attn \
    && rm -rf ./build ./*.egg-info

ENTRYPOINT ["/usr/local/bin/pan25-baseline"]
