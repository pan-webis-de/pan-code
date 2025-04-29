#!/usr/bin/env bash

set -e
docker build -t ghcr.io/pan-webis-de/pan25-generative-authorship-baselines "$@" .
docker push ghcr.io/pan-webis-de/pan25-generative-authorship-baselines
