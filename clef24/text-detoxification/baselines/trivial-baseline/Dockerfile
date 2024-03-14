FROM ubuntu:22.04

RUN apt-get update \
    && apt-get upgrade -y python3 python3-pip python-is-python3 git \
    && pip install datasets

RUN python3 -c "from datasets import load_dataset; load_dataset('textdetox/multilingual_toxic_lexicon')"

COPY trivial_baseline.py .
ADD trivial_baseline.py /trivial_baseline.py
RUN chown root:root /trivial_baseline.py && \
    chmod +x /trivial_baseline.py

CMD ["python3", "/trivial_baseline.py"]
