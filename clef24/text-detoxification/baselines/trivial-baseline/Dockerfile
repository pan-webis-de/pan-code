FROM ubuntu:22.04

RUN apt-get update \
	&& apt-get upgrade -y python3 python3-pip python-is-python3 git \
	&& pip install datasets

ADD delete_baseline.py /delete_baseline.py

RUN chmod +x /trivial_baseline.py && \
	python3 -c "from datasets import load_dataset; load_dataset('textdetox/multilingual_toxic_lexicon')"
