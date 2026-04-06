# docker build -t mam10eks/multi-author-analysis:eval-26 .
FROM ubuntu

RUN apt-get update \
	&& apt-get install -y python3 python3-pip \
	&& pip3 install --break-system-packages tira click pandas scikit-learn

ADD evaluator.py /
