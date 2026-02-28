# docker build -t mam10eks/pan-watermarking-prototype:eval-0.0.1 .
FROM ubuntu

RUN apt-get update \
	&& apt-get install -y python3 python3-pip \
	&& pip3 install --break-system-packages tira click pandas

ADD evaluator.py /

