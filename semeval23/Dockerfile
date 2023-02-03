FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

RUN pip3 install sklearn nltk bert_score jupyterlab \
	&& apt-get update \
	&& apt-get install -y openjdk-11-jre 

RUN python3 -c 'import nltk; nltk.download("punkt"); nltk.download("stopwords")'
RUN python3 -c 'from bert_score import score; print(score(["28-Year-Old Chef Found Dead at San Francisco Mall"], ["28-year-old chef found dead in San Francisco mall"], lang="en"))'
COPY clickbait-spoiling-eval.py .
COPY evaluation/meteor-1.5/meteor-1.5.jar /meteor-1.5.jar
RUN apt-get install -y wget \
	&& mkdir /data \
	&& cd /data \
	&& wget https://files.webis.de/data-in-production/data-research/acl22-clickbait-spoiling/paraphrase-en.gz


ENTRYPOINT [ "./clickbait-spoiling-eval.py" ]

