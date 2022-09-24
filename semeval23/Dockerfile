FROM python:3.6

RUN pip3 install sklearn nltk
RUN python3 -c 'import nltk; nltk.download("punkt"); nltk.download("stopwords")'
COPY clickbait-spoiling-eval.py .

ENTRYPOINT [ "./clickbait-spoiling-eval.py" ]

