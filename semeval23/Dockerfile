FROM python:3.6

RUN pip3 install sklearn nltk
COPY clickbait-spoiling-eval.py .

ENTRYPOINT [ "./clickbait-spoiling-eval.py" ]

