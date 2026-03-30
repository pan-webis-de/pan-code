FROM python:3.13

RUN pip3 install click tira

ADD predict.py /

ENTRYPOINT [ "/predict.py" ]
