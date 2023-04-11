
FROM python:3.6

RUN pip3 install pandas docker

COPY random-baseline-subtask-1.py /random-baseline-subtask-1.py

RUN wget 'https://raw.githubusercontent.com/tira-io/tira/development/application/src/tira/templates/tira/tira_git_cmd.py' -O '/usr/local/lib/python3.6/site-packages/tira.py'

ENTRYPOINT [ "/random-baseline-subtask-1.py" ]

