FROM python:3.6

RUN pip3 install pandas jupyterlab docker

COPY naive-baseline-task-2.py /naive-baseline-task-2.py

RUN wget 'https://raw.githubusercontent.com/tira-io/tira/development/application/src/tira/templates/tira/tira_git_cmd.py' -O '/usr/local/lib/python3.6/site-packages/tira.py'

ENTRYPOINT [ "/naive-baseline-task-2.py" ]

