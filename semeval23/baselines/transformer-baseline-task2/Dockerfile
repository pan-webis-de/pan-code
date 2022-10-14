FROM python:3.6

RUN pip3 install pandas jupyterlab docker datasets transformers torch

RUN wget 'https://raw.githubusercontent.com/tira-io/tira/development/application/src/tira/templates/tira/tira_git_cmd.py' -O '/usr/local/lib/python3.6/site-packages/tira.py' \
	&& apt-get update \
	&& apt-get install git-lfs \
	&& git clone 'https://huggingface.co/webis/clickbait-spoiling-with-question-answering' /model \
	&& cd /model \
	&& git lfs install \
	&& git fetch \
	&& git checkout --track origin/debertalarge-all-cbs20-both-checkpoint-1200 \
	&& rm -Rf .git

COPY transformer-baseline-task2.py run_qa.py trainer_qa.py utils_qa.py /

ENTRYPOINT [ "/transformer-baseline-task2.py" ]

