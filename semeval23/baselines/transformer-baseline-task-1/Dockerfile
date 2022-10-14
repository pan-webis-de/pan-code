FROM python:3.6

RUN pip3 install pandas jupyterlab docker datasets transformers torch simpletransformers

RUN wget 'https://raw.githubusercontent.com/tira-io/tira/development/application/src/tira/templates/tira/tira_git_cmd.py' -O '/usr/local/lib/python3.6/site-packages/tira.py' \
	&& apt-get update \
	&& apt-get install git-lfs \
	&& git clone 'https://huggingface.co/webis/spoiler-type-classification' /model \
	&& cd /model \
	&& git lfs install \
	&& git fetch \
	&& git checkout --track origin/deberta-all-three-types-concat-1-checkpoint-1000-epoch-10 \
	&& rm -Rf .git

COPY transformer-baseline-task-1.py /

ENTRYPOINT [ "/transformer-baseline-task-1.py" ]

