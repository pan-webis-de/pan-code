FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

RUN pip3 install pandas jupyterlab docker datasets transformers

RUN apt-get update \
	&& apt-get install -y git-lfs wget \
	&& wget 'https://raw.githubusercontent.com/tira-io/tira/development/application/src/tira/templates/tira/tira_git_cmd.py' -O '/opt/conda/lib/python3.7/site-packages/tira.py' \
	&& git clone 'https://huggingface.co/webis/clickbait-spoiling-with-question-answering' /model \
	&& cd /model \
	&& git lfs install \
	&& git fetch \
	&& git checkout --track origin/debertalarge-all-cbs20-both-checkpoint-1200 \
	&& rm -Rf .git

COPY transformer-baseline-task-2.py run_qa.py trainer_qa.py utils_qa.py /

ENTRYPOINT [ "/transformer-baseline-task-2.py" ]

