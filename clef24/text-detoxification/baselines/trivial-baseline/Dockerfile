FROM ubuntu:22.04

RUN apt-get update \
	&& apt-get upgrade -y python3 python3-pip python-is-python3 git \
	&& git clone https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words.git /lodnoobw \
	&& rm /lodnoobw/LICENSE && rm /lodnoobw/README.md && rm /lodnoobw/USERS.md

ADD trivial-baseline.py /trivial-baseline.py

