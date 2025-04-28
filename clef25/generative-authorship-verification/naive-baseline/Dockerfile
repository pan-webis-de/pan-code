# A python3 installation that already has python3, sklearn etc. installed.
# See: https://github.com/touche-webis-de/touche-code/blob/main/clef25/advertisement-detection/subtask-2-classification/naive-bayes-baseline/.devcontainer/Dockerfile
FROM webis/advertisement-detection-naive-bayes:0.0.1-dev

ADD predict.py /

ENTRYPOINT [ "/predict.py" ]
