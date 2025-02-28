# WIP

Currently in progress: transforming the dataset into the format so that it is compatible with the PAN text alignment format.

Reading this format is currently in beta in the tira python client, install this via:

```
pip3 install 'git+https://github.com/tira-io/tira.git@pyterrier-artifacts#egg=tira&subdirectory=python-client'
```

Transform the dataset via:

```
cd transform-dataset
unzip sample_raw.zip -d sample_raw
./transform-dataset.py sample_raw ../data
```

