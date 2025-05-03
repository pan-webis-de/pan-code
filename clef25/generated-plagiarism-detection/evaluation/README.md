# Evaluation Script for [PAN 25 Generated Plagiarism Detection](https://pan.webis.de/clef25/pan25-web/generated-plagiarism-detection.html)

This directory contains the evaluation script for the [2025 edition of the generated plagiarism detection task](https://pan.webis.de/clef25/pan25-web/generated-plagiarism-detection.html).

You can either run the script directly, or you can run the dockerized version via the `tira` python package (install via `pip3 install tira`).

## Development

Build the docker image via:
```
docker build -t mam10eks/pan25-generated-plagiarism-detection-evaluator:0.0.1 .
```

Upload the docker image via:

```
docker push mam10eks/pan25-generated-plagiarism-detection-evaluator:0.0.1
```

