from click.testing import CliRunner
from validator import validate, ValidationError
from evaluator import evaluate, _evaluate
from pathlib import Path


def exists(labels_a, labels_non, labels_wf):
    """ Check if the validator responds correctly if different (wrong) files are passed """
    runner = CliRunner()

    # If one of the files does not exist, we should fail with EC2 (click checks)
    result = runner.invoke(validate, f"-t {labels_a} -p {labels_non}")
    assert result.exit_code == 2
    result = runner.invoke(validate, f"-t {labels_non} -p {labels_a}")
    assert result.exit_code == 2
    result = runner.invoke(validate, f"-t {labels_non} -p {labels_non}")
    assert result.exit_code == 2

    # If both files exist and have the correct name, we should terminate with EC0
    result = runner.invoke(validate, f"-t {labels_a} -p {labels_a}")
    assert result.exit_code == 0

    # If a file is not named correctly, throw a value error
    result = runner.invoke(validate, f"-t {labels_a} -p {labels_wf}")  #
    assert isinstance(result.exception, ValidationError)
    assert result.exit_code == 1
    return True


def keys():
    """  """
    pass


def test_validator():
    labels_non = "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/freeform-tag-analysis/dataset/pan23/release/pan23-trigger-detection-validation/labels.ndjson"
    labels_wf = "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/freeform-tag-analysis/dataset/pan23/release/pan23-trigger-detection-validation/works.jsonl"
    labels_a = "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/freeform-tag-analysis/dataset/pan23/release/pan23-trigger-detection-validation/labels.jsonl"
    labels_b = "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/freeform-tag-analysis/dataset/pan23/release/pan23-trigger-detection-input/labels.jsonl"
    labels_c = "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/freeform-tag-analysis/dataset/pan23/tira/pan23-trigger-detection-test-truth/labels.jsonl"
    assert exists(labels_c, labels_non, labels_wf)


def test_evaluator():
    labels_a = "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/freeform-tag-analysis/dataset/pan23/release/pan23-trigger-detection-validation/labels.jsonl"

    _evaluate

    runner = CliRunner()
    result = runner.invoke(evaluate, f"-t {labels_a} -p {labels_a} -o './'")
    assert result.exit_code == 0

    result = runner.invoke(evaluate, f"-t {labels_a} -p {labels_a} -o './' -f json")
    assert result.exit_code == 0

    result = runner.invoke(evaluate, f"-t {labels_a} -p {labels_a} -o './' -e")
    assert result.exit_code == 0


if __name__ == "__main__":
    test_validator()
    test_evaluator()
