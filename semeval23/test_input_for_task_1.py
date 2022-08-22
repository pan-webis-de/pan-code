import importlib
cse = importlib.import_module('clickbait-spoiling-eval')


def test_input_task_1_suceeds():
    expected = {"1": "passage",  "2": "phrase", "3": "multi"}
    actual = cse.spoiler_predictions_to_map(cse.load_json_lines(('test-resources/valid-input-task-1.jsonl')))
    assert actual == expected


def test_input_task_2_fails():
    assert None == cse.spoiler_predictions_to_map(cse.load_json_lines(('test-resources/valid-input-task-2.jsonl')), lambda x: None)

