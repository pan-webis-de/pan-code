import importlib
cse = importlib.import_module('clickbait-spoiling-eval')


def test_input_task_1_fails():
    assert None == cse.spoiler_generations_to_map(cse.load_json_lines(('test-resources/valid-input-task-1.jsonl')), lambda x: None)


def test_input_task_2_fails():
    expected = {"1": "some spoiler 1", "2": "some spoiler 2"}
    actual = cse.spoiler_generations_to_map(cse.load_json_lines(('test-resources/valid-input-task-2.jsonl')), lambda x: None)
    print(actual)
    
    assert expected == actual

