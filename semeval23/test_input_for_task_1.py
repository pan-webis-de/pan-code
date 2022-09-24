import importlib
cse = importlib.import_module('clickbait-spoiling-eval')


def test_input_task_1_suceeds():
    expected = {"1": "passage",  "2": "phrase", "3": "multi"}
    actual = cse.spoiler_predictions_to_map(cse.load_json_lines(('test-resources/valid-input-task-1.jsonl')))
    assert actual == expected


def test_input_task_2_fails():
    assert None == cse.spoiler_predictions_to_map(cse.load_json_lines(('test-resources/valid-input-task-2.jsonl')), lambda x: None)

def test_input_for_task_1_can_be_extracted_from_ground_truth():
    expected = {"1": "passage",  "2": "phrase", "3": "multi"}
    inp = [{"uuid": "1", "tags": ["passage"]}, {"uuid": "2", "tags": ["phrase"]}, {"uuid": "3", "tags": ["multi"]}]
    actual = cse.spoiler_predictions_to_map(inp, lambda x: None, 'tags')

    assert actual == expected

def test_evaluation_protobuff_for_perfect_result():
    expected = '''measure{
  key: "result-size"
  value: "3"
}
measure{
  key: "balanced-accuracy"
  value: "1.0"
}
measure{
  key: "missing_predictions"
  value: "0"
}'''
    inp = cse.spoiler_predictions_to_map([{"uuid": "1", "tags": ["passage"]}, {"uuid": "2", "tags": ["phrase"]}, {"uuid": "3", "tags": ["multi"]}], lambda x: None, 'tags')
    
    actual = cse.create_protobuf_for_task_1(inp, inp)
    print(actual)

    assert actual == expected

def test_evaluation_protobuff_for_non_perfect_result():
    expected = '''measure{
  key: "result-size"
  value: "3"
}
measure{
  key: "balanced-accuracy"
  value: "0.75"
}
measure{
  key: "missing_predictions"
  value: "0"
}'''
    a = cse.spoiler_predictions_to_map([{"uuid": "1", "tags": ["passage"]}, {"uuid": "2", "tags": ["phrase"]}, {"uuid": "3", "tags": ["multi"]}], lambda x: None, 'tags')
    b = cse.spoiler_predictions_to_map([{"uuid": "1", "tags": ["passage"]}, {"uuid": "2", "tags": ["phrase"]}, {"uuid": "3", "tags": ["passage"]}], lambda x: None, 'tags')
    
    actual = cse.create_protobuf_for_task_1(a, b)
    print(actual)

    assert actual == expected
