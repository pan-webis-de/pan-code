import importlib
cse = importlib.import_module('clickbait-spoiling-eval')


def test_input_task_1_fails():
    assert None == cse.spoiler_generations_to_map(cse.load_json_lines(('test-resources/valid-input-task-1.jsonl')), lambda x: None)


def test_input_task_2_fails():
    expected = {"1": "some spoiler 1", "2": "some spoiler 2"}
    actual = cse.spoiler_generations_to_map(cse.load_json_lines(('test-resources/valid-input-task-2.jsonl')), lambda x: None)
    print(actual)
    
    assert expected == actual

def test_evaluation_protobuff_for_perfect_result():
    expected = '''measure{
  key: "result-size"
  value: "2"
}
measure{
  key: "bleu-score"
  value: "1.0"
}
measure{
  key: "missing-predictions"
  value: "0"
}'''
    inp = cse.spoiler_generations_to_map([{"uuid": "1", "spoiler": "cat dog mouse"}, {"uuid": "2", "spoiler": "computer phone notebook"}], lambda x: None)
    actual = cse.create_protobuf_for_task_2(inp, inp)
    print(actual)

    assert actual == expected

def test_evaluation_protobuff_for_non_perfect_result_01():
    expected = '''measure{
  key: "result-size"
  value: "2"
}
measure{
  key: "bleu-score"
  value: "0.28867513459481287"
}
measure{
  key: "missing-predictions"
  value: "0"
}'''
    a = cse.spoiler_generations_to_map([{"uuid": "1", "spoiler": "cat dog mouse"}, {"uuid": "2", "spoiler": "computer notebook"}], lambda x: None)
    b = cse.spoiler_generations_to_map([{"uuid": "1", "spoiler": "cat dog"}, {"uuid": "2", "spoiler": "computer phone notebook"}], lambda x: None)
    
    actual = cse.create_protobuf_for_task_2(a, b)
    print(actual)

    assert actual == expected


def test_evaluation_protobuff_for_non_perfect_result_02():
    expected = '''measure{
  key: "result-size"
  value: "2"
}
measure{
  key: "bleu-score"
  value: "0.7886751345948129"
}
measure{
  key: "missing-predictions"
  value: "0"
}'''
    a = cse.spoiler_generations_to_map([{"uuid": "1", "spoiler": "cat dog mouse"}, {"uuid": "2", "spoiler": "computer phone notebook"}], lambda x: None)
    b = cse.spoiler_generations_to_map([{"uuid": "1", "spoiler": "cat dog"}, {"uuid": "2", "spoiler": "computer phone notebook"}], lambda x: None)
    
    actual = cse.create_protobuf_for_task_2(a, b)
    print(actual)

    assert actual == expected


def test_evaluation_protobuff_for_non_perfect_result_03():
    expected = '''measure{
  key: "result-size"
  value: "2"
}
measure{
  key: "bleu-score"
  value: "0.7886751345948129"
}
measure{
  key: "missing-predictions"
  value: "0"
}'''
    a = cse.spoiler_generations_to_map([{"uuid": "1", "spoiler": ["cat", "dog", "mouse"]}, {"uuid": "2", "spoiler": ["computer", "phone", "notebook"]}], lambda x: None)
    b = cse.spoiler_generations_to_map([{"uuid": "1", "spoiler": "cat dog"}, {"uuid": "2", "spoiler": "computer phone notebook"}], lambda x: None)
    
    actual = cse.create_protobuf_for_task_2(a, b)
    print(actual)

    assert actual == expected
