import importlib
cse = importlib.import_module('clickbait-spoiling-eval')


def test_input_task_1_fails():
    assert None == cse.spoiler_generations_to_map(cse.load_json_lines(('test-resources/valid-input-task-1.jsonl')), lambda x: None)


def test_input_task_2_is_correct():
    expected = {"1": "some spoiler 1", "2": "some spoiler 2"}
    actual = cse.spoiler_generations_to_map(cse.load_json_lines(('test-resources/valid-input-task-2.jsonl')), lambda x: None)
    print('Expected: '+ str(expected))
    print('Actual: ' + str(actual))
    
    assert expected == actual

def test_input_task_2_can_be_filtered_for_multi():
    expected = {"1": "some spoiler 1"}
    actual = cse.spoiler_generations_to_map(cse.load_json_lines(('test-resources/valid-input-task-2.jsonl')),
                                            lambda x: None, 'multi')
    print('Expected: ' + str(expected))
    print('Actual: ' + str(actual))

    assert expected == actual


def test_input_task_2_can_be_filtered_for_phrase():
    expected = {"2": "some spoiler 2"}
    actual = cse.spoiler_generations_to_map(cse.load_json_lines(('test-resources/valid-input-task-2.jsonl')),
                                            lambda x: None, 'phrase')
    print('Expected: ' + str(expected))
    print('Actual: ' + str(actual))

    assert expected == actual

def test_evaluation_protobuff_for_perfect_result():
    expected = {
        'result-size': 2,
        'bleu-score': 1.0,
        'bert-score': 0.9999999403953552,
        'meteor-score': 1.0,
        'missing-predictions': 0
    }

    inp = cse.spoiler_generations_to_map([{"uuid": "1", "spoiler": "cat dog mouse"}, {"uuid": "2", "spoiler": "computer phone notebook"}], lambda x: None)
    actual = cse.create_protobuf_for_task_2(inp, inp)
    print('Expected: '+ str(expected))
    print('Actual: ' + str(actual))

    assert actual == expected

def test_evaluation_protobuff_for_non_perfect_result_01():
    expected = {"result-size": 2,
                "bleu-score": 0.28867513459481287,
                'bert-score': 0.9499878883361816,
                'meteor-score': 0.5593486681745896,
                "missing-predictions": 0
                }
    a = cse.spoiler_generations_to_map([{"uuid": "1", "spoiler": "cat dog mouse"}, {"uuid": "2", "spoiler": "computer notebook"}], lambda x: None)
    b = cse.spoiler_generations_to_map([{"uuid": "1", "spoiler": "cat dog"}, {"uuid": "2", "spoiler": "computer phone notebook"}], lambda x: None)
    
    actual = cse.create_protobuf_for_task_2(a, b)
    print('Expected: '+ str(expected))
    print('Actual: ' + str(actual))

    assert actual == expected


def test_evaluation_protobuff_for_non_perfect_result_02():
    expected = {"result-size": 2,
                "bleu-score": 0.7886751345948129,
                'bert-score': 0.97015780210495,
                'meteor-score': 0.8284543474093324,
                "missing-predictions": 0
                }

    a = cse.spoiler_generations_to_map([{"uuid": "1", "spoiler": "cat dog mouse"}, {"uuid": "2", "spoiler": "computer phone notebook"}], lambda x: None)
    b = cse.spoiler_generations_to_map([{"uuid": "1", "spoiler": "cat dog"}, {"uuid": "2", "spoiler": "computer phone notebook"}], lambda x: None)
    
    actual = cse.create_protobuf_for_task_2(a, b)
    print('Expected: '+ str(expected))
    print('Actual: ' + str(actual))

    assert actual == expected


def test_evaluation_protobuff_for_non_perfect_result_03():
    expected = {"result-size": 2,
                "bleu-score": 0.7886751345948129,
                'bert-score': 0.97015780210495,
                'meteor-score': 0.8284543474093324,
                "missing-predictions": 0
                }

    a = cse.spoiler_generations_to_map([{"uuid": "1", "spoiler": ["cat", "dog", "mouse"]}, {"uuid": "2", "spoiler": ["computer", "phone", "notebook"]}], lambda x: None)
    b = cse.spoiler_generations_to_map([{"uuid": "1", "spoiler": "cat dog"}, {"uuid": "2", "spoiler": "computer phone notebook"}], lambda x: None)
    
    actual = cse.create_protobuf_for_task_2(a, b)
    print('Expected: '+ str(expected))
    print('Actual: ' + str(actual))

    assert actual == expected
