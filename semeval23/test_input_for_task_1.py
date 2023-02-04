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
    expected = {"result-size": 3, "balanced-accuracy": 1.0,
                "precision-for-phrase-spoilers": 1.0, "recall-for-phrase-spoilers": 1.0,
                "f1-for-phrase-spoilers": 1.0,
                "precision-for-passage-spoilers": 1.0, "recall-for-passage-spoilers": 1.0,
                "f1-for-passage-spoilers": 1.0,
                "precision-for-multi-spoilers": 1.0, "recall-for-multi-spoilers": 1.0,
                "f1-for-multi-spoilers": 1.0,
                "missing-predictions": 0,
                }

    inp = cse.spoiler_predictions_to_map([{"uuid": "1", "tags": ["passage"]}, {"uuid": "2", "tags": ["phrase"]}, {"uuid": "3", "tags": ["multi"]}], lambda x: None, 'tags')
    
    actual = cse.create_protobuf_for_task_1(inp, inp)
    print('Expected: ' + str(expected))
    print('Actual: ' + str(actual))

    assert actual == expected

def test_evaluation_protobuff_for_non_perfect_result():
    expected = {"result-size": 3, "balanced-accuracy": 0.75,
                "precision-for-phrase-spoilers": 1.0, "recall-for-phrase-spoilers": 1.0,
                "f1-for-phrase-spoilers": 1.0,
                "precision-for-passage-spoilers": 1.0, "recall-for-passage-spoilers": 0.5,
                "f1-for-passage-spoilers": 0.6666666666666666,
                "precision-for-multi-spoilers": 0.0, "recall-for-multi-spoilers": 0.0,
                "f1-for-multi-spoilers": 0.0,
                "missing-predictions": 0}

    a = cse.spoiler_predictions_to_map([{"uuid": "1", "tags": ["passage"]}, {"uuid": "2", "tags": ["phrase"]}, {"uuid": "3", "tags": ["multi"]}], lambda x: None, 'tags')
    b = cse.spoiler_predictions_to_map([{"uuid": "1", "tags": ["passage"]}, {"uuid": "2", "tags": ["phrase"]}, {"uuid": "3", "tags": ["passage"]}], lambda x: None, 'tags')
    
    actual = cse.create_protobuf_for_task_1(a, b)
    print('Expected: ' + str(expected))
    print('Actual: ' + str(actual))

    assert actual == expected

def test_evaluation_protobuff_for_approach_that_alwayas_predicts_passage():
    expected = {"result-size": 3, "balanced-accuracy": 0.3333333333333333,
                "precision-for-phrase-spoilers": 0.0, "recall-for-phrase-spoilers": 0.0,
                "f1-for-phrase-spoilers": 0.0,
                "precision-for-passage-spoilers": 0.3333333333333333, "recall-for-passage-spoilers": 1.0,
                "f1-for-passage-spoilers": 0.5,
                "precision-for-multi-spoilers": 0.0, "recall-for-multi-spoilers": 0.0,
                "f1-for-multi-spoilers": 0.0,
                "missing-predictions": 0}

    a = cse.spoiler_predictions_to_map(
        [{"uuid": "1", "tags": ["passage"]}, {"uuid": "2", "tags": ["passage"]}, {"uuid": "3", "tags": ["passage"]}],
        lambda x: None, 'tags')
    b = cse.spoiler_predictions_to_map(
        [{"uuid": "1", "tags": ["passage"]}, {"uuid": "2", "tags": ["phrase"]}, {"uuid": "3", "tags": ["multi"]}],
        lambda x: None, 'tags')

    actual = cse.create_protobuf_for_task_1(a, b)
    print('Expected: ' + str(expected))
    print('Actual: ' + str(actual))

    assert actual == expected

def test_evaluation_protobuff_for_approach_that_alwayas_predicts_phrase():
    expected = {"result-size": 3, "balanced-accuracy": 0.3333333333333333,
                "precision-for-phrase-spoilers": 0.3333333333333333, "recall-for-phrase-spoilers": 1.0,
                "f1-for-phrase-spoilers": 0.5,
                "precision-for-passage-spoilers": 0.0, "recall-for-passage-spoilers": 0.0,
                "f1-for-passage-spoilers": 0.0,
                "precision-for-multi-spoilers": 0.0, "recall-for-multi-spoilers": 0.0,
                "f1-for-multi-spoilers": 0.0,
                "missing-predictions": 0}

    a = cse.spoiler_predictions_to_map(
        [{"uuid": "1", "tags": ["phrase"]}, {"uuid": "2", "tags": ["phrase"]}, {"uuid": "3", "tags": ["phrase"]}],
        lambda x: None, 'tags')
    b = cse.spoiler_predictions_to_map(
        [{"uuid": "1", "tags": ["passage"]}, {"uuid": "2", "tags": ["phrase"]}, {"uuid": "3", "tags": ["multi"]}],
        lambda x: None, 'tags')

    actual = cse.create_protobuf_for_task_1(a, b)
    print('Expected: ' + str(expected))
    print('Actual: ' + str(actual))

    assert actual == expected

def test_evaluation_protobuff_for_approach_that_alwayas_predicts_multi():
    expected = {"result-size": 3, "balanced-accuracy": 0.3333333333333333,
                "precision-for-phrase-spoilers": 0.0, "recall-for-phrase-spoilers": 0.0,
                "f1-for-phrase-spoilers": 0.0,
                "precision-for-passage-spoilers": 0.0, "recall-for-passage-spoilers": 0.0,
                "f1-for-passage-spoilers": 0.0,
                "precision-for-multi-spoilers": 0.3333333333333333, "recall-for-multi-spoilers": 1.0,
                "f1-for-multi-spoilers": 0.5,
                "missing-predictions": 0}

    a = cse.spoiler_predictions_to_map(
        [{"uuid": "1", "tags": ["multi"]}, {"uuid": "2", "tags": ["multi"]}, {"uuid": "3", "tags": ["multi"]}],
        lambda x: None, 'tags')
    b = cse.spoiler_predictions_to_map(
        [{"uuid": "1", "tags": ["passage"]}, {"uuid": "2", "tags": ["phrase"]}, {"uuid": "3", "tags": ["multi"]}],
        lambda x: None, 'tags')

    actual = cse.create_protobuf_for_task_1(a, b)
    print('Expected: ' + str(expected))
    print('Actual: ' + str(actual))

    assert actual == expected
