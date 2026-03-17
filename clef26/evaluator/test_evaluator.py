import unittest
import evaluator


class EvaluatorTest(unittest.TestCase):
    def test_multiple_predictions(self):
        f1_score = evaluator.compute_score_multiple_predictions(self.truth, self.solution, 'changes', labels=[0,1])
        self.assertAlmostEqual(f1_score, 0.77, delta=0.01)

    def test_multiple_predictions_2(self):
        truth = {'1': {'changes': [0, 0, 0]}, '2': {'changes': [1, 0, 0]}}
        solution = {'1': {'changes': [0, 0, 0]}, '2': {'changes': [1, 0, 0]}}
        f1_score = evaluator.compute_score_multiple_predictions(truth, solution, 'changes', labels=[0,1])
        self.assertAlmostEqual(f1_score, 1)

    def test_extract_task_results(self):
        task1_result = evaluator.extract_task_results(self.truth, self.solution, 'changes')
        self.assertEqual(task1_result, ([[0, 0, 0], [1, 0, 0]], [[0, 0, 1], [1, 0, 0]]))

    def setUp(self):
        self.truth = {'1': {'changes': [0, 0, 0]},
                      '2': {'changes': [1, 0, 0]}}
        self.solution = {'1': {'changes': [0, 0, 1]},
                         '2': {'changes': [1, 0, 0]}}


if __name__ == '__main__':
    unittest.main()
