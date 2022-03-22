import unittest
import evaluator


class EvaluatorTest(unittest.TestCase):
    def test_multiple_predictions(self):
        f1_score = evaluator.compute_score_multiple_predictions(self.truth, self.solution, 'paragraph-authors', labels=[1,2])
        self.assertAlmostEqual(f1_score, 0.66, delta=0.01)

    def test_multiple_predictions_2(self):
        truth = {'1': {'changes': [0, 0, 0]}, '2': {'changes': [1, 0, 0]}}
        solution = {'1': {'changes': [0, 0, 0]}, '2': {'changes': [1, 0, 0]}}
        f1_score = evaluator.compute_score_multiple_predictions(truth, solution, 'changes', labels=[0,1])
        self.assertAlmostEqual(f1_score, 1)

    def test_jer(self):
        truth = {'1': {'paragraph-authors': [1, 2, 3]}, '2': {'paragraph-authors': [1, 1, 2, 2, 2]}}
        solution = {'1': {'paragraph-authors': [1, 2, 3]}, '2': {'paragraph-authors': [1, 1, 2, 2, 2]}}
        jer_score, _ = evaluator.compute_secondary_metrics(truth, solution, 'paragraph-authors')
        self.assertAlmostEqual(jer_score, 0)

    def test_jer2(self):
        truth = {'1': {'paragraph-authors': [1, 2]}}
        solution = {'1': {'paragraph-authors': [1, 1]}}
        jer_score, _ = evaluator.compute_secondary_metrics(truth, solution, 'paragraph-authors')
        self.assertAlmostEqual(jer_score, 0.75)

    def test_der(self):
        truth = {'1': {'paragraph-authors': [1, 2, 3]}, '2': {'paragraph-authors': [1, 1, 2, 2, 2]}}
        solution = {'1': {'paragraph-authors': [1, 2, 3]}, '2': {'paragraph-authors': [1, 1, 2, 2, 2]}}
        _, der_score = evaluator.compute_secondary_metrics(truth, solution, 'paragraph-authors')
        self.assertAlmostEqual(der_score, 0)

    def test_der2(self):
        truth = {'1': {'paragraph-authors': [1, 2]}}
        solution = {'1': {'paragraph-authors': [1, 1]}}
        _, der_score = evaluator.compute_secondary_metrics(truth, solution, 'paragraph-authors')
        self.assertAlmostEqual(der_score, 0.75)



    def test_der2(self):
        pass

    def test_extract_task_results(self):
        task1_result = evaluator.extract_task_results(self.truth, self.solution, 'changes')
        task2_result = evaluator.extract_task_results(self.truth, self.solution, 'paragraph-authors')
        self.assertEqual(task1_result, ([[0, 0, 0], [1, 0, 0]], [[0, 0, 1], [1, 0, 0]]))
        self.assertEqual(task2_result, ([[1, 1, 1, 1], [1, 2, 2, 2]], [[1, 1, 1, 1], [1, 1, 1, 2]]))

    def setUp(self):
        self.truth = {'1': {'changes': [0, 0, 0], 'paragraph-authors': [1, 1, 1, 1]},
                      '2': {'changes': [1, 0, 0], 'paragraph-authors': [1, 2, 2, 2]}}
        self.solution = {'1': {'changes': [0, 0, 1], 'paragraph-authors': [1, 1, 1, 1]},
                         '2': {'changes': [1, 0, 0], 'paragraph-authors': [1, 1, 1, 2]}}


if __name__ == '__main__':
    unittest.main()
