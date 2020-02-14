import unittest
import evaluator


class EvaluatorTest(unittest.TestCase):
    def test_task1(self):
        f1_score = evaluator.compute_task1_f1_score(self.truth, self.solution)
        self.assertEqual(1, f1_score)

    def test_task2_1(self):
        f1_score = evaluator.compute_task2_f1_score(self.truth, self.solution)
        self.assertAlmostEqual(f1_score, 0.83, delta=0.01)

    def test_task2_2(self):
        truth = {'1': {'multi-author': 0, 'changes': [0, 0, 0]}, '2': {'multi-author': 1, 'changes': [1, 0, 0]}}
        solution = {'1': {'multi-author': 0, 'changes': [0, 0, 0]}, '2': {'multi-author': 1, 'changes': [1, 0, 0]}}
        f1_score = evaluator.compute_task2_f1_score(truth, solution)
        self.assertAlmostEqual(f1_score, 1)

    def test_extract_task_results(self):
        task1_result = evaluator.extract_task_results(self.truth, self.solution, 'multi-author')
        task2_result = evaluator.extract_task_results(self.truth, self.solution, 'changes')
        self.assertEqual(task1_result, ([0, 1], [0, 1]))
        self.assertEqual(task2_result, ([[0, 0, 0], [1, 0, 0]], [[0, 0, 1], [1, 0, 0]]))

    def setUp(self):
        self.truth = {'1': {'multi-author': 0, 'changes': [0, 0, 0]}, '2': {'multi-author': 1, 'changes': [1, 0, 0]}}
        self.solution = {'1': {'multi-author': 0, 'changes': [0, 0, 1]}, '2': {'multi-author': 1, 'changes': [1, 0, 0]}}


if __name__ == '__main__':
    unittest.main()
