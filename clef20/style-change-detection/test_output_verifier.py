import unittest
import tempfile
import json
import output_verifier
import os
import warnings


class OutputVerifierTest(unittest.TestCase):
    """
    Tests output verifier for PAN20 Style Change Detection task
    """

    def test_file_content_invalid_json(self):
        """ test for invalid json file """
        solution_file_path = os.path.join(self.test_dir_name, 'solution-1.json')
        with open(solution_file_path, 'w') as fh:
            fh.write('{"key1: x, ""key2": y')
        self.assertEqual(output_verifier.get_solution_file_check_result(solution_file_path, '1', self.test_dir_name),
                         [output_verifier.ParseError.INVALID_JSON])

    def test_file_content_task1_missing(self):
        """ test for missing task 1 solution in json - warning expected """
        solution = {'changes': [0, 0]}
        solution_file_path = os.path.join(self.test_dir_name, 'solution-1.json')
        with open(solution_file_path, 'w') as fh:
            json.dump(solution, fh)

        with warnings.catch_warnings(record=True) as warnings_occurred:
            # cause all warnings to always be triggered.
            warnings.simplefilter("always")
            output_verifier.get_solution_file_check_result(solution_file_path, '1', self.test_dir_name)

            self.assertEqual(len(warnings_occurred), 1)
            self.assertTrue(issubclass(warnings_occurred[-1].category, SyntaxWarning))
            self.assertTrue("Task 1" in str(warnings_occurred[-1].message))

    def test_file_content_task2_missing(self):
        """ test for missing task 2 solution in json - warning expected """
        solution = {'multi-author': 0}
        solution_file_path = os.path.join(self.test_dir_name, 'solution-1.json')
        with open(solution_file_path, 'w') as fh:
            json.dump(solution, fh)

        with warnings.catch_warnings(record=True) as warnings_occurred:
            # cause all warnings to always be triggered.
            warnings.simplefilter("always")
            output_verifier.get_solution_file_check_result(solution_file_path, '1', self.test_dir_name)

            self.assertEqual(len(warnings_occurred), 1)
            self.assertTrue(issubclass(warnings_occurred[-1].category, SyntaxWarning))
            self.assertTrue("Task 2" in str(warnings_occurred[-1].message))

    def test_file_content_task1_format1(self):
        """ test for task 1 solution not being 0 or 1"""
        solution = {'multi-author': 3, 'changes': [1, 0]}
        solution_file_path = os.path.join(self.test_dir_name, 'solution-1.json')
        with open(solution_file_path, 'w') as fh:
            json.dump(solution, fh)
        self.assertEqual(output_verifier.get_solution_file_check_result(solution_file_path, '1', self.test_dir_name),
                         [output_verifier.ParseError.TASK1_INVALID_FORMAT])

    def test_file_content_task1_format2(self):
        """" test for task 1 format being correct """
        solution = {'multi-author': 1, 'changes': [1, 0]}
        solution_file_path = os.path.join(self.test_dir_name, 'solution-1.json')
        with open(solution_file_path, 'w') as fh:
            json.dump(solution, fh)
        self.assertEqual(output_verifier.get_solution_file_check_result(solution_file_path, '1', self.test_dir_name),
                         [])

    def test_file_content_task2_format1(self):
        """ test for task 2 solution not being an array"""
        solution = {'multi-author': 0, 'changes': 2}
        solution_file_path = os.path.join(self.test_dir_name, 'solution-problem-1.json')
        with open(solution_file_path, 'w') as fh:
            json.dump(solution, fh)
        self.assertEqual(output_verifier.get_solution_file_check_result(solution_file_path, '1', self.test_dir_name),
                         [output_verifier.ParseError.TASK2_INVALID_FORMAT])

    def test_file_content_task2_format2(self):
        """ test for correct json format """
        solution = {'multi-author': 0, 'changes': [0, 0]}
        solution_file_path = os.path.join(self.test_dir_name, 'solution-problem-1.json')
        with open(solution_file_path, 'w') as fh:
            json.dump(solution, fh)
        self.assertEqual(output_verifier.get_solution_file_check_result(solution_file_path, '1', self.test_dir_name),
                         [])

    def test_file_content_task2_format3(self):
        """ test for array for task 2 only holding 0 or 1 """
        solution = {'multi-author': 0, 'changes': [0, 2]}
        solution_file_path = os.path.join(self.test_dir_name, 'solution-problem-1.json')
        with open(solution_file_path, 'w') as fh:
            json.dump(solution, fh)
        self.assertEqual(output_verifier.get_solution_file_check_result(solution_file_path, '1', self.test_dir_name),
                         [output_verifier.ParseError.TASK2_INVALID_FORMAT])


    def test_file_content_task2_length(self):
        """ test whether length of task 2 array is correct """
        solution = {'multi-author': 0, 'changes': [0, 0, 1]}
        solution_file_path = os.path.join(self.test_dir_name, 'solution-problem-1.json')
        with open(solution_file_path, 'w') as fh:
            json.dump(solution, fh)
        self.assertEqual(output_verifier.get_solution_file_check_result(solution_file_path, '1', self.test_dir_name),
                         [output_verifier.ParseError.TASK2_INVALID_LENGTH])

    def test_paragraph_count(self):
        """ test counting of paragraphs """
        self.assertEqual(output_verifier.get_paragraph_count(1, self.test_dir_name), 3)

    def test_get_problem_ids1(self):
        """ test extraction of problem ids """
        self.assertEqual(output_verifier.get_problem_ids(self.test_dir_name), ['1'])

    def test_get_problem_ids2(self):
        """ test extraction of problem ids """
        fh = open(os.path.join(self.test_dir_name, 'problem-99.txt'), 'w')
        self.assertEqual(output_verifier.get_problem_ids(self.test_dir_name), ['1', '99'])
        fh.close()

    def setUp(self):
        """ set up test environment: create tmp folder, problem definition """
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_dir_name = self.test_dir.name

        # create input problem file
        text: str = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut " \
                    "labore et dolore magna aliquyam erat, sed diam voluptua. \n\n At vero eos et accusam et justo duo " \
                    "dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit " \
                    "amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt " \
                    "ut labore et dolore magna aliquyam erat, sed diam voluptua.\n\n At vero eos et accusam et justo duo " \
                    "dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit " \
                    "amet. "

        with open(os.path.join(self.test_dir_name, 'problem-1.txt'), 'w') as problem_file:
            problem_file.write(text)

        self.problem_file = problem_file.name

    def tearDown(self):
        """ clean up after tests, remove tmp folder and files """
        self.test_dir.cleanup()


if __name__ == '__main__':
    unittest.main()
