import unittest
from statistics import mean
import pan19_celebs_evaluator as pev

class TestRecallPrecision(unittest.TestCase):

    def test_pr_gender(self):
        truth = ["male", "female", "binary", "male", "female", "binary", "male", "female", "binary"]
        predictions = ["male", "female", "binary", "female", "binary", "male", "binary", "male", "female"]
        prec, rec = pev.mc_prec_rec(predictions, truth)
        self.assertEqual(mean(prec), 1/3)
        self.assertEqual(mean(rec), 1/3)

    def test_pr_age(self):
        truth = [1938, 1988, 2008, 1938, 1988, 2008, 1938, 1988, 2008]
        predictions = [1929, 1984, 2006, 2006, 1929, 1984, 1984, 2008, 1929]
        prec, rec = pev.mc_prec_rec(predictions, truth, hit_function=pev.age_window_hit)
        self.assertEqual(mean(prec), 2.5/3)
        self.assertEqual(mean(rec), 1/3)

    def test_pr_age2(self):
        truth = [2008, 2009, 2010, 2011, 2012]
        predictions = [2008, 2008, 2008, 2008, 2008]
        # after age_window_hit: predictions = [2008, 2009, 2008, 2008, 2008]
        prec, rec = pev.mc_prec_rec(predictions, truth, hit_function=pev.age_window_hit)
        self.assertEqual(mean([1/3, 1, 1, 0, 0]), mean(prec))
        self.assertEqual(mean([1, 1, 1, 0, 0]), mean(rec))

    def test_pr_age3(self):
        truth = [1978, 1979, 1980, 1981, 1982, 1983, 1984]
        predictions = [1978, 1978, 1978, 1978, 1978, 1978, 1978]
        # after age_window_hit: predictions = [1978, 1979, 1980, 1981, 1982, 1983, 1978]
        prec, rec = pev.mc_prec_rec(predictions, truth, hit_function=pev.age_window_hit)
        self.assertEqual(mean([1/2, 1, 1, 1, 1, 1, 0]), mean(prec))
        self.assertEqual(mean([1, 1, 1, 1, 1, 1, 0]), mean(rec))


class TestAgeWindowHit(unittest.TestCase):

    def test_bounds(self):
        self.assertTrue(pev.age_window_hit(2008, 2008))
        self.assertTrue(pev.age_window_hit(2006, 2008))
        self.assertTrue(pev.age_window_hit(2010, 2008))
        self.assertTrue(not pev.age_window_hit(2011, 2008))
        self.assertTrue(not pev.age_window_hit(2005, 2008))
        self.assertTrue(pev.age_window_hit(2008, 2010))

        self.assertTrue(pev.age_window_hit(1988, 1988))
        self.assertTrue(pev.age_window_hit(1984, 1988))
        self.assertTrue(pev.age_window_hit(1992, 1988))
        self.assertTrue(not pev.age_window_hit(1983, 1988))
        self.assertTrue(not pev.age_window_hit(1993, 1988))

        self.assertTrue(pev.age_window_hit(1929, 1938))
        self.assertTrue(pev.age_window_hit(1947, 1938))
        self.assertTrue(not pev.age_window_hit(1928, 1938))
        self.assertTrue(not pev.age_window_hit(1948, 1938))


if __name__ == "__main__":
    unittest.main()
