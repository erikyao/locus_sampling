import unittest
import pandas as pd
import numpy as np
from locus_sampling.scoring import avg_rank_func, avg_rank_func2
from locus_sampling.exceptions import NoPostiveInGroupError


class ScoringTestCase(unittest.TestCase):
    def test_avg_rank_func_with_groups(self):
        y = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([[0.2, 0.8],
                           [0.1, 0.9],
                           [0.3, 0.7],
                           [0.4, 0.6],
                           [0.5, 0.5]])
        groups = np.array([0, 0, 1, 1, 2])
        # (1 + 2 + 1) / 3 = 1.33
        self.assertAlmostEqual(avg_rank_func(y, y_pred, groups, method="min"), 1.33333, places=5)
        # (1 + 2) / 2 = 1.5
        self.assertAlmostEqual(avg_rank_func(y, y_pred, groups, method="min", exclude_loners=True), 1.5, places=5)

    def test_avg_rank_func_with_y_index(self):
        y = pd.Series(data=[0, 1, 0, 1, 1], index=[1, 2, 3, 4, 5])
        y_pred = np.array([[0.2, 0.8],
                           [0.1, 0.9],
                           [0.3, 0.7],
                           [0.4, 0.6],
                           [0.5, 0.5]])
        groups = np.array([99, 0, 0, 1, 1, 2, 2])
        # (1 + 2 + 1) / 3 = 1.33
        self.assertAlmostEqual(avg_rank_func(y, y_pred, groups, method="min"), 1.33333, places=5)
        # (1 + 2) / 2 = 1.5
        self.assertAlmostEqual(avg_rank_func(y, y_pred, groups, method="min", exclude_loners=True), 1.5, places=5)

    def test_avg_rank_func2(self):
        y = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([[0.2, 0.8],
                           [0.1, 0.9],
                           [0.3, 0.7],
                           [0.4, 0.6],
                           [0.5, 0.5]])
        groups = np.array([1, 1, 1, 1, 1])

        # (1 + 3 + 3) / 3 = 2.33333
        self.assertAlmostEqual(avg_rank_func2(y, y_pred, groups, method="average"), 2.33333, places=5)

        groups = np.array([1, 2, 2, 1, 1])

        # (1 + 2 + 2) / 3 = 1.66667
        self.assertAlmostEqual(avg_rank_func2(y, y_pred, groups, method="average"), 1.66667, places=5)

    def test_avg_rank_func2_exception(self):
        y = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([[0.2, 0.8],
                           [0.1, 0.9],
                           [0.3, 0.7],
                           [0.4, 0.6],
                           [0.5, 0.5]])
        groups = np.array([0, 1, 1, 1, 1])

        with self.assertRaises(NoPostiveInGroupError):
            avg_rank_func2(y, y_pred, groups, method="average")


if __name__ == '__main__':
    unittest.main()
