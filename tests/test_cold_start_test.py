import unittest

import numpy as np

from src.content.cold_start_test import paired_ndcg_interval


class ColdTestReportTests(unittest.TestCase):
    def test_identical_results_have_zero_difference(self):
        scores = np.array([0.1, 0.4, 0.8])

        result = paired_ndcg_interval(scores, scores)

        self.assertAlmostEqual(result["mean_ndcg_difference"], 0.0)
        self.assertAlmostEqual(result["lower_95"], 0.0)
        self.assertAlmostEqual(result["upper_95"], 0.0)

    def test_constant_improvement_has_correct_interval(self):
        baseline = np.array([0.1, 0.2, 0.3])

        result = paired_ndcg_interval(
            baseline + 0.1,
            baseline,
        )

        self.assertAlmostEqual(result["mean_ndcg_difference"], 0.1)
        self.assertAlmostEqual(result["lower_95"], 0.1)
        self.assertAlmostEqual(result["upper_95"], 0.1)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            paired_ndcg_interval([0.1], [0.1, 0.2])

        with self.assertRaises(ValueError):
            paired_ndcg_interval([], [])

        with self.assertRaises(ValueError):
            paired_ndcg_interval([np.nan], [0.1])


if __name__ == "__main__":
    unittest.main()