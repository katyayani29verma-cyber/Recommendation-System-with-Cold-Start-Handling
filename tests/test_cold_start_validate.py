import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.content.cold_start_validate import (
    genre_features,
    ranking_results,
    rows_for_ids,
)


class ColdValidationTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.array([10, 20, 30])
        self.users = np.array([1])
        self.relevance = csr_matrix([[1, 0, 1]])

    def test_perfect_ranking(self):
        result = ranking_results(
            np.array([[3.0, 1.0, 2.0]]),
            self.ids,
            self.relevance,
            self.users,
            k=2,
        ).iloc[0]

        self.assertAlmostEqual(result["ndcg"], 1.0)
        self.assertAlmostEqual(result["ap"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)

    def test_nonrelevant_first_item_reduces_metrics(self):
        result = ranking_results(
            np.array([[2.0, 3.0, 1.0]]),
            self.ids,
            self.relevance,
            self.users,
            k=2,
        ).iloc[0]

        discount = 1.0 / np.log2(3)

        self.assertAlmostEqual(
            result["ndcg"],
            discount / (1.0 + discount),
        )
        self.assertAlmostEqual(result["ap"], 0.25)
        self.assertAlmostEqual(result["recall"], 0.5)

    def test_ties_are_reproducible(self):
        arguments = (
            np.zeros((1, 3)),
            self.ids,
            self.relevance,
            self.users,
        )

        first = ranking_results(*arguments, k=2, tie_seed=42)
        second = ranking_results(*arguments, k=2, tie_seed=42)

        pd.testing.assert_frame_equal(first, second)

    def test_rejects_invalid_scores(self):
        with self.assertRaises(ValueError):
            ranking_results(
                np.array([[np.nan, 0.0, 1.0]]),
                self.ids,
                self.relevance,
                self.users,
            )

    def test_rejects_users_without_relevant_movies(self):
        with self.assertRaises(ValueError):
            ranking_results(
                np.zeros((1, 3)),
                self.ids,
                csr_matrix((1, 3)),
                self.users,
            )

    def test_genre_vocabulary_uses_only_warm_movies(self):
        features, vocabulary = genre_features(
            ["Comedy|Drama", "Drama", "Sci-Fi"],
            np.array([0, 1]),
        )

        self.assertEqual(vocabulary, ["Comedy", "Drama"])
        np.testing.assert_array_equal(
            features,
            [[1, 1], [0, 1], [0, 0]],
        )

    def test_id_alignment_preserves_requested_order(self):
        rows = rows_for_ids(
            np.array([10, 20, 30]),
            np.array([30, 10]),
        )

        np.testing.assert_array_equal(rows, [2, 0])


if __name__ == "__main__":
    unittest.main()