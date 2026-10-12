import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.core.signed_data_prep import (
    prepare_signed_interactions,
    split_signed_interactions,
)


class SignedDataPreparationTests(unittest.TestCase):
    def prepare(self, ratings):
        return prepare_signed_interactions(
            ratings,
            positive_threshold=4.0,
            negative_threshold=2.0,
            evaluation_user_ids=set(),
            catalog_movie_ids={10, 20, 30},
        )

    def test_positive_negative_and_unknown_are_distinct(self):
        ratings = pd.DataFrame({
            "userId": [1, 1, 2, 2, 3],
            "movieId": [10, 20, 10, 30, 20],
            "rating": [5.0, 1.0, 4.0, 3.0, 2.0],
        })

        positive, signed, user_ids, movie_ids = self.prepare(
            ratings
        )

        np.testing.assert_array_equal(user_ids, [1, 2, 3])
        np.testing.assert_array_equal(movie_ids, [10, 20])

        np.testing.assert_array_equal(
            positive.toarray(),
            [[1, 0], [1, 0], [0, 0]],
        )

        np.testing.assert_array_equal(
            signed.toarray(),
            [[1, -1], [1, 0], [0, -1]],
        )

        # User 2 never rated movie 20: it remains unknown.
        self.assertEqual(signed[1, 1], 0)

        # Movie 30 has only a neutral rating in this example.
        self.assertNotIn(30, movie_ids)

    def test_negatives_remain_in_training(self):
        positive = csr_matrix(
            [
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 1, 0],
            ],
            dtype=np.float32,
        )

        negatives = csr_matrix(
            [
                [0, 0, 0, -1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        signed = positive + negatives

        positive_train, signed_train, validation = (
            split_signed_interactions(
                positive,
                signed,
                fraction=0.2,
                seed=42,
            )
        )

        self.assertGreater(validation.nnz, 0)
        self.assertTrue(np.all(validation.data == 1))
        self.assertEqual(signed_train[0, 3], -1)

        self.assertEqual(
            signed_train.multiply(validation).nnz,
            0,
        )

        self.assertEqual(
            (signed_train + validation != signed).nnz,
            0,
        )

        self.assertEqual(
            (positive_train + validation != positive).nnz,
            0,
        )

    def test_rejects_overlapping_thresholds(self):
        ratings = pd.DataFrame({
            "userId": [1],
            "movieId": [10],
            "rating": [4.0],
        })

        with self.assertRaises(ValueError):
            prepare_signed_interactions(
                ratings,
                positive_threshold=2.0,
                negative_threshold=4.0,
                evaluation_user_ids=set(),
                catalog_movie_ids={10},
            )

    def test_rejects_evaluation_user_leakage(self):
        ratings = pd.DataFrame({
            "userId": [1],
            "movieId": [10],
            "rating": [1.0],
        })

        with self.assertRaisesRegex(ValueError, "Evaluation users"):
            prepare_signed_interactions(
                ratings,
                positive_threshold=4.0,
                negative_threshold=2.0,
                evaluation_user_ids={1},
                catalog_movie_ids={10},
            )


if __name__ == "__main__":
    unittest.main()