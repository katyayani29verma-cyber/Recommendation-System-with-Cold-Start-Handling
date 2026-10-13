import unittest

import numpy as np
from scipy.sparse import csr_matrix

from src.core.cold_start_train import compact_warm_matrix


class ColdStartTrainingTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.array([10, 20, 30, 40])
        self.warm_ids = np.array([30, 10])

        self.matrix = csr_matrix(
            np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 1, 0],
            ], dtype=np.float32)
        )

    def test_preserves_requested_movie_order(self):
        result = compact_warm_matrix(
            self.matrix,
            self.ids,
            self.warm_ids,
        )

        np.testing.assert_array_equal(
            result.toarray(),
            [[0, 1], [1, 0], [1, 1]],
        )

    def test_rejects_cold_interactions_in_training(self):
        contaminated = self.matrix.toarray()
        contaminated[0, 1] = 1

        with self.assertRaises(ValueError):
            compact_warm_matrix(
                csr_matrix(contaminated),
                self.ids,
                self.warm_ids,
            )

    def test_rejects_warm_movie_without_support(self):
        with self.assertRaises(ValueError):
            compact_warm_matrix(
                self.matrix,
                self.ids,
                np.array([10, 30, 40]),
            )

    def test_rejects_unknown_or_duplicate_ids(self):
        for warm_ids in (
            np.array([10, 999]),
            np.array([10, 10]),
        ):
            with self.subTest(warm_ids=warm_ids):
                with self.assertRaises(ValueError):
                    compact_warm_matrix(
                        self.matrix,
                        self.ids,
                        warm_ids,
                    )

    def test_does_not_modify_input(self):
        before = self.matrix.toarray().copy()

        compact_warm_matrix(
            self.matrix,
            self.ids,
            self.warm_ids,
        )

        np.testing.assert_array_equal(
            self.matrix.toarray(),
            before,
        )


if __name__ == "__main__":
    unittest.main()