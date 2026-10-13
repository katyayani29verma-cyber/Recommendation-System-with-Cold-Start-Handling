import unittest

import numpy as np
from scipy.sparse import csr_matrix

from src.core.cold_start_split import split_by_movie


class ColdStartSplitTests(unittest.TestCase):
    def setUp(self):
        self.matrix = csr_matrix(
            np.array(
                [
                    [1, 1, 0, 1, 0, 0],
                    [0, 1, 1, 0, 1, 0],
                    [1, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            )
        )

    def test_cold_movies_have_no_training_interactions(self):
        split = split_by_movie(self.matrix)

        warm = split["warm_interactions"]

        self.assertEqual(
            warm[:, split["validation_columns"]].nnz,
            0,
        )
        self.assertEqual(
            warm[:, split["test_columns"]].nnz,
            0,
        )

        for group in ("warm", "validation", "test"):
            columns = split[f"{group}_columns"]

            np.testing.assert_array_equal(
                split[f"{group}_interactions"][:, columns].toarray(),
                self.matrix[:, columns].toarray(),
            )

    def test_groups_partition_supported_movies(self):
        split = split_by_movie(self.matrix)

        groups = [
            set(split[f"{group}_columns"])
            for group in ("warm", "validation", "test")
        ]

        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])

        self.assertEqual(set.union(*groups), {0, 1, 2, 3, 4})

    def test_interactions_reconstruct_source(self):
        split = split_by_movie(self.matrix)

        reconstructed = (
            split["warm_interactions"]
            + split["validation_interactions"]
            + split["test_interactions"]
        )

        self.assertEqual((reconstructed != self.matrix).nnz, 0)

    def test_seed_is_reproducible(self):
        first = split_by_movie(self.matrix, seed=42)
        second = split_by_movie(self.matrix, seed=42)

        for group in ("warm", "validation", "test"):
            np.testing.assert_array_equal(
                first[f"{group}_columns"],
                second[f"{group}_columns"],
            )

    def test_input_is_unchanged(self):
        before = self.matrix.toarray().copy()

        split_by_movie(self.matrix)

        np.testing.assert_array_equal(
            self.matrix.toarray(),
            before,
        )

    def test_users_require_warm_and_cold_support(self):
        split = split_by_movie(self.matrix)

        for group in ("validation", "test"):
            warm_counts = np.asarray(
                split["warm_interactions"].getnnz(axis=1)
            ).ravel()

            cold_counts = np.asarray(
                split[f"{group}_interactions"].getnnz(axis=1)
            ).ravel()

            expected = (warm_counts > 0) & (cold_counts > 0)

            np.testing.assert_array_equal(
                split[f"{group}_users"],
                expected,
            )

            self.assertFalse(split[f"{group}_users"][3])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            split_by_movie(
                self.matrix,
                validation_fraction=0.6,
                test_fraction=0.5,
            )

        with self.assertRaises(ValueError):
            split_by_movie(self.matrix, seed=-1)

        with self.assertRaises(ValueError):
            split_by_movie(csr_matrix([[1, -1, 1]]))

        with self.assertRaises(ValueError):
            split_by_movie(csr_matrix([[1, 1]]))

        with self.assertRaises(ValueError):
            split_by_movie(csr_matrix((3, 5)))


if __name__ == "__main__":
    unittest.main()