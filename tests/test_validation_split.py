import unittest

import numpy as np
from scipy.sparse import csr_matrix

from src.core.validation_split import split_interactions


class ValidationSplitTests(unittest.TestCase):
    def setUp(self):
        self.matrix = csr_matrix(
            [
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

    def split(self):
        return split_interactions(
            self.matrix,
            fraction=0.2,
            seed=42,
        )

    def test_split_preserves_shape_and_dtype(self):
        training, validation = self.split()

        self.assertEqual(training.shape, self.matrix.shape)
        self.assertEqual(validation.shape, self.matrix.shape)
        self.assertEqual(training.dtype, self.matrix.dtype)
        self.assertEqual(validation.dtype, self.matrix.dtype)

    def test_training_and_validation_do_not_overlap(self):
        training, validation = self.split()

        self.assertEqual(
            training.multiply(validation).nnz,
            0,
        )

    def test_split_reconstructs_original_interactions(self):
        training, validation = self.split()

        self.assertEqual(
            (training + validation != self.matrix).nnz,
            0,
        )

    def test_users_keep_training_support(self):
        training, _ = self.split()

        original_counts = np.diff(self.matrix.indptr)
        training_counts = np.diff(training.indptr)

        self.assertTrue(
            np.all(training_counts[original_counts > 0] >= 1)
        )

    def test_movies_keep_training_support(self):
        training, _ = self.split()

        original_counts = self.matrix.getnnz(axis=0)
        training_counts = training.getnnz(axis=0)

        self.assertTrue(
            np.all(training_counts[original_counts > 0] >= 1)
        )

    def test_single_positive_user_is_not_held_out(self):
        training, validation = self.split()

        self.assertEqual(training[3, 3], 1)
        self.assertEqual(validation[3].nnz, 0)

    def test_empty_user_row_is_preserved(self):
        training, validation = self.split()

        self.assertEqual(training[4].nnz, 0)
        self.assertEqual(validation[4].nnz, 0)

    def test_seed_is_reproducible(self):
        first_training, first_validation = self.split()
        second_training, second_validation = self.split()

        self.assertEqual(
            (first_training != second_training).nnz,
            0,
        )

        self.assertEqual(
            (first_validation != second_validation).nnz,
            0,
        )

    def test_source_matrix_is_unchanged(self):
        original = self.matrix.copy()
        self.split()

        self.assertEqual(
            (original != self.matrix).nnz,
            0,
        )

    def test_singleton_movies_remain_in_training(self):
        singleton_matrix = csr_matrix(
            [[1, 0], [0, 1]],
            dtype=np.float32,
        )

        training, validation = split_interactions(
            singleton_matrix,
            fraction=0.2,
            seed=42,
        )

        self.assertEqual(validation.nnz, 0)
        self.assertEqual(
            (training != singleton_matrix).nnz,
            0,
        )

    def test_rejects_invalid_fraction(self):
        for fraction in [0, 1, -0.2, np.nan, np.inf]:
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    split_interactions(
                        self.matrix,
                        fraction=fraction,
                        seed=42,
                    )

    def test_rejects_nonbinary_matrix(self):
        bad = csr_matrix(
            [[1, 2], [1, 1]],
            dtype=np.float32,
        )

        with self.assertRaises(ValueError):
            split_interactions(
                bad,
                fraction=0.2,
                seed=42,
            )


if __name__ == "__main__":
    unittest.main()