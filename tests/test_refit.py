import unittest

import numpy as np
from scipy.sparse import csr_matrix

from src.core.refit import restore_full_training


class RefitTests(unittest.TestCase):
    def test_restores_positive_holdouts(self):
        training = csr_matrix(
            [[1, 0, 0], [0, 1, 0]],
            dtype=np.float32,
        )

        validation = csr_matrix(
            [[0, 1, 0], [0, 0, 1]],
            dtype=np.float32,
        )

        full = restore_full_training(training, validation)

        np.testing.assert_array_equal(
            full.toarray(),
            [[1, 1, 0], [0, 1, 1]],
        )

        self.assertEqual(full.nnz, 4)

    def test_preserves_negative_entries(self):
        training = csr_matrix(
            [[1, -1, 0], [0, 1, 0]],
            dtype=np.float32,
        )

        validation = csr_matrix(
            [[0, 0, 1], [1, 0, 0]],
            dtype=np.float32,
        )

        full = restore_full_training(training, validation)

        self.assertEqual(full[0, 1], -1)
        self.assertEqual(full.nnz, 5)

    def test_rejects_overlapping_interactions(self):
        matrix = csr_matrix([[1, 0]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "overlap"):
            restore_full_training(matrix, matrix)

    def test_rejects_negative_validation_labels(self):
        training = csr_matrix([[1, 0]], dtype=np.float32)
        validation = csr_matrix([[0, -1]], dtype=np.float32)

        with self.assertRaises(ValueError):
            restore_full_training(training, validation)

    def test_does_not_modify_source_matrices(self):
        training = csr_matrix([[1, 0]], dtype=np.float32)
        validation = csr_matrix([[0, 1]], dtype=np.float32)

        original_training = training.copy()
        original_validation = validation.copy()

        restore_full_training(training, validation)

        self.assertEqual(
            (training != original_training).nnz,
            0,
        )

        self.assertEqual(
            (validation != original_validation).nnz,
            0,
        )


if __name__ == "__main__":
    unittest.main()