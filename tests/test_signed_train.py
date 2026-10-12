from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from src.core.train import (
    confidence_matrix,
    train_als,
    validate_settings,
)


class SignedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = csr_matrix(
            [
                [1, 1, -1, 0],
                [-1, 1, 1, 0],
                [0, -1, 1, 1],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        cls.original = cls.training.copy()

        cls.settings = {
            "factors": 3,
            "regularization": 0.1,
            "positive_confidence": 10.0,
            "negative_confidence": 7.0,
            "iterations": 5,
            "num_threads": 1,
            "seed": 42,
        }

        cls.model, cls.history = train_als(
            cls.training,
            cls.settings,
            show_progress=False,
        )

    def test_confidence_weights_preserve_unknown_entries(self):
        weighted = confidence_matrix(
            self.training,
            self.settings,
        )

        np.testing.assert_array_equal(
            weighted.toarray()[0],
            [10, 10, -7, 0],
        )

        self.assertEqual(
            weighted.nnz,
            self.training.nnz,
        )

    def test_training_does_not_modify_labels(self):
        self.assertEqual(
            (self.training != self.original).nnz,
            0,
        )

    def test_factors_have_correct_dimensions_and_finite_values(self):
        self.assertEqual(
            self.model.user_factors.shape,
            (4, 3),
        )

        self.assertEqual(
            self.model.item_factors.shape,
            (4, 3),
        )

        self.assertTrue(
            np.isfinite(self.model.user_factors).all()
        )

        self.assertTrue(
            np.isfinite(self.model.item_factors).all()
        )

    def test_saved_model_preserves_scores(self):
        original_scores = (
            self.model.user_factors
            @ self.model.item_factors.T
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            self.model.save(str(path))

            with threadpool_limits(limits=1, user_api="blas"):
                restored = AlternatingLeastSquares.load(str(path))

            restored_scores = (
                restored.user_factors
                @ restored.item_factors.T
            )

            np.testing.assert_allclose(
                original_scores,
                restored_scores,
                rtol=1e-6,
                atol=1e-6,
            )

    def test_invalid_negative_confidence_is_rejected(self):
        for value in (0, -1, np.nan, np.inf, True):
            with self.subTest(value=value):
                settings = self.settings.copy()
                settings["negative_confidence"] = value

                with self.assertRaises(ValueError):
                    validate_settings(settings)


if __name__ == "__main__":
    unittest.main()
    