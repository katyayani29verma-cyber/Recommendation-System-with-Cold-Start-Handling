import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from src.core.als_core import ALSRecommenderCore
from src.core.hybrid_core import ContentEncoder, HybridRecommenderCore


class HybridCoreTests(unittest.TestCase):
    def setUp(self):
        self.als = ALSRecommenderCore(
            movie_ids=np.array([10, 20]),
            item_vectors=np.array(
                [[2.0, 0.0], [99.0, 99.0]],
                dtype=np.float32,
            ),
            available_mask=np.array([True, False]),
            prior_mean=np.array([1.0, 0.0], dtype=np.float32),
            prior_covariance=np.eye(2),
        )

        # Deliberately use an order different from the ALS catalog.
        self.ids = np.array([30, 10, 20])

        self.predictions = np.array(
            [[3.0, 0.0], [-5.0, -5.0], [0.0, 4.0]],
            dtype=np.float32,
        )

    def make_core(self, content_encoder=None):
        return HybridRecommenderCore(
            self.als,
            self.ids,
            self.predictions,
            artifact_version="test",
            content_encoder=content_encoder,
        )

    def test_supported_als_vector_is_preserved_by_movie_id(self):
        core = self.make_core()

        np.testing.assert_array_equal(
            core.get_item_vector(10),
            [2.0, 0.0],
        )

        self.assertEqual(core.vector_source(10), "als")

    def test_unsupported_als_row_uses_content(self):
        core = self.make_core()

        np.testing.assert_array_equal(
            core.get_item_vector(20),
            [0.0, 4.0],
        )

        self.assertEqual(core.vector_source(20), "content")

    def test_movie_absent_from_als_can_be_recommended(self):
        core = self.make_core()

        recommendations = core.recommend(
            np.array([1.0, 0.0]),
            k=3,
        )

        self.assertEqual(
            [result.movie_id for result in recommendations],
            [30, 10, 20],
        )

    def test_exclusions_apply_to_content_movies(self):
        core = self.make_core()

        recommendations = core.recommend(
            np.array([1.0, 0.0]),
            k=3,
            exclude_ids={30},
        )

        self.assertEqual(
            [result.movie_id for result in recommendations],
            [10, 20],
        )

    def test_inputs_and_prior_are_preserved(self):
        original = self.predictions.copy()
        core = self.make_core()

        np.testing.assert_array_equal(self.predictions, original)
        np.testing.assert_array_equal(
            core.prior_mean,
            self.als.prior_mean,
        )
        np.testing.assert_array_equal(
            core.prior_covariance,
            self.als.prior_covariance,
        )

    def test_missing_supported_movie_is_rejected(self):
        with self.assertRaises(ValueError):
            HybridRecommenderCore(
                self.als,
                np.array([20, 30]),
                np.zeros((2, 2), dtype=np.float32),
                artifact_version="test",
            )

    def test_wrong_prediction_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            HybridRecommenderCore(
                self.als,
                self.ids,
                np.zeros((3, 3), dtype=np.float32),
                artifact_version="test",
            )

    def test_new_movie_uses_formatted_text_and_saved_mapping(self):
        encoder = ContentEncoder(
            Path("unused-in-this-test"),
            weights=np.array([[2.0, 0.0], [0.0, 3.0]]),
            bias=np.array([0.1, 0.2]),
        )

        # The unit test replaces only the expensive text encoder.
        encoder._encoder = Mock()
        encoder._encoder.encode.return_value = np.array([[1.0, 0.0]])

        core = self.make_core(content_encoder=encoder)

        result = core.embed_new_item({
            "title": "Example",
            "genres": ["Thriller", "Sci-Fi"],
        })

        np.testing.assert_allclose(result, [2.1, 0.2])

        encoder._encoder.encode.assert_called_once_with(
            ["Title: Example\nGenres: Sci-Fi, Thriller"],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def test_invalid_text_embedding_is_rejected(self):
        encoder = ContentEncoder(
            Path("unused-in-this-test"),
            weights=np.eye(2),
            bias=np.zeros(2),
        )

        with self.assertRaises(ValueError):
            encoder.project(np.array([np.nan, 0.0]))

        with self.assertRaises(ValueError):
            encoder.project(np.zeros(3))


if __name__ == "__main__":
    unittest.main()