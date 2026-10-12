import unittest

import numpy as np

from src.core.als_core import ALSRecommenderCore


class ALSCoreTests(unittest.TestCase):
    def setUp(self):
        self.core = ALSRecommenderCore(
            movie_ids=np.array([30, 10, 20]),
            item_vectors=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.5, 0.5],
                ],
                dtype=np.float32,
            ),
        )

    def test_ranks_by_dot_product(self):
        results = self.core.recommend(
            np.array([0.8, 0.2]),
            k=3,
        )

        self.assertEqual(
            [result.movie_id for result in results],
            [30, 20, 10],
        )

        np.testing.assert_allclose(
            [result.score for result in results],
            [0.8, 0.5, 0.2],
        )

    def test_respects_exclusions(self):
        results = self.core.recommend(
            np.array([0.8, 0.2]),
            k=3,
            exclude_ids={30, 999},
        )

        self.assertEqual(
            [result.movie_id for result in results],
            [20, 10],
        )

    def test_boundary_ties_use_movie_id(self):
        results = self.core.recommend(
            np.array([0.0, 0.0]),
            k=2,
        )

        self.assertEqual(
            [result.movie_id for result in results],
            [10, 20],
        )

    def test_handles_k_larger_than_catalog(self):
        results = self.core.recommend(
            np.array([0.8, 0.2]),
            k=100,
        )

        self.assertEqual(len(results), 3)

    def test_handles_all_movies_excluded(self):
        results = self.core.recommend(
            np.array([0.8, 0.2]),
            exclude_ids={10, 20, 30},
        )

        self.assertEqual(results, [])

    def test_unavailable_movies_are_not_ranked(self):
        core = ALSRecommenderCore(
            movie_ids=np.array([10, 20]),
            item_vectors=np.array(
                [[1.0, 0.0], [0.0, 0.0]],
                dtype=np.float32,
            ),
            available_mask=np.array([True, False]),
        )

        results = core.recommend(
            np.array([1.0, 0.0]),
            k=10,
        )

        self.assertEqual(
            [result.movie_id for result in results],
            [10],
        )

        with self.assertRaises(KeyError):
            core.get_item_vector(20)

    def test_returned_vector_does_not_modify_model(self):
        vector = self.core.get_item_vector(30)
        vector[0] = 99

        np.testing.assert_array_equal(
            self.core.get_item_vector(30),
            [1.0, 0.0],
        )

    def test_unknown_id_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.core.get_item_vector(999)

    def test_rejects_invalid_user_vectors(self):
        for vector in (
            [1.0],
            [1.0, 2.0, 3.0],
            [np.nan, 0.0],
            [np.inf, 0.0],
        ):
            with self.subTest(vector=vector):
                with self.assertRaises(ValueError):
                    self.core.recommend(np.array(vector))

    def test_rejects_invalid_k(self):
        for k in (0, -1, True, 1.5):
            with self.subTest(k=k):
                with self.assertRaises(ValueError):
                    self.core.recommend(
                        np.array([0.8, 0.2]),
                        k=k,
                    )

    def test_content_embedding_requires_encoder(self):
        with self.assertRaises(RuntimeError):
            self.core.embed_new_item({"title": "Example"})

    def test_content_encoder_output_is_checked(self):
        core = ALSRecommenderCore(
            movie_ids=np.array([10]),
            item_vectors=np.array(
                [[1.0, 0.0]],
                dtype=np.float32,
            ),
            content_encoder=lambda metadata: np.array([0.2, 0.3]),
        )

        np.testing.assert_allclose(
            core.embed_new_item({"title": "Example"}),
            [0.2, 0.3],
        )


if __name__ == "__main__":
    unittest.main()