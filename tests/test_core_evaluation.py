import unittest

import numpy as np

from src.core.evaluate import (
    exclusion_columns,
    rank_scores,
)


class CoreEvaluationTests(unittest.TestCase):
    def test_excludes_requested_movies(self):
        ranked = rank_scores(
            scores=np.array([0.9, 0.8, 0.7]),
            movie_ids=np.array([30, 10, 20]),
            excluded_columns={0},
            k=2,
        )

        self.assertEqual(ranked, [10, 20])

    def test_equal_scores_use_movie_id_for_ties(self):
        ranked = rank_scores(
            scores=np.array([1.0, 1.0, 1.0]),
            movie_ids=np.array([30, 10, 20]),
            excluded_columns=set(),
            k=2,
        )

        self.assertEqual(ranked, [10, 20])

    def test_held_out_movies_remain_eligible(self):
        excluded = exclusion_columns(
            observed_movie_ids={10, 20, 30, 99},
            validation_columns=[1],
            movie_to_column={10: 0, 20: 1, 30: 2},
        )

        self.assertEqual(excluded, {0, 2})

    def test_handles_fewer_candidates_than_k(self):
        ranked = rank_scores(
            scores=np.array([0.9, 0.8]),
            movie_ids=np.array([10, 20]),
            excluded_columns={0},
            k=10,
        )

        self.assertEqual(ranked, [20])

    def test_handles_all_movies_excluded(self):
        ranked = rank_scores(
            scores=np.array([0.9, 0.8]),
            movie_ids=np.array([10, 20]),
            excluded_columns={0, 1},
            k=10,
        )

        self.assertEqual(ranked, [])

    def test_rejects_invalid_scores(self):
        with self.assertRaises(ValueError):
            rank_scores(
                scores=np.array([np.nan]),
                movie_ids=np.array([10]),
                excluded_columns=set(),
                k=1,
            )


if __name__ == "__main__":
    unittest.main()