"""Tests for Track A preprocessing."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from src.core.data_prep import (
    file_sha256,
    prepare_interactions,
    save_prepared,
)


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.ratings = pd.DataFrame({
            "userId": [30, 10, 10, 20],
            "movieId": [200, 100, 200, 300],
            "rating": [4.0, 5.0, 2.0, 1.0],
        })

    def prepare(
        self,
        ratings=None,
        threshold=4.0,
        evaluation_ids=None,
        catalog=None,
    ):
        return prepare_interactions(
            self.ratings if ratings is None else ratings,
            positive_threshold=threshold,
            evaluation_user_ids=(
                set() if evaluation_ids is None else evaluation_ids
            ),
            catalog_movie_ids=(
                {100, 200, 300} if catalog is None else catalog
            ),
        )

    def test_exact_matrix_and_id_round_trip(self):
        result = self.prepare()

        np.testing.assert_array_equal(
            result.user_ids,
            [10, 20, 30],
        )
        np.testing.assert_array_equal(
            result.movie_ids,
            [100, 200],
        )
        np.testing.assert_array_equal(
            result.matrix.toarray(),
            [[1, 0], [0, 0], [0, 1]],
        )

        self.assertEqual(
            result.matrix.dtype,
            np.dtype("float32"),
        )

        for user_id, row in result.user_to_row.items():
            self.assertEqual(
                int(result.user_ids[row]),
                user_id,
            )

        for movie_id, column in result.movie_to_column.items():
            self.assertEqual(
                int(result.movie_ids[column]),
                movie_id,
            )

    def test_row_order_does_not_change_output(self):
        first = self.prepare()
        shuffled = self.ratings.sample(
            frac=1,
            random_state=7,
        )
        second = self.prepare(shuffled)

        np.testing.assert_array_equal(
            first.user_ids,
            second.user_ids,
        )
        np.testing.assert_array_equal(
            first.movie_ids,
            second.movie_ids,
        )
        self.assertEqual(
            (first.matrix != second.matrix).nnz,
            0,
        )

    def test_threshold_is_inclusive_and_configurable(self):
        self.assertEqual(
            self.prepare(threshold=4.0).matrix.nnz,
            2,
        )
        self.assertEqual(
            self.prepare(threshold=4.5).matrix.nnz,
            1,
        )

    def test_rejects_duplicates(self):
        duplicate = pd.concat(
            [self.ratings, self.ratings.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.prepare(duplicate)

    def test_rejects_invalid_records(self):
        cases = [
            ("rating", np.nan),
            ("rating", np.inf),
            ("rating", 6.0),
            ("rating", "invalid"),
            ("userId", -1),
            ("movieId", 1.5),
            ("rating", True),
            ("userId", True),
        ]

        for column, value in cases:
            with self.subTest(column=column, value=value):
                bad = self.ratings.astype(object).copy()
                bad.loc[0, column] = value

                with self.assertRaises(ValueError):
                    self.prepare(bad)

    def test_rejects_missing_columns_empty_data_and_no_positives(self):
        cases = [
            self.ratings.drop(columns="rating"),
            self.ratings.iloc[:0],
            self.ratings.assign(rating=1.0),
        ]

        for bad in cases:
            with self.subTest(shape=bad.shape):
                with self.assertRaises(ValueError):
                    self.prepare(bad)

    def test_rejects_evaluation_users_even_in_low_ratings(self):
        with self.assertRaisesRegex(ValueError, "Evaluation users"):
            self.prepare(evaluation_ids={20})

    def test_rejects_unknown_movies(self):
        with self.assertRaisesRegex(
            ValueError,
            "missing from the catalog",
        ):
            self.prepare(catalog={100, 200})

    def test_rejects_invalid_thresholds(self):
        for threshold in [True, np.nan, np.inf, 0.0, 6.0, "invalid"]:
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    self.prepare(threshold=threshold)

    def test_does_not_modify_original_table(self):
        original = self.ratings.copy(deep=True)
        self.prepare()

        pd.testing.assert_frame_equal(
            self.ratings,
            original,
        )

    def test_saved_files_reload_and_match_checksums(self):
        result = self.prepare()

        with TemporaryDirectory() as directory:
            output = Path(directory)

            save_prepared(
                result,
                output,
                provenance={"dataset": "test"},
            )

            loaded_matrix = load_npz(
                output / "interactions.npz"
            )
            self.assertEqual(
                (result.matrix != loaded_matrix).nnz,
                0,
            )

            np.testing.assert_array_equal(
                np.load(output / "user_ids.npy"),
                result.user_ids,
            )
            np.testing.assert_array_equal(
                np.load(output / "movie_ids.npy"),
                result.movie_ids,
            )

            manifest = json.loads(
                (output / "manifest.json").read_text()
            )

            self.assertEqual(
                manifest["shape"],
                [3, 2],
            )
            self.assertEqual(
                manifest["users_without_positive_interactions"],
                1,
            )

            for filename, checksum in manifest["file_sha256"].items():
                self.assertEqual(
                    file_sha256(output / filename),
                    checksum,
                )


if __name__ == "__main__":
    unittest.main()