import unittest

import numpy as np

from src.content.metadata import metadata_text
from src.content.train_mapping import fit_mapping


class ContentMappingTests(unittest.TestCase):
    def test_genre_order_and_duplicates_do_not_change_text(self):
        first = metadata_text({
            "title": "Example",
            "genres": "Comedy|Action|Comedy",
        })

        second = metadata_text({
            "title": "Example",
            "genres": ["Action", "Comedy"],
        })

        self.assertEqual(first, second)

    def test_missing_genres_are_allowed(self):
        self.assertEqual(
            metadata_text({"title": "Example"}),
            "Title: Example",
        )

    def test_empty_title_is_rejected(self):
        with self.assertRaises(ValueError):
            metadata_text({"title": " "})

    def test_invalid_genre_values_are_rejected(self):
        with self.assertRaises(ValueError):
            metadata_text({
                "title": "Example",
                "genres": ["Comedy", 123],
            })

    def test_exported_coefficients_preserve_predictions(self):
        rng = np.random.default_rng(42)

        x = rng.normal(size=(24, 4))
        true_weights = rng.normal(size=(4, 2))
        true_bias = np.array([0.2, -0.1])

        y = x @ true_weights + true_bias

        mapping = fit_mapping(x, y, alpha=0.1)

        weights = mapping.coef_.T.astype(np.float32)
        bias = np.asarray(mapping.intercept_, dtype=np.float32)

        np.testing.assert_allclose(
            x @ weights + bias,
            mapping.predict(x),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_mapping_output_has_target_dimensions(self):
        rng = np.random.default_rng(42)

        x = rng.normal(size=(12, 4))
        y = rng.normal(size=(12, 2))

        mapping = fit_mapping(x, y, alpha=1.0)

        self.assertEqual(
            mapping.predict(x[:3]).shape,
            (3, 2),
        )

    def test_invalid_mapping_inputs_are_rejected(self):
        x = np.ones((4, 3))
        y = np.ones((4, 2))

        with self.assertRaises(ValueError):
            fit_mapping(x, y[:2], alpha=1.0)

        with self.assertRaises(ValueError):
            fit_mapping(x, y, alpha=0)

        bad = x.copy()
        bad[0, 0] = np.nan

        with self.assertRaises(ValueError):
            fit_mapping(bad, y, alpha=1.0)


if __name__ == "__main__":
    unittest.main()