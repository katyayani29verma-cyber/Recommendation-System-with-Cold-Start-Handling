"""Combine learned ALS vectors with metadata-based movie vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from threading import Lock

import numpy as np

from src.content.metadata import FORMAT_VERSION, metadata_text
from src.core.als_core import ALSRecommenderCore, load_core
from src.core.data_prep import file_sha256
from src.data.config import REPO_ROOT


def resolve_directory(value: str | Path) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = REPO_ROOT / path

    return path.resolve()


def validate_ids(values: np.ndarray) -> np.ndarray:
    ids = np.asarray(values)

    if (
        ids.ndim != 1
        or len(ids) == 0
        or not np.issubdtype(ids.dtype, np.integer)
        or np.any(ids < 1)
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("Movie IDs must be unique positive integers.")

    return ids


class ContentEncoder:
    """Convert movie metadata into the saved ALS coordinate system."""

    def __init__(
        self,
        encoder_directory: Path,
        weights: np.ndarray,
        bias: np.ndarray,
    ):
        if np.iscomplexobj(weights) or np.iscomplexobj(bias):
            raise ValueError("Mapping parameters must be real-valued.")

        self.weights = np.array(weights, dtype=np.float64, copy=True)
        self.bias = np.array(bias, dtype=np.float64, copy=True)

        if (
            self.weights.ndim != 2
            or min(self.weights.shape) < 1
            or self.bias.shape != (self.weights.shape[1],)
        ):
            raise ValueError("Invalid mapping parameter dimensions.")

        if (
            not np.isfinite(self.weights).all()
            or not np.isfinite(self.bias).all()
        ):
            raise ValueError("Mapping parameters must be finite.")

        self.weights.setflags(write=False)
        self.bias.setflags(write=False)

        self.encoder_directory = Path(encoder_directory)
        self._encoder = None
        self._lock = Lock()

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """Apply the same mapping calculation used during training."""
        if np.iscomplexobj(embedding):
            raise ValueError("Text embeddings must be real-valued.")

        embedding = np.asarray(embedding, dtype=np.float64)

        if embedding.shape != (self.weights.shape[0],):
            raise ValueError("Text embedding has the wrong dimensions.")

        if not np.isfinite(embedding).all():
            raise ValueError("Text embedding contains invalid numbers.")

        vector = (
            embedding @ self.weights + self.bias
        ).astype(np.float32)

        if not np.isfinite(vector).all():
            raise ValueError("Projected vector contains invalid numbers.")

        return vector

    def __call__(self, metadata: dict) -> np.ndarray:
        text = metadata_text(metadata)

        # Load the text model only when a new description needs encoding.
        with self._lock:
            if self._encoder is None:
                import torch
                from sentence_transformers import SentenceTransformer

                torch.set_num_threads(4)

                self._encoder = SentenceTransformer(
                    str(self.encoder_directory),
                    device="cpu",
                    local_files_only=True,
                    token=False,
                    trust_remote_code=False,
                )

            embedding = self._encoder.encode(
                [text],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]

        return self.project(embedding)


class HybridRecommenderCore(ALSRecommenderCore):
    """Use supported ALS vectors first, then content predictions."""

    def __init__(
        self,
        als_core: ALSRecommenderCore,
        catalog_ids: np.ndarray,
        predictions: np.ndarray,
        *,
        artifact_version: str,
        content_encoder=None,
    ):
        ids = validate_ids(catalog_ids)
        vectors = np.asarray(predictions)

        if (
            vectors.shape != (len(ids), als_core.n_dims)
            or not np.issubdtype(vectors.dtype, np.floating)
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("Invalid catalog prediction vectors.")

        movie_to_row = {
            int(movie_id): row
            for row, movie_id in enumerate(ids)
        }

        supported_ids = frozenset(als_core.available_movie_ids)

        if not supported_ids.issubset(movie_to_row):
            raise ValueError("The catalog is missing supported ALS movies.")

        # Start with metadata predictions, then preserve supported ALS rows.
        combined = vectors.astype(np.float32, copy=True)

        for movie_id in supported_ids:
            combined[movie_to_row[movie_id]] = (
                als_core.get_item_vector(movie_id)
            )

        super().__init__(
            movie_ids=ids,
            item_vectors=combined,
            artifact_version=artifact_version,
            prior_mean=als_core.prior_mean,
            prior_covariance=als_core.prior_covariance,
            content_encoder=content_encoder,
        )

        self.als_movie_ids = supported_ids

    def vector_source(self, movie_id: int) -> str:
        movie_id = self._validate_movie_id(movie_id)

        if not self.has_item(movie_id):
            raise KeyError(f"Unknown movie ID: {movie_id}")

        return "als" if movie_id in self.als_movie_ids else "content"


def load_hybrid_core(
    als_artifact: str | Path,
    content_artifact: str | Path,
) -> HybridRecommenderCore:
    als_directory = resolve_directory(als_artifact)
    content_directory = resolve_directory(content_artifact)

    als_core = load_core(als_directory)

    with (content_directory / "manifest.json").open(
        "r", encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)

    if (
        manifest.get("schema_version") != 1
        or manifest.get("stage") != "content_mapping"
    ):
        raise ValueError("Unsupported content artifact.")

    if manifest.get("artifact_version") != content_directory.name:
        raise ValueError("Content artifact version does not match its folder.")

    # Equal dimensions alone do not mean two models use the same space.
    if (
        manifest.get("source_als_version") != als_core.artifact_version
        or manifest.get("source_als_manifest_sha256")
        != file_sha256(als_directory / "manifest.json")
    ):
        raise ValueError(
            "This content mapping belongs to a different ALS model."
        )

    if manifest.get("text_format_version") != FORMAT_VERSION:
        raise ValueError("Movie text formatting differs from training.")

    if (
        manifest.get("normalize_text_embeddings") is not True
        or manifest.get("normalize_latent_vectors") is not False
    ):
        raise ValueError("Unexpected vector normalization settings.")

    if manifest.get("latent_dimension") != als_core.n_dims:
        raise ValueError("ALS and content dimensions do not match.")

    checksums = manifest.get("file_sha256", {})

    required_files = {
        "mapping_weights.npy",
        "mapping_bias.npy",
        "catalog_movie_ids.npy",
        "catalog_predictions.npy",
        "warm_movie_ids.npy",
        "mapping_validation_movie_ids.npy",
        "encoder/modules.json",
        "encoder/config.json",
        "encoder/model.safetensors",
    }

    if not required_files.issubset(checksums):
        raise ValueError("Content manifest is missing required file hashes.")

    encoder_directory = content_directory / "encoder"

    encoder_files = {
        path.relative_to(content_directory).as_posix()
        for path in encoder_directory.rglob("*")
        if path.is_file()
    }

    if not encoder_files.issubset(checksums):
        raise ValueError("The encoder contains files absent from its manifest.")

    for filename, expected_hash in checksums.items():
        path = (content_directory / filename).resolve()

        if not path.is_relative_to(content_directory):
            raise ValueError("A file path escapes the artifact directory.")

        if not path.is_file():
            raise FileNotFoundError(path)

        if file_sha256(path) != expected_hash:
            raise ValueError(f"Artifact checksum mismatch: {filename}")

    def read_array(filename: str) -> np.ndarray:
        return np.load(
            content_directory / filename,
            allow_pickle=False,
        )

    ids = validate_ids(read_array("catalog_movie_ids.npy"))
    predictions = read_array("catalog_predictions.npy")
    warm_ids = validate_ids(read_array("warm_movie_ids.npy"))
    weights = read_array("mapping_weights.npy")
    bias = read_array("mapping_bias.npy")

    if len(ids) != manifest["catalog_movie_count"]:
        raise ValueError("Catalog size differs from the manifest.")

    if (
        len(warm_ids) != manifest["mapping_training_movies"]
        or not np.array_equal(
            np.sort(warm_ids),
            np.array(sorted(als_core.available_movie_ids)),
        )
    ):
        raise ValueError("Mapping targets differ from supported ALS movies.")

    if weights.shape != (
        manifest["embedding_dimension"],
        als_core.n_dims,
    ):
        raise ValueError("Mapping dimensions differ from the manifest.")

    content_encoder = ContentEncoder(
        encoder_directory,
        weights,
        bias,
    )

    return HybridRecommenderCore(
        als_core,
        ids,
        predictions,
        artifact_version=(
            f"{als_core.artifact_version}+{manifest['artifact_version']}"
        ),
        content_encoder=content_encoder,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--als-artifact", required=True)
    parser.add_argument("--content-artifact", required=True)
    args = parser.parse_args()

    core = load_hybrid_core(
        args.als_artifact,
        args.content_artifact,
    )

    als_count = len(core.als_movie_ids)
    catalog_count = len(core.available_movie_ids)

    print(f"Loaded version: {core.artifact_version}")
    print(f"Latent dimensions: {core.n_dims}")
    print(f"Scorable movies: {catalog_count:,}")
    print(f"Movies using ALS vectors: {als_count:,}")
    print(f"Movies using content vectors: {catalog_count - als_count:,}")

    new_vector = core.embed_new_item({
        "title": "The Last Signal (2026)",
        "genres": ["Sci-Fi", "Thriller"],
    })

    print(f"\nExample new-movie vector shape: {new_vector.shape}")
    print(f"New-movie vector is finite: {np.isfinite(new_vector).all()}")

    print("\nRecommendations for the average training preference:")

    for recommendation in core.recommend(core.prior_mean, k=10):
        source = core.vector_source(recommendation.movie_id)

        print(
            f"Movie {recommendation.movie_id}: "
            f"{recommendation.score:.6f} "
            f"[{source}]"
        )


if __name__ == "__main__":
    main()