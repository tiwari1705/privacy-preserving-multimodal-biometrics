"""
Embedding extraction interface.

The actual pretrained model checkpoints and dataset files
are intentionally not included in this repository.
"""

import numpy as np


def validate_embedding(
    embedding,
    expected_dimension=512
):
    """
    Validate the dimensionality of a base embedding.
    """
    embedding = np.asarray(
        embedding,
        dtype=np.float64
    )

    if embedding.ndim != 1:
        raise ValueError(
            "Embedding must be a one-dimensional vector."
        )

    if embedding.shape[0] != expected_dimension:
        raise ValueError(
            f"Expected {expected_dimension}-D embedding, "
            f"received {embedding.shape[0]}-D."
        )

    return embedding


def normalize_embedding(embedding):
    """L2-normalize a base embedding."""
    embedding = np.asarray(
        embedding,
        dtype=np.float64
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


def prepare_embedding(embedding):
    """
    Validate and normalize a 512-D embedding.
    """
    embedding = validate_embedding(
        embedding,
        expected_dimension=512
    )

    return normalize_embedding(embedding)
