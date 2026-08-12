"""
Cancellable biometric template transformation.

The transformation applies a random projection followed by
an optional sign/binary operation.
"""

import numpy as np


def create_projection_matrix(
    input_dim,
    output_dim,
    seed=42
):
    """
    Create a reproducible Gaussian random projection matrix.
    """
    rng = np.random.default_rng(seed)

    return rng.normal(
        loc=0.0,
        scale=1.0,
        size=(output_dim, input_dim)
    )


def project_embedding(embedding, projection_matrix):
    """Project an embedding into the protected space."""
    embedding = np.asarray(embedding, dtype=np.float64)

    return projection_matrix @ embedding


def binary_transform(projected):
    """Convert projected values into {-1,+1}."""
    projected = np.asarray(projected)

    return np.where(projected >= 0, 1.0, -1.0)


def cancellable_transform(
    embedding,
    projection_matrix,
    binary=False
):
    """
    Generate a cancellable template.

    binary=False:
        protected real-valued template

    binary=True:
        {-1,+1} protected template
    """
    projected = project_embedding(
        embedding,
        projection_matrix
    )

    if binary:
        return binary_transform(projected)

    return projected
