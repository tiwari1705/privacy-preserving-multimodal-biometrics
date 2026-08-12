"""
Similarity and verification utilities.
"""

import numpy as np


def l2_normalize(x, axis=-1, eps=1e-12):
    """L2-normalize an embedding or embedding matrix."""
    x = np.asarray(x, dtype=np.float64)

    norm = np.linalg.norm(x, axis=axis, keepdims=True)

    return x / np.maximum(norm, eps)


def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings."""
    a = l2_normalize(a)
    b = l2_normalize(b)

    return float(np.dot(a, b))


def pairwise_cosine(A, B):
    """Compute pairwise cosine similarities."""
    A = l2_normalize(A)
    B = l2_normalize(B)

    return A @ B.T


def minmax_normalize(scores, minimum, maximum):
    """Min-max normalize scores using supplied reference bounds."""
    scores = np.asarray(scores, dtype=np.float64)

    denominator = maximum - minimum

    if denominator == 0:
        return np.zeros_like(scores)

    return (scores - minimum) / denominator


def weighted_fusion(scores, weights):
    """
    Weighted score-level fusion.

    scores: dictionary containing modality scores.
    weights: dictionary containing modality weights.
    """
    fused = 0.0

    for modality in scores:
        fused += weights[modality] * scores[modality]

    return fused
