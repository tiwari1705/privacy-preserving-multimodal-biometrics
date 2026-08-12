"""
Score-level multimodal fusion.
"""

import numpy as np


FINAL_WEIGHTS = {
    "face": 0.35,
    "fingerprint": 0.25,
    "iris": 0.40
}


def minmax_score(scores, minimum, maximum):
    denominator = maximum - minimum

    if denominator == 0:
        return 0.0

    return (scores - minimum) / denominator


def fuse_scores(
    face_score,
    fingerprint_score,
    iris_score,
    weights=FINAL_WEIGHTS
):
    """Fuse normalized modality scores."""

    return (
        weights["face"] * face_score
        + weights["fingerprint"] * fingerprint_score
        + weights["iris"] * iris_score
    )


def batch_fusion(
    face_scores,
    fingerprint_scores,
    iris_scores,
    weights=FINAL_WEIGHTS
):
    """Fuse arrays of modality scores."""

    face_scores = np.asarray(face_scores)
    fingerprint_scores = np.asarray(fingerprint_scores)
    iris_scores = np.asarray(iris_scores)

    return (
        weights["face"] * face_scores
        + weights["fingerprint"] * fingerprint_scores
        + weights["iris"] * iris_scores
    )
