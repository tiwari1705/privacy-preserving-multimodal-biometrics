"""
Verification metrics.
"""

import numpy as np
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score
)


def compute_roc(labels, scores):
    """Compute ROC curve."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores
    )

    return fpr, tpr, thresholds


def compute_auc(labels, scores):
    """Compute ROC-AUC."""
    return roc_auc_score(labels, scores)


def compute_eer(labels, scores):
    """
    Compute Equal Error Rate.

    EER is estimated at the ROC operating point
    where FAR and FRR are closest.
    """
    fpr, tpr, thresholds = roc_curve(
        labels,
        scores
    )

    fnr = 1.0 - tpr

    index = np.nanargmin(
        np.abs(fpr - fnr)
    )

    eer = (fpr[index] + fnr[index]) / 2.0

    return float(eer)


def compute_accuracy(labels, scores, threshold=0.5):
    """Compute verification accuracy."""
    predictions = (np.asarray(scores) >= threshold).astype(int)

    return accuracy_score(
        labels,
        predictions
    )


def tar_at_far(labels, scores, target_far=0.01):
    """
    Estimate TAR at a specified FAR.
    """
    fpr, tpr, thresholds = roc_curve(
        labels,
        scores
    )

    valid = np.where(fpr <= target_far)[0]

    if len(valid) == 0:
        return 0.0

    return float(np.max(tpr[valid]))
