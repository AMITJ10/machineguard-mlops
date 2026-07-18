from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    y_true: Any,
    probabilities: np.ndarray,
    threshold: float = 0.50,
    false_negative_cost: float = 100_000.0,
    false_positive_cost: float = 5_000.0,
) -> dict[str, Any]:
    """Calculate classification and business metrics.

    Args:
        y_true: Ground-truth binary labels.
        probabilities: Predicted probability of machine failure.
        threshold: Probability threshold for classification.
        false_negative_cost: Business cost of a missed failure.
        false_positive_cost: Business cost of an unnecessary inspection.

    Returns:
        Dictionary containing model and business metrics.
    """
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1:
        raise ValueError("Probabilities must be a one-dimensional array.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Classification threshold must be between 0 and 1.")

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    business_cost = fn * false_negative_cost + fp * false_positive_cost

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "business_cost": float(business_cost),
        "threshold": float(threshold),
    }

    return metrics
