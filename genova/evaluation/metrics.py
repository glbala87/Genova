"""Evaluation metrics for Genova models.

Provides functions for computing MLM metrics (accuracy, perplexity),
classification metrics (AUROC, AUPRC), calibration, and regression
correlation -- all unified through :func:`compute_metrics`.

Example::

    from genova.evaluation.metrics import compute_metrics

    results = compute_metrics(predictions, targets, task_type="classification")
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# MLM metrics
# ---------------------------------------------------------------------------


def mlm_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    ignore_index: int = -100,
) -> float:
    """Compute masked-language-model accuracy.

    Args:
        predictions: Predicted token IDs, shape ``(N,)`` or ``(N, seq_len)``.
        targets: Ground-truth token IDs (same shape).  Positions set to
            *ignore_index* are excluded.
        ignore_index: Label value to ignore.

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    predictions = np.asarray(predictions).ravel()
    targets = np.asarray(targets).ravel()
    mask = targets != ignore_index
    if not mask.any():
        return 0.0
    return float((predictions[mask] == targets[mask]).mean())


def perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss.

    Args:
        loss: Average cross-entropy loss value.

    Returns:
        Perplexity (capped at ``exp(20)`` to avoid overflow).
    """
    return math.exp(min(loss, 20.0))


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def auroc(
    scores: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute Area Under the ROC Curve.

    Uses the trapezoidal rule over sorted thresholds (no sklearn dependency).

    Args:
        scores: Predicted probabilities or decision values, shape ``(N,)``.
        targets: Binary ground truth labels (0 or 1), shape ``(N,)``.

    Returns:
        AUROC value.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.int64).ravel()

    if len(np.unique(targets)) < 2:
        logger.warning("AUROC undefined with single-class targets; returning 0.0.")
        return 0.0

    # Sort descending by score
    desc = np.argsort(-scores)
    targets_sorted = targets[desc]

    n_pos = targets.sum()
    n_neg = len(targets) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    tpr_prev, fpr_prev = 0.0, 0.0
    tp, fp = 0, 0
    auc = 0.0

    for i in range(len(targets_sorted)):
        if targets_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += 0.5 * (tpr + tpr_prev) * (fpr - fpr_prev)
        tpr_prev, fpr_prev = tpr, fpr

    return float(auc)


def auprc(
    scores: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute Area Under the Precision-Recall Curve.

    Args:
        scores: Predicted probabilities, shape ``(N,)``.
        targets: Binary ground truth labels, shape ``(N,)``.

    Returns:
        AUPRC value.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.int64).ravel()

    if len(np.unique(targets)) < 2:
        logger.warning("AUPRC undefined with single-class targets; returning 0.0.")
        return 0.0

    desc = np.argsort(-scores)
    targets_sorted = targets[desc]

    n_pos = targets.sum()
    if n_pos == 0:
        return 0.0

    tp = 0
    precision_sum = 0.0

    for i in range(len(targets_sorted)):
        if targets_sorted[i] == 1:
            tp += 1
            precision_sum += tp / (i + 1)

    return float(precision_sum / n_pos)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(
    scores: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        scores: Predicted probabilities, shape ``(N,)``.
        targets: Binary ground truth labels, shape ``(N,)``.
        n_bins: Number of equal-width bins.

    Returns:
        ECE value (lower is better).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(scores)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores > lo) & (scores <= hi) if lo > 0 else (scores >= lo) & (scores <= hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_confidence = scores[mask].mean()
        avg_accuracy = targets[mask].mean()
        ece += (count / n) * abs(avg_confidence - avg_accuracy)

    return float(ece)


def brier_score(
    scores: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute the Brier score (mean squared error of probabilities).

    Args:
        scores: Predicted probabilities, shape ``(N,)``.
        targets: Binary ground truth labels, shape ``(N,)``.

    Returns:
        Brier score (lower is better).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()
    return float(np.mean((scores - targets) ** 2))


# ---------------------------------------------------------------------------
# Regression / correlation metrics
# ---------------------------------------------------------------------------


def pearson_correlation(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute Pearson correlation coefficient.

    Args:
        predictions: Predicted values, shape ``(N,)``.
        targets: Ground truth values, shape ``(N,)``.

    Returns:
        Pearson *r* value.
    """
    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()

    if len(predictions) < 2:
        return 0.0

    p_mean = predictions.mean()
    t_mean = targets.mean()
    num = ((predictions - p_mean) * (targets - t_mean)).sum()
    denom = (
        np.sqrt(((predictions - p_mean) ** 2).sum())
        * np.sqrt(((targets - t_mean) ** 2).sum())
    )
    if denom == 0:
        return 0.0
    return float(num / denom)


def spearman_correlation(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute Spearman rank correlation coefficient.

    Args:
        predictions: Predicted values, shape ``(N,)``.
        targets: Ground truth values, shape ``(N,)``.

    Returns:
        Spearman *rho* value.
    """

    def _rankdata(arr: np.ndarray) -> np.ndarray:
        order = arr.argsort()
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
        return ranks

    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()
    return pearson_correlation(_rankdata(predictions), _rankdata(targets))


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute mean squared error.

    Args:
        predictions: Predicted values, shape ``(N,)``.
        targets: Ground truth values, shape ``(N,)``.

    Returns:
        MSE value.
    """
    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()
    return float(np.mean((predictions - targets) ** 2))


# ---------------------------------------------------------------------------
# Unified compute_metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    task_type: str = "classification",
    *,
    loss: Optional[float] = None,
    ignore_index: int = -100,
    calibration_bins: int = 10,
) -> Dict[str, float]:
    """Compute a suite of metrics appropriate for the given task type.

    Args:
        predictions: Model outputs.  For classification, these should be
            predicted probabilities; for MLM, predicted token IDs; for
            regression, predicted continuous values.
        targets: Ground truth labels / values.
        task_type: One of ``"classification"``, ``"mlm"``, or
            ``"regression"``.
        loss: Optional pre-computed loss (used for perplexity in MLM).
        ignore_index: Label value to ignore (MLM only).
        calibration_bins: Number of bins for ECE computation.

    Returns:
        Dictionary of metric name to value.

    Raises:
        ValueError: If *task_type* is not recognised.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if task_type == "mlm":
        results: Dict[str, float] = {
            "accuracy": mlm_accuracy(predictions, targets, ignore_index=ignore_index),
        }
        if loss is not None:
            results["perplexity"] = perplexity(loss)
        return results

    if task_type == "classification":
        return {
            "auroc": auroc(predictions, targets),
            "auprc": auprc(predictions, targets),
            "ece": expected_calibration_error(
                predictions, targets, n_bins=calibration_bins
            ),
            "brier_score": brier_score(predictions, targets),
        }

    if task_type == "regression":
        return {
            "pearson_r": pearson_correlation(predictions, targets),
            "spearman_rho": spearman_correlation(predictions, targets),
            "mse": mse(predictions, targets),
        }

    raise ValueError(
        f"Unknown task_type {task_type!r}. "
        "Choose from 'classification', 'mlm', or 'regression'."
    )
