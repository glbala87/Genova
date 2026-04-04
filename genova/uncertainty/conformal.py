"""Conformal prediction for distribution-free uncertainty quantification.

Provides split conformal prediction for classification and conformalized
quantile regression for continuous outputs, offering guaranteed coverage
at user-specified confidence levels.

Example::

    from genova.uncertainty.conformal import ConformalPredictor, ConformalRegressor

    # Classification
    cp = ConformalPredictor()
    cp.calibrate(cal_scores, cal_labels, alpha=0.1)
    prediction_sets = cp.predict_set(test_scores)

    # Regression
    cr = ConformalRegressor()
    cr.calibrate(cal_residuals, alpha=0.1)
    lower, upper = cr.predict_interval(test_predictions)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# ConformalPredictor (classification)
# ---------------------------------------------------------------------------


class ConformalPredictor:
    """Split conformal prediction for classification tasks.

    Constructs prediction sets with guaranteed marginal coverage at the
    ``1 - alpha`` confidence level, without distributional assumptions.

    The calibration step computes nonconformity scores on a held-out
    calibration set.  At prediction time, all classes whose softmax
    probability exceeds a calibrated threshold are included in the
    prediction set.

    Args:
        score_type: Nonconformity score function.
            - ``"softmax"``: ``1 - softmax(y_true)`` (default).
            - ``"adaptive"`` (APS): cumulative softmax scores.
    """

    VALID_SCORE_TYPES = ("softmax", "adaptive")

    def __init__(self, score_type: str = "softmax") -> None:
        if score_type not in self.VALID_SCORE_TYPES:
            raise ValueError(
                f"score_type must be one of {self.VALID_SCORE_TYPES}, "
                f"got {score_type!r}"
            )
        self.score_type = score_type
        self._quantile: Optional[float] = None
        self._alpha: float = 0.1
        self._calibrated: bool = False

    @property
    def is_calibrated(self) -> bool:
        """Whether the predictor has been calibrated."""
        return self._calibrated

    @property
    def alpha(self) -> float:
        """Miscoverage level."""
        return self._alpha

    @property
    def quantile_threshold(self) -> Optional[float]:
        """Calibrated quantile threshold."""
        return self._quantile

    # ------------------------------------------------------------------
    # Nonconformity scores
    # ------------------------------------------------------------------

    def _compute_scores(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Compute nonconformity scores for calibration samples.

        Args:
            probabilities: ``(N, C)`` softmax probabilities.
            labels: ``(N,)`` integer true labels.

        Returns:
            ``(N,)`` nonconformity scores.
        """
        n = len(labels)
        labels = labels.astype(int)

        if self.score_type == "softmax":
            # Score = 1 - P(y_true)
            true_probs = probabilities[np.arange(n), labels]
            return 1.0 - true_probs

        elif self.score_type == "adaptive":
            # Adaptive Prediction Sets (APS)
            # Sort probabilities in descending order, accumulate until
            # the true class is included
            scores = np.zeros(n)
            for i in range(n):
                sorted_indices = np.argsort(probabilities[i])[::-1]
                cumsum = 0.0
                for j, idx in enumerate(sorted_indices):
                    cumsum += probabilities[i, idx]
                    if idx == labels[i]:
                        scores[i] = cumsum
                        break
            return scores

        raise ValueError(f"Unknown score_type: {self.score_type}")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        cal_scores: np.ndarray,
        cal_labels: np.ndarray,
        alpha: float = 0.1,
    ) -> float:
        """Calibrate the conformal predictor on a held-out calibration set.

        Computes the ``ceil((n+1)(1-alpha))/n`` quantile of the
        nonconformity scores to set the prediction threshold.

        Args:
            cal_scores: ``(N, C)`` softmax probabilities on the
                calibration set.
            cal_labels: ``(N,)`` true integer labels.
            alpha: Desired miscoverage rate (e.g. 0.1 for 90% coverage).

        Returns:
            The calibrated quantile threshold.
        """
        cal_scores = np.asarray(cal_scores, dtype=np.float64)
        cal_labels = np.asarray(cal_labels).ravel()
        self._alpha = alpha

        if cal_scores.ndim == 1:
            # Binary case: convert to two-class
            cal_scores = np.stack([1 - cal_scores, cal_scores], axis=-1)

        n = len(cal_labels)
        if n == 0:
            raise ValueError("Calibration set is empty.")

        nonconformity = self._compute_scores(cal_scores, cal_labels)

        # Quantile level with finite-sample correction
        quantile_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        self._quantile = float(np.quantile(nonconformity, quantile_level))

        self._calibrated = True
        logger.info(
            "ConformalPredictor calibrated: alpha={}, n_cal={}, "
            "quantile={:.4f}, score_type={}",
            alpha,
            n,
            self._quantile,
            self.score_type,
        )
        return self._quantile

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_set(
        self,
        scores: np.ndarray,
    ) -> List[List[int]]:
        """Construct prediction sets for new samples.

        Args:
            scores: ``(N, C)`` softmax probabilities for test samples.

        Returns:
            List of lists, where each inner list contains the class indices
            included in the prediction set for that sample.

        Raises:
            RuntimeError: If the predictor has not been calibrated.
        """
        if not self._calibrated or self._quantile is None:
            raise RuntimeError(
                "Predictor not calibrated. Call calibrate() first."
            )

        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim == 1:
            scores = np.stack([1 - scores, scores], axis=-1)

        n, c = scores.shape
        prediction_sets: List[List[int]] = []

        if self.score_type == "softmax":
            # Include all classes with P(y) >= 1 - quantile
            threshold = 1.0 - self._quantile
            for i in range(n):
                pset = [j for j in range(c) if scores[i, j] >= threshold]
                # Guarantee at least one class
                if not pset:
                    pset = [int(np.argmax(scores[i]))]
                prediction_sets.append(pset)

        elif self.score_type == "adaptive":
            for i in range(n):
                sorted_indices = np.argsort(scores[i])[::-1]
                cumsum = 0.0
                pset: List[int] = []
                for idx in sorted_indices:
                    cumsum += scores[i, idx]
                    pset.append(int(idx))
                    if cumsum >= self._quantile:
                        break
                if not pset:
                    pset = [int(np.argmax(scores[i]))]
                prediction_sets.append(pset)

        return prediction_sets

    def evaluate_coverage(
        self,
        test_scores: np.ndarray,
        test_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate the empirical coverage and set size.

        Args:
            test_scores: ``(N, C)`` softmax probabilities.
            test_labels: ``(N,)`` true labels.

        Returns:
            Dict with ``"coverage"`` (fraction of test samples whose true
            label is in the prediction set) and ``"avg_set_size"`` (mean
            prediction set size).
        """
        test_labels = np.asarray(test_labels).ravel().astype(int)
        prediction_sets = self.predict_set(test_scores)

        covered = sum(
            1 for i, pset in enumerate(prediction_sets)
            if test_labels[i] in pset
        )
        coverage = covered / max(len(test_labels), 1)
        avg_size = np.mean([len(pset) for pset in prediction_sets])

        return {
            "coverage": float(coverage),
            "avg_set_size": float(avg_size),
            "target_coverage": 1.0 - self._alpha,
        }


# ---------------------------------------------------------------------------
# ConformalRegressor (regression)
# ---------------------------------------------------------------------------


class ConformalRegressor:
    """Conformalized quantile regression for continuous outputs.

    Constructs prediction intervals with guaranteed coverage at the
    ``1 - alpha`` level using absolute residuals as nonconformity scores.

    Supports both symmetric intervals (from point predictions) and
    asymmetric intervals (from quantile predictions).

    Args:
        method: ``"absolute"`` for symmetric intervals from point
            predictions, or ``"quantile"`` for asymmetric intervals
            from quantile predictions.
    """

    VALID_METHODS = ("absolute", "quantile")

    def __init__(self, method: str = "absolute") -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"method must be one of {self.VALID_METHODS}, got {method!r}"
            )
        self.method = method
        self._quantile: Optional[float] = None
        self._quantile_lower: Optional[float] = None
        self._quantile_upper: Optional[float] = None
        self._alpha: float = 0.1
        self._calibrated: bool = False

    @property
    def is_calibrated(self) -> bool:
        """Whether the regressor has been calibrated."""
        return self._calibrated

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        residuals: np.ndarray,
        alpha: float = 0.1,
        lower_residuals: Optional[np.ndarray] = None,
        upper_residuals: Optional[np.ndarray] = None,
    ) -> Union[float, Tuple[float, float]]:
        """Calibrate the conformal regressor on calibration residuals.

        For ``method="absolute"``: uses ``|y - y_hat|`` residuals.
        For ``method="quantile"``: uses separate lower and upper residuals.

        Args:
            residuals: ``(N,)`` absolute residuals (for ``"absolute"``
                method) or ignored if lower/upper are provided.
            alpha: Desired miscoverage rate.
            lower_residuals: ``(N,)`` residuals for lower quantile
                (``y_hat_lower - y``). Required for ``"quantile"`` method.
            upper_residuals: ``(N,)`` residuals for upper quantile
                (``y - y_hat_upper``). Required for ``"quantile"`` method.

        Returns:
            Calibrated quantile(s).
        """
        self._alpha = alpha
        residuals = np.asarray(residuals, dtype=np.float64).ravel()
        n = len(residuals)

        if n == 0:
            raise ValueError("Calibration residuals are empty.")

        quantile_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)

        if self.method == "absolute":
            abs_residuals = np.abs(residuals)
            self._quantile = float(np.quantile(abs_residuals, quantile_level))
            self._calibrated = True
            logger.info(
                "ConformalRegressor calibrated (absolute): alpha={}, "
                "n_cal={}, quantile={:.4f}",
                alpha,
                n,
                self._quantile,
            )
            return self._quantile

        elif self.method == "quantile":
            if lower_residuals is None or upper_residuals is None:
                raise ValueError(
                    "For 'quantile' method, both lower_residuals and "
                    "upper_residuals must be provided."
                )
            lower_residuals = np.asarray(lower_residuals, dtype=np.float64).ravel()
            upper_residuals = np.asarray(upper_residuals, dtype=np.float64).ravel()

            # Symmetric quantile for both sides
            combined = np.maximum(lower_residuals, upper_residuals)
            q = float(np.quantile(combined, quantile_level))
            self._quantile_lower = q
            self._quantile_upper = q
            self._quantile = q
            self._calibrated = True

            logger.info(
                "ConformalRegressor calibrated (quantile): alpha={}, "
                "n_cal={}, quantile={:.4f}",
                alpha,
                n,
                q,
            )
            return (self._quantile_lower, self._quantile_upper)

        raise ValueError(f"Unknown method: {self.method}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_interval(
        self,
        predictions: np.ndarray,
        lower_predictions: Optional[np.ndarray] = None,
        upper_predictions: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct prediction intervals for new samples.

        Args:
            predictions: ``(N,)`` point predictions (for ``"absolute"``
                method) or ignored if quantile predictions are given.
            lower_predictions: ``(N,)`` lower quantile predictions
                (for ``"quantile"`` method).
            upper_predictions: ``(N,)`` upper quantile predictions
                (for ``"quantile"`` method).

        Returns:
            Tuple of ``(lower, upper)`` bound arrays, each of shape ``(N,)``.

        Raises:
            RuntimeError: If the regressor has not been calibrated.
        """
        if not self._calibrated:
            raise RuntimeError(
                "Regressor not calibrated. Call calibrate() first."
            )

        predictions = np.asarray(predictions, dtype=np.float64).ravel()

        if self.method == "absolute":
            assert self._quantile is not None
            lower = predictions - self._quantile
            upper = predictions + self._quantile
            return lower, upper

        elif self.method == "quantile":
            assert self._quantile_lower is not None
            assert self._quantile_upper is not None

            if lower_predictions is None or upper_predictions is None:
                # Fall back to symmetric intervals around point prediction
                lower = predictions - self._quantile_lower
                upper = predictions + self._quantile_upper
            else:
                lower_predictions = np.asarray(lower_predictions).ravel()
                upper_predictions = np.asarray(upper_predictions).ravel()
                lower = lower_predictions - self._quantile_lower
                upper = upper_predictions + self._quantile_upper

            return lower, upper

        raise ValueError(f"Unknown method: {self.method}")

    def evaluate_coverage(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray,
        lower_predictions: Optional[np.ndarray] = None,
        upper_predictions: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate empirical coverage and interval width.

        Args:
            predictions: ``(N,)`` point predictions.
            true_values: ``(N,)`` true target values.
            lower_predictions: Optional lower quantile predictions.
            upper_predictions: Optional upper quantile predictions.

        Returns:
            Dict with ``"coverage"`` (fraction of true values within
            intervals), ``"avg_width"`` (mean interval width), and
            ``"target_coverage"``.
        """
        true_values = np.asarray(true_values).ravel()
        lower, upper = self.predict_interval(
            predictions, lower_predictions, upper_predictions
        )

        covered = np.sum((true_values >= lower) & (true_values <= upper))
        coverage = float(covered / max(len(true_values), 1))
        avg_width = float(np.mean(upper - lower))

        return {
            "coverage": coverage,
            "avg_width": avg_width,
            "target_coverage": 1.0 - self._alpha,
        }
