"""Calibration analysis for genomic model predictions.

Provides reliability diagrams, temperature scaling, Platt scaling,
and Expected Calibration Error (ECE) computation.

Example::

    from genova.uncertainty.calibration import CalibrationAnalyzer

    analyzer = CalibrationAnalyzer()
    ece = analyzer.compute_ece(probabilities, labels)
    analyzer.reliability_diagram(probabilities, labels, save_path="reliability.pdf")
    optimal_temp = analyzer.temperature_scale(logits, labels)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False


# ---------------------------------------------------------------------------
# CalibrationAnalyzer
# ---------------------------------------------------------------------------


class CalibrationAnalyzer:
    """Analyse and improve model calibration.

    Supports reliability diagrams, temperature scaling, Platt scaling,
    and ECE computation.

    Args:
        n_bins: Default number of bins for ECE and reliability diagrams.
    """

    def __init__(self, n_bins: int = 15) -> None:
        self.n_bins = n_bins
        self._temperature: Optional[float] = None
        self._platt_a: Optional[float] = None
        self._platt_b: Optional[float] = None

    # -- ECE -----------------------------------------------------------------

    def compute_ece(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        n_bins: Optional[int] = None,
    ) -> float:
        """Compute Expected Calibration Error.

        Args:
            probabilities: Predicted probabilities, shape ``(N,)``.
            labels: Binary ground truth labels, shape ``(N,)``.
            n_bins: Number of equal-width bins.  Uses instance default if
                ``None``.

        Returns:
            ECE value (lower is better).
        """
        n_bins = n_bins or self.n_bins
        probs = np.asarray(probabilities, dtype=np.float64).ravel()
        labs = np.asarray(labels, dtype=np.float64).ravel()

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(probs)

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            if lo == 0.0:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs > lo) & (probs <= hi)
            count = int(mask.sum())
            if count == 0:
                continue
            avg_confidence = float(probs[mask].mean())
            avg_accuracy = float(labs[mask].mean())
            ece += (count / n) * abs(avg_confidence - avg_accuracy)

        return float(ece)

    def compute_mce(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        n_bins: Optional[int] = None,
    ) -> float:
        """Compute Maximum Calibration Error.

        Args:
            probabilities: Predicted probabilities, shape ``(N,)``.
            labels: Binary ground truth labels, shape ``(N,)``.
            n_bins: Number of bins.

        Returns:
            MCE value.
        """
        n_bins = n_bins or self.n_bins
        probs = np.asarray(probabilities, dtype=np.float64).ravel()
        labs = np.asarray(labels, dtype=np.float64).ravel()

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        max_gap = 0.0

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            if lo == 0.0:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs > lo) & (probs <= hi)
            if mask.sum() == 0:
                continue
            gap = abs(float(probs[mask].mean()) - float(labs[mask].mean()))
            max_gap = max(max_gap, gap)

        return max_gap

    # -- Reliability diagram -------------------------------------------------

    def reliability_diagram(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        n_bins: Optional[int] = None,
        save_path: Optional[str] = None,
        title: str = "Reliability Diagram",
        figsize: Tuple[float, float] = (6, 5),
        dpi: int = 300,
    ) -> Optional[Any]:
        """Generate a reliability (calibration) diagram.

        Plots the observed accuracy vs predicted confidence per bin,
        with a histogram of sample counts.

        Args:
            probabilities: Predicted probabilities, shape ``(N,)``.
            labels: Binary ground truth labels, shape ``(N,)``.
            n_bins: Number of bins.
            save_path: Path to save the figure.
            title: Figure title.
            figsize: Figure dimensions.
            dpi: Resolution.

        Returns:
            Matplotlib figure or ``None`` if matplotlib is unavailable.
        """
        if not _HAS_PLOTTING:
            logger.warning("matplotlib not available; skipping reliability diagram.")
            return None

        n_bins = n_bins or self.n_bins
        probs = np.asarray(probabilities, dtype=np.float64).ravel()
        labs = np.asarray(labels, dtype=np.float64).ravel()

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        accuracies = np.zeros(n_bins)
        confidences = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=int)

        for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if lo == 0.0:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs > lo) & (probs <= hi)
            count = int(mask.sum())
            counts[i] = count
            if count > 0:
                accuracies[i] = labs[mask].mean()
                confidences[i] = probs[mask].mean()

        ece = self.compute_ece(probs, labs, n_bins=n_bins)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )

        # Calibration plot
        non_empty = counts > 0
        ax1.bar(
            bin_centers[non_empty],
            accuracies[non_empty],
            width=1.0 / n_bins,
            alpha=0.6,
            color="#4C72B0",
            edgecolor="white",
            linewidth=0.5,
            label="Observed",
        )
        ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        ax1.set_ylabel("Observed Accuracy", fontsize=10)
        ax1.set_title(f"{title} (ECE = {ece:.4f})", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=9, frameon=False)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # Histogram
        ax2.bar(
            bin_centers,
            counts,
            width=1.0 / n_bins,
            alpha=0.6,
            color="#DD8452",
            edgecolor="white",
            linewidth=0.5,
        )
        ax2.set_xlabel("Predicted Probability", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            logger.info("Reliability diagram saved to {}", save_path)

        return fig

    # -- Temperature scaling -------------------------------------------------

    def temperature_scale(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        init_temp: float = 1.5,
    ) -> float:
        """Find optimal temperature for temperature scaling.

        Minimises NLL on the provided logits/labels via grid search.

        Args:
            logits: Raw model logits (pre-sigmoid/softmax), shape ``(N,)``
                or ``(N, C)``.
            labels: Ground truth labels, shape ``(N,)``.
            init_temp: Starting temperature for search.

        Returns:
            Optimal temperature value.
        """
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64).ravel()

        is_binary = logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] <= 1)

        def _nll(temp: float) -> float:
            if is_binary:
                scaled = logits.ravel() / temp
                probs = 1.0 / (1.0 + np.exp(-scaled))
                probs = np.clip(probs, 1e-10, 1 - 1e-10)
                return float(
                    -np.mean(
                        labels * np.log(probs)
                        + (1 - labels) * np.log(1 - probs)
                    )
                )
            else:
                scaled = logits / temp
                exp_s = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
                probs = exp_s / exp_s.sum(axis=-1, keepdims=True)
                probs = np.clip(probs, 1e-10, 1.0)
                return float(
                    -np.mean(np.log(probs[np.arange(len(labels)), labels]))
                )

        # Coarse search
        best_temp = init_temp
        best_loss = _nll(init_temp)

        for t in np.linspace(0.1, 10.0, 100):
            loss = _nll(t)
            if loss < best_loss:
                best_loss = loss
                best_temp = float(t)

        # Fine search
        for t in np.linspace(max(0.01, best_temp - 1.0), best_temp + 1.0, 200):
            loss = _nll(t)
            if loss < best_loss:
                best_loss = loss
                best_temp = float(t)

        self._temperature = best_temp
        logger.info(
            "Temperature scaling: optimal T={:.4f}, NLL={:.4f}",
            best_temp,
            best_loss,
        )
        return best_temp

    def apply_temperature(
        self,
        logits: np.ndarray,
        temperature: Optional[float] = None,
    ) -> np.ndarray:
        """Apply temperature scaling to logits.

        Args:
            logits: Raw logits, shape ``(N,)`` or ``(N, C)``.
            temperature: Temperature value.  Uses the last fitted value
                if ``None``.

        Returns:
            Calibrated probabilities.
        """
        temp = temperature or self._temperature
        if temp is None:
            raise ValueError(
                "No temperature set. Call temperature_scale() first or "
                "provide a temperature value."
            )
        logits = np.asarray(logits, dtype=np.float64)
        scaled = logits / temp

        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] <= 1):
            return 1.0 / (1.0 + np.exp(-scaled.ravel()))
        else:
            exp_s = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
            return exp_s / exp_s.sum(axis=-1, keepdims=True)

    # -- Platt scaling -------------------------------------------------------

    def platt_scale(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 200,
        lr: float = 0.01,
    ) -> Tuple[float, float]:
        """Fit Platt scaling: ``P(y=1|x) = sigmoid(a * logit + b)``.

        Uses gradient descent to find optimal ``a`` and ``b`` parameters.

        Args:
            logits: Raw logits, shape ``(N,)``.
            labels: Binary labels, shape ``(N,)``.
            max_iter: Maximum gradient descent iterations.
            lr: Learning rate.

        Returns:
            Tuple of ``(a, b)`` parameters.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()

        # Target values with label smoothing (Platt's original formulation)
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        t_pos = (n_pos + 1) / (n_pos + 2) if n_pos > 0 else 0.5
        t_neg = 1.0 / (n_neg + 2) if n_neg > 0 else 0.5
        targets = np.where(labels == 1, t_pos, t_neg)

        a = 1.0
        b = 0.0

        for iteration in range(max_iter):
            z = a * logits + b
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)

            # Gradients
            diff = p - targets
            grad_a = float(np.mean(diff * logits))
            grad_b = float(np.mean(diff))

            a -= lr * grad_a
            b -= lr * grad_b

            if iteration % 50 == 0:
                nll = -float(np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p)))
                logger.debug("Platt scaling iter {}: a={:.4f}, b={:.4f}, NLL={:.4f}", iteration, a, b, nll)

        self._platt_a = a
        self._platt_b = b
        logger.info("Platt scaling: a={:.4f}, b={:.4f}", a, b)
        return a, b

    def apply_platt(
        self,
        logits: np.ndarray,
        a: Optional[float] = None,
        b: Optional[float] = None,
    ) -> np.ndarray:
        """Apply Platt scaling to logits.

        Args:
            logits: Raw logits, shape ``(N,)``.
            a: Scale parameter.  Uses fitted value if ``None``.
            b: Shift parameter.  Uses fitted value if ``None``.

        Returns:
            Calibrated probabilities.
        """
        a = a if a is not None else self._platt_a
        b = b if b is not None else self._platt_b
        if a is None or b is None:
            raise ValueError(
                "Platt parameters not set. Call platt_scale() first."
            )
        logits = np.asarray(logits, dtype=np.float64).ravel()
        z = a * logits + b
        return 1.0 / (1.0 + np.exp(-z))

    # -- Summary -------------------------------------------------------------

    def full_analysis(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        logits: Optional[np.ndarray] = None,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a full calibration analysis.

        Computes ECE, MCE, generates reliability diagram, and optionally
        fits temperature/Platt scaling if logits are provided.

        Args:
            probabilities: Predicted probabilities.
            labels: Ground truth labels.
            logits: Optional raw logits for scaling methods.
            save_dir: Directory to save figures and results.

        Returns:
            Dictionary of calibration analysis results.
        """
        results: Dict[str, Any] = {
            "ece": self.compute_ece(probabilities, labels),
            "mce": self.compute_mce(probabilities, labels),
        }

        if save_dir:
            from pathlib import Path

            Path(save_dir).mkdir(parents=True, exist_ok=True)
            self.reliability_diagram(
                probabilities,
                labels,
                save_path=str(Path(save_dir) / "reliability_diagram.pdf"),
                title="Before Calibration",
            )

        if logits is not None:
            # Temperature scaling
            temp = self.temperature_scale(logits, labels)
            cal_probs = self.apply_temperature(logits, temp)
            results["temperature"] = temp
            results["ece_after_temperature"] = self.compute_ece(cal_probs, labels)

            # Platt scaling
            a, b = self.platt_scale(logits, labels)
            platt_probs = self.apply_platt(logits, a, b)
            results["platt_a"] = a
            results["platt_b"] = b
            results["ece_after_platt"] = self.compute_ece(platt_probs, labels)

            if save_dir:
                from pathlib import Path

                self.reliability_diagram(
                    cal_probs,
                    labels,
                    save_path=str(Path(save_dir) / "reliability_after_temp.pdf"),
                    title="After Temperature Scaling",
                )
                self.reliability_diagram(
                    platt_probs,
                    labels,
                    save_path=str(Path(save_dir) / "reliability_after_platt.pdf"),
                    title="After Platt Scaling",
                )

        logger.info("Calibration analysis: {}", results)
        return results
