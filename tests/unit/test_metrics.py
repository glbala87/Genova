"""Tests for evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from genova.evaluation.metrics import (
    mlm_accuracy,
    perplexity,
    auroc,
    auprc,
    expected_calibration_error,
    brier_score,
    pearson_correlation,
    spearman_correlation,
    mse,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# MLM metrics
# ---------------------------------------------------------------------------

class TestMLMAccuracy:

    def test_perfect(self):
        preds = np.array([1, 2, 3, 4])
        targets = np.array([1, 2, 3, 4])
        assert mlm_accuracy(preds, targets) == 1.0

    def test_half_correct(self):
        preds = np.array([1, 2, 3, 4])
        targets = np.array([1, 2, 0, 0])
        assert mlm_accuracy(preds, targets) == 0.5

    def test_ignore_index(self):
        preds = np.array([1, 2, 99, 99])
        targets = np.array([1, 2, -100, -100])
        assert mlm_accuracy(preds, targets) == 1.0

    def test_all_ignored(self):
        preds = np.array([1, 2])
        targets = np.array([-100, -100])
        assert mlm_accuracy(preds, targets) == 0.0

    def test_2d_input(self):
        preds = np.array([[1, 2], [3, 4]])
        targets = np.array([[1, 2], [3, 4]])
        assert mlm_accuracy(preds, targets) == 1.0


class TestPerplexity:

    def test_zero_loss(self):
        assert perplexity(0.0) == pytest.approx(1.0)

    def test_known_value(self):
        assert perplexity(1.0) == pytest.approx(math.e, rel=1e-5)

    def test_cap(self):
        # Loss > 20 should be capped
        assert perplexity(100.0) == pytest.approx(math.exp(20.0), rel=1e-5)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

class TestAUROC:

    def test_perfect_separation(self):
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        targets = np.array([1, 1, 0, 0])
        assert auroc(scores, targets) == pytest.approx(1.0)

    def test_random(self):
        rng = np.random.RandomState(42)
        scores = rng.rand(1000)
        targets = rng.randint(0, 2, 1000)
        val = auroc(scores, targets)
        assert 0.4 < val < 0.6  # should be near 0.5

    def test_single_class(self):
        scores = np.array([0.5, 0.6])
        targets = np.array([1, 1])
        assert auroc(scores, targets) == 0.0


class TestAUPRC:

    def test_perfect(self):
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        targets = np.array([1, 1, 0, 0])
        assert auprc(scores, targets) == pytest.approx(1.0)

    def test_single_class(self):
        scores = np.array([0.5, 0.6])
        targets = np.array([0, 0])
        assert auprc(scores, targets) == 0.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:

    def test_perfect_calibration(self):
        # Perfectly calibrated: score matches outcome
        scores = np.array([0.0, 0.0, 1.0, 1.0])
        targets = np.array([0, 0, 1, 1])
        ece = expected_calibration_error(scores, targets, n_bins=2)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_brier_perfect(self):
        scores = np.array([0.0, 1.0])
        targets = np.array([0, 1])
        assert brier_score(scores, targets) == 0.0

    def test_brier_worst(self):
        scores = np.array([1.0, 0.0])
        targets = np.array([0, 1])
        assert brier_score(scores, targets) == 1.0


# ---------------------------------------------------------------------------
# Correlation / regression
# ---------------------------------------------------------------------------

class TestCorrelation:

    def test_pearson_perfect(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert pearson_correlation(x, x) == pytest.approx(1.0)

    def test_pearson_negative(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([4.0, 3.0, 2.0, 1.0])
        assert pearson_correlation(x, y) == pytest.approx(-1.0)

    def test_pearson_constant(self):
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([2.0, 3.0, 4.0])
        assert pearson_correlation(x, y) == 0.0

    def test_spearman_perfect(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert spearman_correlation(x, x) == pytest.approx(1.0)

    def test_spearman_monotonic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([10.0, 200.0, 3000.0, 40000.0])
        assert spearman_correlation(x, y) == pytest.approx(1.0)

    def test_mse_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        assert mse(x, x) == 0.0

    def test_mse_known(self):
        preds = np.array([1.0, 2.0])
        targets = np.array([3.0, 4.0])
        assert mse(preds, targets) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# compute_metrics dispatcher
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_mlm(self):
        preds = np.array([1, 2, 3])
        targets = np.array([1, 2, 3])
        result = compute_metrics(preds, targets, task_type="mlm", loss=1.0)
        assert result["accuracy"] == 1.0
        assert "perplexity" in result

    def test_classification(self):
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        targets = np.array([1, 1, 0, 0])
        result = compute_metrics(scores, targets, task_type="classification")
        assert "auroc" in result
        assert "auprc" in result
        assert "ece" in result
        assert "brier_score" in result

    def test_regression(self):
        preds = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.1, 2.1, 3.1])
        result = compute_metrics(preds, targets, task_type="regression")
        assert "pearson_r" in result
        assert "spearman_rho" in result
        assert "mse" in result

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task_type"):
            compute_metrics(np.array([1]), np.array([1]), task_type="segmentation")
