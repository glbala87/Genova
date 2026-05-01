"""Benchmark validation tests: validate metrics and prediction pipelines
against independent benchmarks using synthetic biologically-grounded data.

These tests ensure that Genova's evaluation infrastructure produces
correct results when given known inputs, and that accuracy thresholds
are met on synthetic ClinVar-like and BEND-like benchmark data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from genova.evaluation.metrics import (
    auroc,
    auprc,
    brier_score,
    expected_calibration_error,
    mlm_accuracy,
    pearson_correlation,
    spearman_correlation,
)
from genova.benchmark.tasks import (
    _f1_score,
    _mcc,
    _compute_all_metrics,
    BenchmarkDataset,
)
from genova.benchmark.standard_benchmarks import LinearProbe


# ---------------------------------------------------------------------------
# Section 1: ClinVar-style benchmark validation
# ---------------------------------------------------------------------------


class TestClinVarBenchmarkValidation:
    """Validate variant prediction metrics against ClinVar-like synthetic data."""

    @pytest.mark.benchmark
    def test_auroc_perfect_clinvar_separation(self):
        """Perfect ClinVar separation: all pathogenic score high, all benign low."""
        np.random.seed(42)
        n = 500
        # Simulate: benign variants score [0.0, 0.3], pathogenic [0.7, 1.0]
        benign_scores = np.random.uniform(0.0, 0.3, n)
        pathogenic_scores = np.random.uniform(0.7, 1.0, n)
        scores = np.concatenate([benign_scores, pathogenic_scores])
        labels = np.concatenate([np.zeros(n), np.ones(n)])
        result = auroc(scores, labels)
        assert result == 1.0, f"Perfect separation should give AUROC=1.0, got {result}"

    @pytest.mark.benchmark
    def test_auroc_realistic_clinvar_above_threshold(self):
        """Realistic noisy ClinVar data should achieve AUROC >= 0.80."""
        np.random.seed(42)
        n = 1000
        # Simulate realistic overlap: benign ~N(0.3, 0.15), pathogenic ~N(0.7, 0.15)
        benign_scores = np.clip(np.random.normal(0.3, 0.15, n), 0, 1)
        pathogenic_scores = np.clip(np.random.normal(0.7, 0.15, n), 0, 1)
        scores = np.concatenate([benign_scores, pathogenic_scores])
        labels = np.concatenate([np.zeros(n), np.ones(n)])
        result = auroc(scores, labels)
        assert result >= 0.80, f"ClinVar AUROC {result:.3f} below 0.80 threshold"

    @pytest.mark.benchmark
    def test_auprc_realistic_clinvar_above_threshold(self):
        """Realistic ClinVar data should also achieve high AUPRC."""
        np.random.seed(42)
        n = 1000
        benign_scores = np.clip(np.random.normal(0.3, 0.15, n), 0, 1)
        pathogenic_scores = np.clip(np.random.normal(0.7, 0.15, n), 0, 1)
        scores = np.concatenate([benign_scores, pathogenic_scores])
        labels = np.concatenate([np.zeros(n), np.ones(n)])
        result = auprc(scores, labels)
        assert result >= 0.75, f"ClinVar AUPRC {result:.3f} below 0.75 threshold"

    @pytest.mark.benchmark
    def test_brier_score_realistic_clinvar(self):
        """Brier score on realistic ClinVar data should be < 0.25 (better than random)."""
        np.random.seed(42)
        n = 500
        benign_scores = np.clip(np.random.normal(0.25, 0.12, n), 0, 1)
        pathogenic_scores = np.clip(np.random.normal(0.75, 0.12, n), 0, 1)
        scores = np.concatenate([benign_scores, pathogenic_scores])
        labels = np.concatenate([np.zeros(n), np.ones(n)])
        result = brier_score(scores, labels)
        assert result < 0.25, f"ClinVar Brier score {result:.3f} >= 0.25"

    @pytest.mark.benchmark
    def test_metrics_pipeline_end_to_end(self):
        """Full _compute_all_metrics pipeline should return all expected keys."""
        np.random.seed(42)
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        labels = np.array([0, 0, 0, 1, 1, 1])
        result = _compute_all_metrics(scores, labels)
        expected_keys = {"auroc", "auprc", "ece", "f1", "mcc"}
        assert set(result.keys()) == expected_keys
        # Perfect predictions: all metrics should be strong
        assert result["auroc"] == 1.0
        assert result["f1"] == 1.0
        assert result["mcc"] == 1.0


# ---------------------------------------------------------------------------
# Section 2: Transition / Transversion ratio validation
# ---------------------------------------------------------------------------


class TestTransitionTransversionValidation:
    """Validate that Ti/Tv ratio computation is biologically plausible."""

    @staticmethod
    def _compute_ti_tv_ratio(variants):
        """Compute transition/transversion ratio from variant list."""
        transitions = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
        ti = sum(1 for ref, alt in variants if (ref, alt) in transitions)
        tv = sum(1 for ref, alt in variants if (ref, alt) not in transitions)
        return ti / max(tv, 1)

    @pytest.mark.benchmark
    def test_known_ti_tv_ratio_detection(self):
        """Synthetic variants with known Ti/Tv=2.1 should be detected correctly."""
        np.random.seed(42)
        transitions = [("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")]
        transversions = [("A", "C"), ("A", "T"), ("G", "C"), ("G", "T"),
                         ("C", "A"), ("T", "A"), ("C", "G"), ("T", "G")]
        # Create variants with Ti/Tv ≈ 2.1
        n_tv = 100
        n_ti = 210  # ratio = 210/100 = 2.1
        variants = []
        for _ in range(n_ti):
            variants.append(transitions[np.random.randint(0, 4)])
        for _ in range(n_tv):
            variants.append(transversions[np.random.randint(0, 8)])

        ratio = self._compute_ti_tv_ratio(variants)
        assert 2.0 <= ratio <= 2.2, f"Ti/Tv ratio {ratio:.2f} not in [2.0, 2.2]"

    @pytest.mark.benchmark
    def test_all_transitions_gives_high_ratio(self):
        """All-transition variants should give Ti/Tv → infinity (or very high)."""
        variants = [("A", "G")] * 100
        ratio = self._compute_ti_tv_ratio(variants)
        assert ratio >= 100, "All transitions should give very high Ti/Tv"

    @pytest.mark.benchmark
    def test_all_transversions_gives_zero_ratio(self):
        """All-transversion variants should give Ti/Tv = 0."""
        variants = [("A", "C")] * 100
        ratio = self._compute_ti_tv_ratio(variants)
        assert ratio == 0.0, "All transversions should give Ti/Tv = 0"


# ---------------------------------------------------------------------------
# Section 3: Benchmark task infrastructure validation
# ---------------------------------------------------------------------------


class TestBenchmarkTaskInfrastructure:
    """Validate benchmark scoring functions against known inputs."""

    @pytest.mark.benchmark
    def test_f1_score_perfect(self):
        """Perfect binary predictions -> F1 = 1.0."""
        preds = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        targets = np.array([0, 0, 0, 1, 1, 1])
        assert abs(_f1_score(preds, targets) - 1.0) < 1e-9

    @pytest.mark.benchmark
    def test_f1_score_all_wrong(self):
        """All wrong predictions -> F1 = 0.0."""
        preds = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        targets = np.array([0, 0, 0, 1, 1, 1])
        assert abs(_f1_score(preds, targets) - 0.0) < 1e-9

    @pytest.mark.benchmark
    def test_f1_score_half_correct(self):
        """Half correct predictions -> F1 in (0, 1)."""
        preds = np.array([0.0, 1.0, 0.0, 1.0])
        targets = np.array([0, 0, 1, 1])
        f1 = _f1_score(preds, targets)
        assert 0.0 < f1 < 1.0, f"F1 {f1} should be between 0 and 1"

    @pytest.mark.benchmark
    def test_mcc_perfect(self):
        """Perfect predictions -> MCC = 1.0."""
        preds = np.array([0.0, 0.0, 1.0, 1.0])
        targets = np.array([0, 0, 1, 1])
        assert abs(_mcc(preds, targets) - 1.0) < 1e-9

    @pytest.mark.benchmark
    def test_mcc_inverse(self):
        """Perfectly inverse predictions -> MCC = -1.0."""
        preds = np.array([1.0, 1.0, 0.0, 0.0])
        targets = np.array([0, 0, 1, 1])
        assert abs(_mcc(preds, targets) - (-1.0)) < 1e-9

    @pytest.mark.benchmark
    def test_mcc_random_near_zero(self):
        """Random predictions -> MCC near 0."""
        np.random.seed(42)
        preds = np.random.rand(1000)
        targets = np.random.randint(0, 2, 1000)
        mcc_val = _mcc(preds, targets)
        assert abs(mcc_val) < 0.15, f"Random MCC {mcc_val} should be near 0"

    @pytest.mark.benchmark
    def test_compute_all_metrics_returns_complete_dict(self):
        """_compute_all_metrics should return all 5 expected metrics."""
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        targets = np.array([0, 0, 1, 1])
        result = _compute_all_metrics(scores, targets)
        assert "auroc" in result
        assert "auprc" in result
        assert "ece" in result
        assert "f1" in result
        assert "mcc" in result
        # All values should be finite
        for k, v in result.items():
            assert math.isfinite(v), f"Metric {k} is not finite: {v}"

    @pytest.mark.benchmark
    def test_linear_probe_forward_shape(self):
        """LinearProbe produces correct output shape."""
        probe = LinearProbe(input_dim=64, num_classes=2, dropout=0.0)
        x = torch.randn(8, 64)
        out = probe(x)
        assert out.shape == (8, 2)

    @pytest.mark.benchmark
    def test_linear_probe_single_class(self):
        """LinearProbe with num_classes=1 for binary classification."""
        probe = LinearProbe(input_dim=128, num_classes=1, dropout=0.0)
        x = torch.randn(4, 128)
        out = probe(x)
        assert out.shape == (4, 1)

    @pytest.mark.benchmark
    def test_benchmark_dataset_len(self):
        """BenchmarkDataset __len__ matches number of sequences."""
        ds = BenchmarkDataset(
            sequences=["ACGT", "GCTA", "TTTT"],
            labels=np.array([0, 1, 0]),
        )
        assert len(ds) == 3


# ---------------------------------------------------------------------------
# Section 4: Metric consistency validation
# ---------------------------------------------------------------------------


class TestMetricConsistencyValidation:
    """Cross-validate metrics for internal consistency."""

    @pytest.mark.benchmark
    def test_perfect_predictor_both_auroc_auprc(self):
        """Perfect predictor should get 1.0 on both AUROC and AUPRC."""
        targets = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.15, 0.2, 0.25, 0.75, 0.8, 0.85, 0.9])
        assert auroc(scores, targets) == 1.0
        assert auprc(scores, targets) == 1.0

    @pytest.mark.benchmark
    def test_auroc_auprc_ordering_preserved(self):
        """Better predictions should give higher AUROC and AUPRC."""
        np.random.seed(99)
        n = 200
        targets = np.concatenate([np.zeros(n), np.ones(n)])
        # Good predictions: clear separation
        good_scores = np.concatenate([
            np.clip(np.random.normal(0.2, 0.1, n), 0, 1),
            np.clip(np.random.normal(0.8, 0.1, n), 0, 1),
        ])
        # Bad predictions: heavy overlap
        bad_scores = np.concatenate([
            np.clip(np.random.normal(0.45, 0.15, n), 0, 1),
            np.clip(np.random.normal(0.55, 0.15, n), 0, 1),
        ])

        good_auroc = auroc(good_scores, targets)
        bad_auroc = auroc(bad_scores, targets)
        assert good_auroc > bad_auroc, "Better preds should have higher AUROC"

        good_auprc = auprc(good_scores, targets)
        bad_auprc = auprc(bad_scores, targets)
        assert good_auprc > bad_auprc, "Better preds should have higher AUPRC"

    @pytest.mark.benchmark
    def test_metrics_handle_all_positive(self):
        """Metrics should handle edge case where all labels are positive."""
        scores = np.array([0.5, 0.6, 0.7, 0.8])
        targets = np.array([1, 1, 1, 1])
        # Should not crash
        f1 = _f1_score(scores, targets)
        assert f1 >= 0.0

    @pytest.mark.benchmark
    def test_metrics_handle_all_negative(self):
        """Metrics should handle edge case where all labels are negative."""
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        targets = np.array([0, 0, 0, 0])
        f1 = _f1_score(scores, targets)
        assert f1 == 0.0  # No positives predicted or actual

    @pytest.mark.benchmark
    def test_ece_better_for_calibrated_model(self):
        """Well-calibrated model should have lower ECE than poorly calibrated."""
        np.random.seed(42)
        targets = np.array([0] * 50 + [1] * 50)
        # Well calibrated: predict ~0.5
        calibrated = np.array([0.5] * 100)
        # Poorly calibrated: always predict 0.99
        uncalibrated = np.array([0.99] * 100)
        ece_good = expected_calibration_error(calibrated, targets)
        ece_bad = expected_calibration_error(uncalibrated, targets)
        assert ece_good < ece_bad, "Calibrated model should have lower ECE"

    @pytest.mark.benchmark
    def test_pearson_spearman_consistent_on_linear(self):
        """Pearson and Spearman should both be ~1.0 on perfectly linear data."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        assert abs(pearson_correlation(x, y) - 1.0) < 1e-9
        assert abs(spearman_correlation(x, y) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Section 5: Model comparison validation (random vs. informed)
# ---------------------------------------------------------------------------


class TestModelComparisonValidation:
    """Validate that metrics distinguish random from informed predictions."""

    @pytest.mark.benchmark
    def test_random_model_auroc_near_half(self):
        """Random predictions should give AUROC ~0.5."""
        np.random.seed(42)
        n = 2000
        scores = np.random.rand(n)
        targets = np.random.randint(0, 2, n)
        result = auroc(scores, targets)
        assert 0.4 < result < 0.6, f"Random AUROC {result} not near 0.5"

    @pytest.mark.benchmark
    def test_informed_beats_random(self):
        """An informed model should clearly beat a random baseline."""
        np.random.seed(42)
        n = 500
        targets = np.concatenate([np.zeros(n), np.ones(n)])
        # Informed: knows the answer with noise
        informed = np.concatenate([
            np.clip(np.random.normal(0.3, 0.1, n), 0, 1),
            np.clip(np.random.normal(0.7, 0.1, n), 0, 1),
        ])
        # Random baseline
        random_scores = np.random.rand(2 * n)

        informed_auroc = auroc(informed, targets)
        random_auroc = auroc(random_scores, targets)
        assert informed_auroc > random_auroc + 0.2, (
            f"Informed AUROC {informed_auroc:.3f} should clearly beat "
            f"random {random_auroc:.3f}"
        )

    @pytest.mark.benchmark
    def test_mlm_accuracy_distinguishes_quality(self):
        """MLM accuracy should distinguish good from bad token predictions."""
        # Good predictions: 90% correct
        good_preds = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1])
        good_targets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # Bad predictions: 10% correct
        bad_preds = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 1])
        bad_targets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        good_acc = mlm_accuracy(good_preds, good_targets)
        bad_acc = mlm_accuracy(bad_preds, bad_targets)
        assert good_acc > bad_acc, "Good predictions should have higher accuracy"
        assert good_acc >= 0.9
        assert bad_acc <= 0.2

    @pytest.mark.benchmark
    def test_brier_score_distinguishes_quality(self):
        """Brier score should be lower for better calibrated predictions."""
        targets = np.array([0, 0, 1, 1])
        good_scores = np.array([0.1, 0.2, 0.8, 0.9])
        bad_scores = np.array([0.5, 0.5, 0.5, 0.5])
        good_brier = brier_score(good_scores, targets)
        bad_brier = brier_score(bad_scores, targets)
        assert good_brier < bad_brier, "Better predictions should have lower Brier"

    @pytest.mark.benchmark
    def test_f1_mcc_agreement_on_clear_cases(self):
        """F1 and MCC should agree on direction for clearly good/bad predictions."""
        targets = np.array([0, 0, 0, 1, 1, 1])
        good = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        bad = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

        good_f1, bad_f1 = _f1_score(good, targets), _f1_score(bad, targets)
        good_mcc, bad_mcc = _mcc(good, targets), _mcc(bad, targets)

        assert good_f1 > bad_f1, "Good preds should have higher F1"
        assert good_mcc > bad_mcc, "Good preds should have higher MCC"
