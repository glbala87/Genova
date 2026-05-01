"""Tests for training utilities and evaluation modules.

Covers:
- genova.training.ema.EMAModel
- genova.training.curriculum.CurriculumScheduler, CurriculumSampler
- genova.evaluation.cross_validation.FoldResult, CVResult
- genova.evaluation.statistical_tests.bootstrap_ci, cohens_d, fdr_correction
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from genova.training.ema import EMAModel
from genova.training.curriculum import CurriculumScheduler, CurriculumSampler
from genova.evaluation.cross_validation import FoldResult, CVResult
from genova.evaluation.statistical_tests import bootstrap_ci, cohens_d, fdr_correction


# =========================================================================
# EMAModel
# =========================================================================


class TestEMAModel:

    @pytest.fixture()
    def model(self):
        torch.manual_seed(0)
        return nn.Linear(10, 5)

    def test_creation_does_not_crash(self, model):
        ema = EMAModel(model, decay=0.999)
        assert ema.decay == 0.999

    def test_update_moves_shadow_toward_model(self, model):
        torch.manual_seed(0)
        ema = EMAModel(model, decay=0.9)
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        ema.update()

        for name in ema.shadow:
            old_dist = (initial_shadow[name] - model.state_dict()[name]).abs().sum()
            new_dist = (ema.shadow[name] - model.state_dict()[name]).abs().sum()
            assert new_dist < old_dist

    def test_apply_shadow_and_restore_roundtrip(self, model):
        torch.manual_seed(1)
        ema = EMAModel(model, decay=0.99)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))

        original_weights = {n: p.data.clone() for n, p in model.named_parameters()}

        ema.apply_shadow()
        for name, param in model.named_parameters():
            if name in ema.shadow:
                assert torch.allclose(param.data, ema.shadow[name])

        ema.restore()
        for name, param in model.named_parameters():
            assert torch.allclose(param.data, original_weights[name])

    def test_decay_zero_shadow_equals_model(self, model):
        torch.manual_seed(2)
        ema = EMAModel(model, decay=0.0)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 2.0)

        ema.update()

        for name, param in model.named_parameters():
            if name in ema.shadow:
                assert torch.allclose(ema.shadow[name], param.data)

    def test_decay_one_shadow_stays_initial(self, model):
        torch.manual_seed(3)
        ema = EMAModel(model, decay=1.0)
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 5.0)

        ema.update()

        for name in ema.shadow:
            assert torch.allclose(ema.shadow[name], initial_shadow[name])


# =========================================================================
# CurriculumScheduler
# =========================================================================


class TestCurriculumScheduler:

    def test_score_difficulty_returns_float(self):
        scheduler = CurriculumScheduler()
        score = scheduler.score_difficulty("ACGTACGTACGT")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_high_n_content_scores_harder(self):
        scheduler = CurriculumScheduler()
        pure_seq = "ACGT" * 100
        n_seq = "NNNN" * 100
        pure_score = scheduler.score_difficulty(pure_seq)
        n_score = scheduler.score_difficulty(n_seq)
        assert n_score > pure_score

    def test_empty_sequence_does_not_crash(self):
        scheduler = CurriculumScheduler()
        score = scheduler.score_difficulty("")
        assert isinstance(score, float)


# =========================================================================
# CurriculumSampler
# =========================================================================


class TestCurriculumSampler:

    @pytest.fixture()
    def difficulties(self):
        np.random.seed(42)
        return np.random.rand(100)

    def test_yields_valid_indices(self, difficulties):
        sampler = CurriculumSampler(
            difficulties=difficulties, dataset_size=100, competence=1.0
        )
        indices = list(sampler)
        assert len(indices) == 100
        assert all(0 <= idx < 100 for idx in indices)

    def test_competence_limits_accessible_samples(self, difficulties):
        sampler = CurriculumSampler(
            difficulties=difficulties, dataset_size=100, competence=0.5
        )
        indices = list(sampler)
        assert len(indices) == 50
        assert all(0 <= idx < 100 for idx in indices)

    def test_set_epoch_changes_sampling(self, difficulties):
        sampler = CurriculumSampler(
            difficulties=difficulties, dataset_size=100, competence=0.5
        )
        indices_epoch0 = list(sampler)
        sampler.set_epoch(1)
        indices_epoch1 = list(sampler)
        # Same competence but different epoch -> different shuffle
        assert indices_epoch0 != indices_epoch1


# =========================================================================
# FoldResult and CVResult
# =========================================================================


class TestFoldResult:

    def test_creation_with_metrics(self):
        fr = FoldResult(
            fold_idx=0,
            metrics={"accuracy": 0.95, "f1": 0.90},
            train_size=800,
            val_size=200,
        )
        assert fr.fold_idx == 0
        assert fr.metrics["accuracy"] == 0.95
        assert fr.train_size == 800
        assert fr.val_size == 200
        assert fr.fold_name is None

    def test_creation_with_fold_name(self):
        fr = FoldResult(
            fold_idx=2,
            metrics={"auroc": 0.88},
            train_size=500,
            val_size=100,
            fold_name="chr1",
        )
        assert fr.fold_name == "chr1"


class TestCVResult:

    @pytest.fixture()
    def fold_results(self):
        return [
            FoldResult(fold_idx=0, metrics={"acc": 0.90, "f1": 0.85}, train_size=80, val_size=20),
            FoldResult(fold_idx=1, metrics={"acc": 0.92, "f1": 0.88}, train_size=80, val_size=20),
            FoldResult(fold_idx=2, metrics={"acc": 0.88, "f1": 0.83}, train_size=80, val_size=20),
        ]

    def test_cvresult_creation(self, fold_results):
        # Compute mean/std manually
        acc_vals = [fr.metrics["acc"] for fr in fold_results]
        f1_vals = [fr.metrics["f1"] for fr in fold_results]
        mean_metrics = {
            "acc": float(np.mean(acc_vals)),
            "f1": float(np.mean(f1_vals)),
        }
        std_metrics = {
            "acc": float(np.std(acc_vals, ddof=1)),
            "f1": float(np.std(f1_vals, ddof=1)),
        }
        result = CVResult(
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            cv_type="kfold",
            n_folds=3,
        )
        assert result.n_folds == 3
        assert result.cv_type == "kfold"
        assert "acc" in result.mean_metrics
        assert "f1" in result.std_metrics


# =========================================================================
# Statistical tests
# =========================================================================


class TestBootstrapCI:

    def test_returns_tuple_of_two_floats(self):
        np.random.seed(0)
        scores = np.random.rand(50)
        labels = (scores > 0.5).astype(int)

        def accuracy(s, l):
            return float(np.mean((s > 0.5).astype(int) == l))

        ci = bootstrap_ci(scores, labels, metric_fn=accuracy, n_bootstrap=500, seed=42)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert isinstance(ci[0], float)
        assert isinstance(ci[1], float)

    def test_ci_lower_less_than_upper(self):
        np.random.seed(1)
        scores = np.random.rand(100)
        labels = (scores > 0.5).astype(int)

        def accuracy(s, l):
            return float(np.mean((s > 0.5).astype(int) == l))

        ci_low, ci_high = bootstrap_ci(
            scores, labels, metric_fn=accuracy, n_bootstrap=1000, seed=42
        )
        assert ci_low <= ci_high


class TestCohensD:

    def test_identical_distributions_near_zero(self):
        np.random.seed(10)
        a = np.random.randn(200)
        b = np.random.randn(200)
        d = cohens_d(a, b)
        assert abs(d) < 0.3

    def test_very_different_distributions_large(self):
        np.random.seed(11)
        a = np.random.randn(100) + 10.0
        b = np.random.randn(100)
        d = cohens_d(a, b)
        assert d > 2.0


class TestFDRCorrection:

    def test_returns_adjusted_pvalues(self):
        p_values = np.array([0.01, 0.04, 0.03, 0.20])
        adjusted, rejected = fdr_correction(p_values, alpha=0.05)
        assert adjusted.shape == p_values.shape
        assert rejected.shape == p_values.shape
        assert rejected.dtype == bool

    def test_adjusted_geq_raw(self):
        p_values = np.array([0.001, 0.01, 0.05, 0.10, 0.50])
        adjusted, _ = fdr_correction(p_values, alpha=0.05)
        assert np.all(adjusted >= p_values - 1e-15)

    def test_adjusted_at_most_one(self):
        p_values = np.array([0.5, 0.8, 0.99])
        adjusted, _ = fdr_correction(p_values, alpha=0.05)
        assert np.all(adjusted <= 1.0 + 1e-15)

    def test_significant_pvalues_rejected(self):
        p_values = np.array([0.001, 0.002, 0.5, 0.9])
        _, rejected = fdr_correction(p_values, alpha=0.05)
        assert rejected[0]
        assert rejected[1]
