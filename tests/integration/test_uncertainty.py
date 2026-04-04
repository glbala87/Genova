"""Integration test: uncertainty estimation.

Tests MC Dropout prediction, BayesianLinear KL divergence, BayesianWrapper
conversion, and calibration analysis.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from genova.utils.config import ModelConfig
from genova.models.transformer import GenovaForMLM
from genova.uncertainty.mc_dropout import MCDropoutPredictor
from genova.uncertainty.bayesian import BayesianLinear, BayesianWrapper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_config():
    return ModelConfig(
        arch="transformer",
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        dropout=0.1,  # need dropout for MC dropout
        attention_dropout=0.1,
        max_position_embeddings=128,
        rotary_emb=False,
        flash_attention=False,
        gradient_checkpointing=False,
        tie_word_embeddings=False,
        vocab_size=32,
        pad_token_id=0,
    )


@pytest.fixture
def sample_input():
    B, L = 4, 16
    return torch.randint(5, 30, (B, L))


# ---------------------------------------------------------------------------
# MC Dropout tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMCDropout:
    """MC Dropout uncertainty estimation tests."""

    def test_predict_with_uncertainty(self, tiny_config, sample_input):
        """MC Dropout produces mean, variance, and entropy."""
        model = GenovaForMLM(tiny_config)
        predictor = MCDropoutPredictor(
            model, n_forward_passes=5, device="cpu"
        )

        result = predictor.predict_with_uncertainty(sample_input)

        assert "mean" in result, "Should return mean predictions"
        assert "variance" in result, "Should return variance"
        assert "entropy" in result, "Should return entropy"
        assert "std" in result, "Should return std"

    def test_uncertainty_non_negative(self, tiny_config, sample_input):
        """Variance and entropy should be non-negative."""
        model = GenovaForMLM(tiny_config)
        predictor = MCDropoutPredictor(
            model, n_forward_passes=5, device="cpu"
        )

        result = predictor.predict_with_uncertainty(sample_input)

        assert (result["variance"] >= 0).all(), "Variance should be non-negative"
        assert (result["std"] >= 0).all(), "Std should be non-negative"

    def test_all_passes_returned(self, tiny_config, sample_input):
        """All individual forward pass predictions returned when requested."""
        model = GenovaForMLM(tiny_config)
        n_passes = 5
        predictor = MCDropoutPredictor(
            model, n_forward_passes=n_passes, device="cpu"
        )

        result = predictor.predict_with_uncertainty(
            sample_input, return_all_passes=True
        )

        assert "all_passes" in result, "Should include all_passes"
        assert result["all_passes"].shape[0] == n_passes, (
            f"Expected {n_passes} passes, got {result['all_passes'].shape[0]}"
        )

    def test_mc_dropout_varies_across_passes(self, tiny_config, sample_input):
        """With dropout active, predictions should vary across passes."""
        model = GenovaForMLM(tiny_config)
        predictor = MCDropoutPredictor(
            model, n_forward_passes=10, device="cpu"
        )

        result = predictor.predict_with_uncertainty(
            sample_input, return_all_passes=True
        )

        # Check that there is some variance (passes are not identical)
        total_var = result["variance"].sum()
        assert total_var > 0, (
            "With dropout enabled, predictions should vary across passes"
        )


# ---------------------------------------------------------------------------
# BayesianLinear tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBayesianLinear:
    """BayesianLinear layer tests."""

    def test_forward_shape(self):
        """BayesianLinear forward produces correct output shape."""
        layer = BayesianLinear(64, 32, bias=True)
        x = torch.randn(4, 64)

        out = layer(x)

        assert out.shape == (4, 32), f"Expected (4, 32), got {out.shape}"

    def test_kl_divergence_positive(self):
        """KL divergence should be non-negative."""
        layer = BayesianLinear(64, 32)

        kl = layer.kl_divergence()

        assert kl.item() >= 0, f"KL should be non-negative, got {kl.item()}"
        assert torch.isfinite(kl), "KL should be finite"

    def test_stochastic_forward(self):
        """Multiple forward passes should give different results."""
        layer = BayesianLinear(64, 32)
        x = torch.randn(1, 64)

        outputs = [layer(x).detach() for _ in range(10)]
        # Check that outputs are not all identical
        diffs = [
            (outputs[i] - outputs[0]).abs().sum().item()
            for i in range(1, len(outputs))
        ]
        assert any(d > 0 for d in diffs), (
            "Bayesian layer should produce stochastic outputs"
        )

    def test_no_bias(self):
        """BayesianLinear with bias=False works."""
        layer = BayesianLinear(64, 32, bias=False)
        x = torch.randn(4, 64)

        out = layer(x)
        assert out.shape == (4, 32)

        kl = layer.kl_divergence()
        assert torch.isfinite(kl)


# ---------------------------------------------------------------------------
# BayesianWrapper tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBayesianWrapper:
    """BayesianWrapper model conversion tests."""

    def test_wrapper_converts_model(self):
        """BayesianWrapper replaces Linear layers with BayesianLinear."""
        base_model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

        wrapper = BayesianWrapper(base_model)

        # Check that linear layers were replaced
        assert len(wrapper._bayesian_layers) >= 2, (
            f"Expected at least 2 replaced layers, got {len(wrapper._bayesian_layers)}"
        )

    def test_wrapper_forward_returns_kl(self):
        """BayesianWrapper forward returns (output, kl) tuple."""
        base_model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        wrapper = BayesianWrapper(base_model)

        x = torch.randn(4, 64)
        output, kl = wrapper(x)

        assert output.shape == (4, 10), f"Expected (4, 10), got {output.shape}"
        assert kl.item() >= 0, "KL should be non-negative"
        assert torch.isfinite(kl), "KL should be finite"

    def test_elbo_loss(self):
        """ELBO loss computation works correctly."""
        base_model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        wrapper = BayesianWrapper(base_model)

        x = torch.randn(4, 64)
        output, kl = wrapper(x)

        nll = nn.functional.cross_entropy(
            output, torch.randint(0, 10, (4,))
        )

        elbo = wrapper.elbo_loss(nll, n_samples=100)
        assert torch.isfinite(elbo), "ELBO should be finite"

    def test_target_modules_filtering(self):
        """Only target modules are replaced."""

        class TwoHeadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(64, 32)
                self.classifier = nn.Linear(32, 10)

            def forward(self, x):
                return self.classifier(torch.relu(self.encoder(x)))

        model = TwoHeadModel()
        wrapper = BayesianWrapper(model, target_modules=["classifier"])

        assert len(wrapper._bayesian_layers) == 1, (
            "Only the classifier layer should be replaced"
        )
        # Verify the encoder is still a regular Linear
        assert isinstance(wrapper.model.encoder, nn.Linear)
        assert not isinstance(wrapper.model.encoder, BayesianLinear)

    def test_posterior_predictive(self):
        """Posterior predictive returns uncertainty estimates."""
        base_model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        wrapper = BayesianWrapper(base_model)

        x = torch.randn(4, 64)
        result = wrapper.posterior_predictive(x, n_samples=5)

        assert "mean" in result
        assert "variance" in result
        assert "std" in result
        assert "entropy" in result
        assert (result["variance"] >= 0).all(), "Variance should be non-negative"
