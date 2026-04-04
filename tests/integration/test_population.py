"""Integration test: population-aware model.

Tests PopulationAwareEncoder, PopulationAwareVariantPredictor, and
AlleleFrequencyEncoder with synthetic data.
"""

from __future__ import annotations

import pytest
import torch

from genova.utils.config import ModelConfig
from genova.population.population_model import (
    PopulationAwareEncoder,
    PopulationAwareVariantPredictor,
    PopulationEmbedding,
    VariantFrequencyEncoder,
)
from genova.population.frequency_encoder import AlleleFrequencyEncoder


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
        dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=128,
        rotary_emb=False,
        flash_attention=False,
        gradient_checkpointing=False,
        vocab_size=32,
        pad_token_id=0,
    )


@pytest.fixture
def sample_input():
    B, L = 2, 16
    input_ids = torch.randint(5, 30, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    return input_ids, attention_mask


# ---------------------------------------------------------------------------
# PopulationEmbedding tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPopulationEmbedding:
    """PopulationEmbedding standalone tests."""

    def test_forward_shape(self):
        emb = PopulationEmbedding(num_populations=7, embedding_dim=64)
        pop_ids = torch.tensor([1, 3, 5])
        out = emb(pop_ids)

        assert out.shape == (3, 64), f"Expected (3, 64), got {out.shape}"

    def test_label_to_index(self):
        emb = PopulationEmbedding(num_populations=7)

        assert emb.label_to_index("EUR") == 1
        assert emb.label_to_index("AFR") == 2
        assert emb.label_to_index("UNKNOWN") == 0
        assert emb.label_to_index("nonexistent") == 0  # maps to UNKNOWN

    def test_labels_to_tensor(self):
        emb = PopulationEmbedding(num_populations=7)

        t = emb.labels_to_tensor(["EUR", "AFR", "UNKNOWN"])
        assert t.shape == (3,)
        assert t[0].item() == 1  # EUR
        assert t[1].item() == 2  # AFR
        assert t[2].item() == 0  # UNKNOWN


# ---------------------------------------------------------------------------
# PopulationAwareEncoder tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPopulationAwareEncoder:
    """PopulationAwareEncoder forward pass tests."""

    def test_forward_without_population(self, tiny_config, sample_input):
        """Forward pass works without population context."""
        encoder = PopulationAwareEncoder(
            tiny_config,
            num_populations=7,
            num_af_features=7,
            population_embedding_dim=32,
        )
        input_ids, attention_mask = sample_input

        out = encoder(input_ids, attention_mask=attention_mask)

        B, L = input_ids.shape
        assert "last_hidden_state" in out
        assert out["last_hidden_state"].shape == (B, L, tiny_config.d_model)

    def test_forward_with_population(self, tiny_config, sample_input):
        """Forward pass with population labels."""
        encoder = PopulationAwareEncoder(
            tiny_config,
            num_populations=7,
            num_af_features=7,
            population_embedding_dim=32,
        )
        input_ids, attention_mask = sample_input
        B = input_ids.shape[0]
        pop_ids = torch.tensor([1, 3])  # EUR, SAS

        out = encoder(
            input_ids,
            attention_mask=attention_mask,
            population_ids=pop_ids,
        )

        assert "last_hidden_state" in out
        assert "population_embedding" in out
        assert out["population_embedding"].shape == (B, tiny_config.d_model)

    def test_forward_with_af_features(self, tiny_config, sample_input):
        """Forward pass with allele frequency features."""
        num_af = 7
        encoder = PopulationAwareEncoder(
            tiny_config,
            num_populations=7,
            num_af_features=num_af,
            population_embedding_dim=32,
        )
        input_ids, attention_mask = sample_input
        B = input_ids.shape[0]
        af_features = torch.randn(B, num_af)

        out = encoder(
            input_ids,
            attention_mask=attention_mask,
            af_features=af_features,
        )

        assert "last_hidden_state" in out
        assert "af_embedding" in out

    def test_forward_with_all_context(self, tiny_config, sample_input):
        """Forward pass with both population and AF context."""
        num_af = 7
        encoder = PopulationAwareEncoder(
            tiny_config,
            num_populations=7,
            num_af_features=num_af,
            population_embedding_dim=32,
        )
        input_ids, attention_mask = sample_input
        B = input_ids.shape[0]
        pop_ids = torch.tensor([2, 5])
        af_features = torch.randn(B, num_af)

        out = encoder(
            input_ids,
            attention_mask=attention_mask,
            population_ids=pop_ids,
            af_features=af_features,
        )

        assert "last_hidden_state" in out
        assert "population_embedding" in out
        assert "af_embedding" in out


# ---------------------------------------------------------------------------
# PopulationAwareVariantPredictor tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPopulationAwareVariantPredictor:
    """PopulationAwareVariantPredictor end-to-end tests."""

    def test_forward_without_labels(self, tiny_config, sample_input):
        """Forward pass without labels produces logits and probabilities."""
        predictor = PopulationAwareVariantPredictor(
            tiny_config,
            num_populations=7,
            num_af_features=7,
            num_variant_classes=5,
        )
        input_ids, attention_mask = sample_input
        B = input_ids.shape[0]
        pop_ids = torch.tensor([1, 4])

        out = predictor(
            input_ids,
            attention_mask=attention_mask,
            population_ids=pop_ids,
        )

        assert "logits" in out
        assert "probabilities" in out
        assert out["logits"].shape == (B, 5), (
            f"Expected logits shape (2, 5), got {out['logits'].shape}"
        )
        assert out["probabilities"].shape == (B, 5)

        # Probabilities should sum to 1
        prob_sums = out["probabilities"].sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(B), atol=1e-5), (
            f"Probabilities should sum to 1, got {prob_sums}"
        )

    def test_forward_with_labels(self, tiny_config, sample_input):
        """Forward pass with labels computes loss."""
        predictor = PopulationAwareVariantPredictor(
            tiny_config,
            num_populations=7,
            num_af_features=7,
            num_variant_classes=5,
        )
        input_ids, attention_mask = sample_input
        B = input_ids.shape[0]
        labels = torch.randint(0, 5, (B,))
        pop_ids = torch.tensor([1, 4])
        af_features = torch.randn(B, 7)

        out = predictor(
            input_ids,
            attention_mask=attention_mask,
            population_ids=pop_ids,
            af_features=af_features,
            labels=labels,
        )

        assert "loss" in out, "Should contain loss when labels provided"
        assert torch.isfinite(out["loss"]), "Loss should be finite"

    def test_gradient_flow(self, tiny_config, sample_input):
        """Gradients flow through the full population-aware predictor."""
        predictor = PopulationAwareVariantPredictor(
            tiny_config,
            num_populations=7,
            num_af_features=7,
            num_variant_classes=5,
        )
        predictor.train()
        input_ids, attention_mask = sample_input
        B = input_ids.shape[0]
        labels = torch.randint(0, 5, (B,))
        pop_ids = torch.tensor([1, 4])
        af_features = torch.randn(B, 7)

        out = predictor(
            input_ids,
            attention_mask=attention_mask,
            population_ids=pop_ids,
            af_features=af_features,
            labels=labels,
        )
        out["loss"].backward()

        grad_count = sum(
            1 for p in predictor.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0, "Should have non-zero gradients"


# ---------------------------------------------------------------------------
# AlleleFrequencyEncoder tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAlleleFrequencyEncoder:
    """AlleleFrequencyEncoder with synthetic data."""

    def test_encode_single_variant(self):
        """Encode a single variant's allele frequencies."""
        encoder = AlleleFrequencyEncoder(
            populations=["EUR", "AFR", "EAS"],
            include_global_af=True,
        )

        af_values = {
            "EUR": 0.01,
            "AFR": 0.05,
            "EAS": 0.001,
            "GLOBAL": 0.02,
        }
        features, mask = encoder.encode_frequencies(af_values)

        # 3 populations + 1 global = 4 features
        assert features.shape == (4,), f"Expected (4,), got {features.shape}"
        assert mask.shape == (4,)
        assert mask.all(), "All values provided, mask should be all True"

    def test_encode_with_missing_values(self):
        """Missing AF values are imputed and masked."""
        encoder = AlleleFrequencyEncoder(
            populations=["EUR", "AFR", "EAS"],
            include_global_af=True,
        )

        af_values = {"EUR": 0.01}  # AFR, EAS, GLOBAL missing
        features, mask = encoder.encode_frequencies(af_values)

        assert features.shape == (4,)
        assert mask[0] is True or mask[0].item() is True  # EUR observed
        assert not mask[1], "AFR should be masked (missing)"
        assert not mask[2], "EAS should be masked (missing)"

    def test_encode_batch(self):
        """Batch encoding of multiple variants."""
        encoder = AlleleFrequencyEncoder(
            populations=["EUR", "AFR"],
            include_global_af=False,
        )

        records = [
            {"EUR": 0.01, "AFR": 0.02},
            {"EUR": 0.1, "AFR": None},
            {"EUR": None, "AFR": 0.001},
        ]
        features, mask = encoder.encode_batch(records)

        assert features.shape == (3, 2), f"Expected (3, 2), got {features.shape}"
        assert mask.shape == (3, 2)

    def test_num_features_property(self):
        """num_features reflects populations + optional global."""
        enc_with = AlleleFrequencyEncoder(
            populations=["EUR", "AFR", "EAS"],
            include_global_af=True,
        )
        assert enc_with.num_features == 4

        enc_without = AlleleFrequencyEncoder(
            populations=["EUR", "AFR", "EAS"],
            include_global_af=False,
        )
        assert enc_without.num_features == 3

    def test_log_scaling(self):
        """Log scaling produces reasonable values."""
        encoder = AlleleFrequencyEncoder(
            populations=["EUR"],
            include_global_af=False,
            log_base=10.0,
        )

        features_common, _ = encoder.encode_frequencies({"EUR": 0.5})
        features_rare, _ = encoder.encode_frequencies({"EUR": 1e-6})

        # Common variants should have higher (less negative) log-scaled values
        assert features_common[0] > features_rare[0], (
            f"Common AF ({features_common[0]:.2f}) should have higher "
            f"log-scaled value than rare AF ({features_rare[0]:.2f})"
        )
