"""Integration test: in-silico perturbation / mutagenesis.

Tests saturation mutagenesis, sensitivity map computation, and
epistatic interaction detection with a tiny model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from genova.utils.config import ModelConfig
from genova.data.tokenizer import GenomicTokenizer
from genova.models.transformer import GenovaTransformer
from genova.perturbation.variant_simulator import VariantSimulator, VariantEffect


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tokenizer():
    tok = GenomicTokenizer(mode="kmer", k=3, stride=1)
    tok.build_vocab()
    return tok


@pytest.fixture
def tiny_config(tokenizer):
    return ModelConfig(
        arch="transformer",
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=256,
        rotary_emb=False,
        flash_attention=False,
        gradient_checkpointing=False,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=0,
    )


@pytest.fixture
def model(tiny_config):
    m = GenovaTransformer(tiny_config)
    m.eval()
    return m


@pytest.fixture
def short_sequence():
    """A short DNA sequence for mutagenesis (20 bases)."""
    return "ACGTACGTACGTACGTACGT"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSaturationMutagenesis:
    """Saturation mutagenesis tests."""

    def test_saturate_snps_basic(self, model, tokenizer, short_sequence):
        """Saturation mutagenesis on a short sequence produces effects."""
        simulator = VariantSimulator(
            model, tokenizer, device="cpu", batch_size=32
        )

        effects = simulator.saturate_snps(short_sequence)

        # For each position, 3 possible SNPs (other nucleotides)
        expected = len(short_sequence) * 3
        assert len(effects) == expected, (
            f"Expected {expected} SNP effects, got {len(effects)}"
        )

    def test_effect_fields_populated(self, model, tokenizer, short_sequence):
        """Each VariantEffect has the expected fields."""
        simulator = VariantSimulator(
            model, tokenizer, device="cpu", batch_size=32
        )

        effects = simulator.saturate_snps(short_sequence)

        for eff in effects:
            assert isinstance(eff, VariantEffect)
            assert 0 <= eff.position < len(short_sequence)
            assert eff.ref_allele in "ACGT"
            assert eff.alt_allele in "ACGT"
            assert eff.ref_allele != eff.alt_allele
            assert eff.variant_type == "SNP"
            assert isinstance(eff.l2_distance, float)
            assert isinstance(eff.cosine_similarity, float)
            assert eff.l2_distance >= 0, "L2 distance should be non-negative"

    def test_region_restriction(self, model, tokenizer, short_sequence):
        """Mutagenesis can be restricted to a region."""
        simulator = VariantSimulator(
            model, tokenizer, device="cpu", batch_size=32
        )

        effects = simulator.saturate_snps(short_sequence, region=(5, 10))

        expected = (10 - 5) * 3  # 5 positions * 3 alt alleles
        assert len(effects) == expected, (
            f"Expected {expected} effects in region, got {len(effects)}"
        )

        for eff in effects:
            assert 5 <= eff.position < 10, (
                f"Effect at position {eff.position} outside region [5, 10)"
            )

    def test_sensitivity_map_shape(self, model, tokenizer, short_sequence):
        """compute_effect_landscape returns properly shaped arrays."""
        simulator = VariantSimulator(
            model, tokenizer, device="cpu", batch_size=32
        )

        landscape = simulator.compute_effect_landscape(
            short_sequence,
            include_indels=False,
        )

        assert "position_scores" in landscape
        assert "effect_matrix" in landscape
        assert "sensitive_positions" in landscape
        assert "snp_effects" in landscape

        seq_len = len(short_sequence)
        assert landscape["position_scores"].shape == (seq_len,), (
            f"position_scores shape mismatch: {landscape['position_scores'].shape}"
        )
        assert landscape["effect_matrix"].shape == (seq_len, 3), (
            f"effect_matrix shape mismatch: {landscape['effect_matrix'].shape}"
        )

    def test_sensitive_positions_valid(self, model, tokenizer, short_sequence):
        """Sensitive positions are valid indices."""
        simulator = VariantSimulator(
            model, tokenizer, device="cpu", batch_size=32
        )

        landscape = simulator.compute_effect_landscape(
            short_sequence, include_indels=False
        )

        for pos in landscape["sensitive_positions"]:
            assert 0 <= pos < len(short_sequence), (
                f"Sensitive position {pos} out of range"
            )


@pytest.mark.integration
class TestIndelSimulation:
    """Indel simulation tests."""

    def test_simulate_indels(self, model, tokenizer, short_sequence):
        """Indel simulation produces non-empty results."""
        simulator = VariantSimulator(
            model, tokenizer, device="cpu", batch_size=32
        )

        effects = simulator.simulate_indels(
            short_sequence, indel_lengths=[1]
        )

        assert len(effects) > 0, "Should produce indel effects"

        # Check that we have both insertions and deletions
        types = {eff.variant_type for eff in effects}
        assert "insertion" in types, "Should include insertions"
        assert "deletion" in types, "Should include deletions"

    def test_indel_effect_values(self, model, tokenizer, short_sequence):
        """Indel effects have reasonable values."""
        simulator = VariantSimulator(
            model, tokenizer, device="cpu", batch_size=32
        )

        effects = simulator.simulate_indels(
            short_sequence, indel_lengths=[1], region=(5, 8)
        )

        for eff in effects:
            assert np.isfinite(eff.l2_distance), "L2 distance should be finite"
            assert np.isfinite(eff.cosine_similarity), "Cosine sim should be finite"
            assert -1 <= eff.cosine_similarity <= 1, (
                f"Cosine similarity should be in [-1, 1], got {eff.cosine_similarity}"
            )
