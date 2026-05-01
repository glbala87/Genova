"""Extended tests for population-aware genomics modules."""

from __future__ import annotations

import math

import pytest
import torch

from genova.population.frequency_encoder import AlleleFrequencyEncoder
from genova.population.population_model import PopulationAwareVariantPredictor
from genova.utils.config import ModelConfig
from genova.models.transformer import GenovaTransformer
from genova.data.tokenizer import GenomicTokenizer


# ---------------------------------------------------------------------------
# AlleleFrequencyEncoder edge cases
# ---------------------------------------------------------------------------


class TestAlleleFrequencyEncoderExtended:

    def test_zero_af_uses_floor(self):
        """AF=0.0 should be floored to avoid log(0)."""
        encoder = AlleleFrequencyEncoder(
            populations=["EUR"], include_global_af=False
        )
        feat, mask = encoder.encode_frequencies({"EUR": 0.0})
        assert torch.isfinite(feat).all()

    def test_af_one(self):
        """AF=1.0 should encode to log10(1.0) = 0."""
        encoder = AlleleFrequencyEncoder(
            populations=["EUR"], log_base=10.0, include_global_af=False
        )
        feat, _ = encoder.encode_frequencies({"EUR": 1.0})
        assert abs(feat[0].item() - 0.0) < 1e-5

    def test_multiple_populations(self):
        pops = ["EUR", "AFR", "EAS", "SAS", "AMR"]
        encoder = AlleleFrequencyEncoder(populations=pops, include_global_af=True)
        af_dict = {p: 0.1 for p in pops}
        feat, mask = encoder.encode_frequencies(af_dict)
        assert feat.shape[0] == len(pops) + 1  # +1 for global
        assert mask.shape[0] == len(pops) + 1

    def test_ordering_preserved(self):
        """Feature ordering should follow population list order."""
        encoder = AlleleFrequencyEncoder(
            populations=["EUR", "AFR"], log_base=10.0, include_global_af=False
        )
        feat, _ = encoder.encode_frequencies({"EUR": 0.1, "AFR": 0.01})
        # EUR at index 0, AFR at index 1
        eur_val = feat[0].item()
        afr_val = feat[1].item()
        # log10(0.1) = -1, log10(0.01) = -2
        assert abs(eur_val - (-1.0)) < 1e-5
        assert abs(afr_val - (-2.0)) < 1e-5

    def test_num_features_with_global(self):
        pops = ["EUR", "AFR"]
        encoder = AlleleFrequencyEncoder(populations=pops, include_global_af=True)
        assert encoder.num_features == 3

    def test_num_features_without_global(self):
        pops = ["EUR", "AFR"]
        encoder = AlleleFrequencyEncoder(populations=pops, include_global_af=False)
        assert encoder.num_features == 2


# ---------------------------------------------------------------------------
# PopulationAwareVariantPredictor
# ---------------------------------------------------------------------------


class TestPopulationAwareVariantPredictor:

    @pytest.fixture
    def tiny_setup(self):
        config = ModelConfig(
            arch="transformer",
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            dropout=0.0,
            attention_dropout=0.0,
            max_position_embeddings=128,
            rotary_emb=False,
            flash_attention=False,
            gradient_checkpointing=False,
            vocab_size=10,
            pad_token_id=0,
        )
        model = GenovaTransformer(config)
        model.eval()
        tok = GenomicTokenizer(mode="nucleotide")
        tok.build_vocab()
        return model, tok, config

    def test_creation(self, tiny_setup):
        model, tok, config = tiny_setup
        predictor = PopulationAwareVariantPredictor(
            config=config,
            num_populations=2,
        )
        assert predictor is not None
