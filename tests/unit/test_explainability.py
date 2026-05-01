"""Tests for explainability modules: integrated gradients and attention analysis."""

from __future__ import annotations

import pytest
import torch

from genova.utils.config import ModelConfig
from genova.models.transformer import GenovaTransformer
from genova.data.tokenizer import GenomicTokenizer
from genova.explainability.integrated_gradients import IntegratedGradientsExplainer
from genova.explainability.attention_analysis import AttentionAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_config():
    return ModelConfig(
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


@pytest.fixture
def tiny_model(tiny_config):
    model = GenovaTransformer(tiny_config)
    model.eval()
    return model


@pytest.fixture
def tokenizer():
    tok = GenomicTokenizer(mode="nucleotide")
    tok.build_vocab()
    return tok


# ---------------------------------------------------------------------------
# IntegratedGradientsExplainer
# ---------------------------------------------------------------------------


class TestIntegratedGradientsExplainer:

    def test_creation(self, tiny_model, tokenizer):
        explainer = IntegratedGradientsExplainer(tiny_model, tokenizer)
        assert explainer is not None

    def test_explain_returns_attributions(self, tiny_model, tokenizer):
        explainer = IntegratedGradientsExplainer(tiny_model, tokenizer)
        seq = "ACGTACGT"
        result = explainer.explain(seq)
        assert result is not None
        assert isinstance(result, dict)
        # Should have attributions or scores key
        assert len(result) > 0


# ---------------------------------------------------------------------------
# AttentionAnalyzer
# ---------------------------------------------------------------------------


class TestAttentionAnalyzer:

    def test_creation(self, tiny_model, tokenizer):
        analyzer = AttentionAnalyzer(tiny_model, tokenizer)
        assert analyzer is not None

    def test_extract_attention(self, tiny_model, tokenizer):
        analyzer = AttentionAnalyzer(tiny_model, tokenizer)
        result = analyzer.extract_attention("ACGTACGT")
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0  # Should have attention from at least one layer
