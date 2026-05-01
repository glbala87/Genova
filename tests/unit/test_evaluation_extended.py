"""Extended tests for evaluation modules: bias audit, TF binding, chromatin, structural variants."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from genova.utils.config import ModelConfig
from genova.models.transformer import GenovaTransformer
from genova.data.tokenizer import GenomicTokenizer
from genova.evaluation.bias_audit import BiasAuditor
from genova.evaluation.tf_binding import TFBindingPredictor
from genova.evaluation.chromatin import ChromatinStatePredictor
from genova.evaluation.structural_variants import StructuralVariantPredictor


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
# BiasAuditor
# ---------------------------------------------------------------------------


class TestBiasAuditor:

    def test_creation(self):
        auditor = BiasAuditor()
        assert auditor is not None

    def test_creation_with_model(self, tiny_model):
        auditor = BiasAuditor(model=tiny_model)
        assert auditor is not None


# ---------------------------------------------------------------------------
# TFBindingPredictor (takes encoder, num_tfs, d_model)
# ---------------------------------------------------------------------------


class TestTFBindingPredictor:

    def test_creation(self, tiny_model):
        predictor = TFBindingPredictor(
            encoder=tiny_model,
            num_tfs=3,
            d_model=64,
        )
        assert predictor is not None

    def test_forward_shape(self, tiny_model):
        predictor = TFBindingPredictor(
            encoder=tiny_model,
            num_tfs=3,
            d_model=64,
        )
        input_ids = torch.randint(1, 10, (2, 16))
        with torch.no_grad():
            result = predictor(input_ids)
        assert result is not None


# ---------------------------------------------------------------------------
# ChromatinStatePredictor (takes encoder, num_marks, d_model)
# ---------------------------------------------------------------------------


class TestChromatinStatePredictor:

    def test_creation(self, tiny_model):
        predictor = ChromatinStatePredictor(
            encoder=tiny_model,
            num_marks=5,
            d_model=64,
        )
        assert predictor is not None


# ---------------------------------------------------------------------------
# StructuralVariantPredictor (takes model, tokenizer)
# ---------------------------------------------------------------------------


class TestStructuralVariantPredictor:

    def test_creation(self, tiny_model, tokenizer):
        predictor = StructuralVariantPredictor(
            model=tiny_model,
            tokenizer=tokenizer,
        )
        assert predictor is not None
