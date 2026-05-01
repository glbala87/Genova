"""Extended tests for generative modules: autoregressive, beam search, infilling."""

from __future__ import annotations

import pytest
import torch

from genova.utils.config import ModelConfig
from genova.models.model_factory import create_model
from genova.data.tokenizer import GenomicTokenizer
from genova.generative.autoregressive import AutoregressiveGenerator
from genova.generative.beam_search import BeamSearchGenerator
from genova.generative.infilling import SequenceInfiller


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    tok = GenomicTokenizer(mode="nucleotide")
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
        max_position_embeddings=128,
        rotary_emb=False,
        flash_attention=False,
        gradient_checkpointing=False,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=0,
    )


@pytest.fixture
def tiny_model(tiny_config):
    model = create_model(tiny_config, task="mlm")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# AutoregressiveGenerator
# ---------------------------------------------------------------------------


class TestAutoregressiveGenerator:

    def test_creation(self, tiny_model, tokenizer):
        gen = AutoregressiveGenerator(tiny_model, tokenizer=tokenizer)
        assert gen is not None

    def test_generate_returns_sequences(self, tiny_model, tokenizer):
        gen = AutoregressiveGenerator(tiny_model, tokenizer=tokenizer)
        result = gen.generate(num_sequences=2, max_length=20, temperature=1.0)
        if isinstance(result, dict):
            assert "sequences" in result
            assert len(result["sequences"]) == 2
        elif isinstance(result, list):
            assert len(result) == 2

    def test_generate_respects_max_length(self, tiny_model, tokenizer):
        gen = AutoregressiveGenerator(tiny_model, tokenizer=tokenizer)
        result = gen.generate(num_sequences=1, max_length=10, temperature=1.0)
        if isinstance(result, dict):
            seqs = result["sequences"]
        else:
            seqs = result
        for seq in seqs:
            if isinstance(seq, str):
                assert len(seq) <= 10


# ---------------------------------------------------------------------------
# BeamSearchGenerator
# ---------------------------------------------------------------------------


class TestBeamSearchGenerator:

    def test_creation(self, tiny_model, tokenizer):
        gen = BeamSearchGenerator(tiny_model, tokenizer)
        assert gen is not None

    def test_generate_returns_results(self, tiny_model, tokenizer):
        gen = BeamSearchGenerator(tiny_model, tokenizer)
        result = gen.generate(beam_width=2, max_length=10)
        assert result is not None
        if isinstance(result, list):
            assert len(result) > 0


# ---------------------------------------------------------------------------
# SequenceInfiller
# ---------------------------------------------------------------------------


class TestSequenceInfiller:

    def test_creation(self, tiny_model, tokenizer):
        infiller = SequenceInfiller(tiny_model, tokenizer=tokenizer)
        assert infiller is not None

    def test_infill_returns_result(self, tiny_model, tokenizer):
        infiller = SequenceInfiller(tiny_model, tokenizer=tokenizer)
        result = infiller.infill(
            prefix="ACGT",
            suffix="TGCA",
            max_length=8,
        )
        assert result is not None
