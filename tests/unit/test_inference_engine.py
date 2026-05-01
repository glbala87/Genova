"""Unit tests for the InferenceEngine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from genova.api.inference import InferenceEngine
from genova.utils.config import GenovaConfig, ModelConfig


@pytest.fixture
def small_config():
    """Create a minimal GenovaConfig for testing."""
    config = GenovaConfig()
    config.model = ModelConfig(
        arch="transformer",
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        vocab_size=10,
        max_position_embeddings=128,
        dropout=0.0,
        attention_dropout=0.0,
    )
    return config


@pytest.fixture
def engine(small_config):
    """Create an InferenceEngine with a tiny model."""
    eng = InferenceEngine(
        model_path=None,
        config=small_config,
        device="cpu",
        max_batch_size=4,
        max_sequence_length=64,
    )
    return eng


class TestEngineInit:
    """Test engine initialization."""

    def test_init_without_model_path(self, engine):
        assert engine.model_path is None
        assert engine.is_loaded() is False
        assert engine.device == torch.device("cpu")

    def test_init_with_max_batch_size(self, engine):
        assert engine.max_batch_size == 4

    def test_init_with_max_sequence_length(self, engine):
        assert engine.max_sequence_length == 64


class TestEngineLoadUnload:
    """Test model loading and unloading."""

    def test_load_creates_model(self, engine):
        engine.load()
        assert engine.is_loaded() is True
        assert engine.model is not None
        assert engine.tokenizer is not None

    def test_double_load_is_noop(self, engine):
        engine.load()
        model_id = id(engine.model)
        engine.load()
        assert id(engine.model) == model_id

    def test_unload_frees_model(self, engine):
        engine.load()
        engine.unload()
        assert engine.model is None
        assert engine.tokenizer is None
        assert engine.is_loaded() is False


class TestEngineInference:
    """Test inference methods on a loaded engine."""

    @pytest.fixture(autouse=True)
    def loaded_engine(self, engine):
        engine.load()
        return engine

    def test_embed_returns_list(self, engine):
        results = engine.embed(["ACGT", "GGCC"])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_embed_returns_numpy_arrays(self, engine):
        results = engine.embed(["ACGT"])
        assert isinstance(results[0], np.ndarray)

    def test_embed_dimension_matches_d_model(self, engine):
        results = engine.embed(["ACGTACGT"])
        assert results[0].shape == (32,)

    def test_embed_pooling_strategies(self, engine):
        for pooling in ["mean", "cls", "max"]:
            results = engine.embed(["ACGT"], pooling=pooling)
            assert len(results) == 1
            assert results[0].shape == (32,)

    def test_predict_variant_returns_results(self, engine):
        results = engine.predict_variant(
            ref_sequences=["ACGTACGT"],
            alt_sequences=["ACGGACGT"],
        )
        assert len(results) == 1
        assert "score" in results[0]
        assert "label" in results[0]
        assert "confidence" in results[0]

    def test_predict_variant_score_range(self, engine):
        results = engine.predict_variant(
            ref_sequences=["ACGT"],
            alt_sequences=["GCGT"],
        )
        assert 0.0 <= results[0]["score"] <= 1.0
        assert 0.0 <= results[0]["confidence"] <= 1.0

    def test_predict_variant_label_values(self, engine):
        results = engine.predict_variant(
            ref_sequences=["ACGT"],
            alt_sequences=["GCGT"],
        )
        assert results[0]["label"] in ("benign", "pathogenic")

    def test_predict_variant_mismatched_lengths_raises(self, engine):
        with pytest.raises(AssertionError):
            engine.predict_variant(
                ref_sequences=["ACGT", "GGCC"],
                alt_sequences=["GCGT"],
            )

    def test_predict_expression(self, engine):
        results = engine.predict_expression(["ACGTACGT"], num_targets=2)
        assert len(results) == 1
        assert isinstance(results[0], np.ndarray)

    def test_predict_methylation(self, engine):
        results = engine.predict_methylation(["ACGTACGT"], num_targets=1)
        assert len(results) == 1

    def test_get_model_info(self, engine):
        info = engine.get_model_info()
        assert info["architecture"] == "transformer"
        assert info["d_model"] == 32
        assert info["n_layers"] == 1
        assert info["n_heads"] == 2


class TestEngineNotLoaded:
    """Test that methods raise when model is not loaded."""

    def test_embed_raises(self, engine):
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.embed(["ACGT"])

    def test_predict_variant_raises(self, engine):
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.predict_variant(["ACGT"], ["GCGT"])

    def test_get_model_info_raises(self, engine):
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.get_model_info()
