"""Tests for the GenovaConfig system."""

from __future__ import annotations

import pytest
import yaml

from genova.utils.config import (
    GenovaConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    _apply_overrides,
    _cast_value,
)


# ---------------------------------------------------------------------------
# Default construction
# ---------------------------------------------------------------------------

class TestDefaults:

    def test_default_construction(self):
        cfg = GenovaConfig()
        assert cfg.model.d_model == 768
        assert cfg.model.n_heads == 12
        assert cfg.data.seq_length == 512
        assert cfg.training.lr == 1e-4

    def test_data_defaults(self):
        d = DataConfig()
        assert d.tokenizer == "kmer"
        assert d.kmer_size == 6
        assert d.mask_prob == 0.15

    def test_model_defaults(self):
        m = ModelConfig()
        assert m.activation == "gelu"
        assert m.rotary_emb is False
        assert m.tie_word_embeddings is True

    def test_training_defaults(self):
        t = TrainingConfig()
        assert t.lr_scheduler == "cosine"
        assert t.seed == 42

    def test_evaluation_defaults(self):
        e = EvaluationConfig()
        assert "perplexity" in e.metrics
        assert "promoter_detection" in e.downstream_tasks


# ---------------------------------------------------------------------------
# from_dict / to_dict
# ---------------------------------------------------------------------------

class TestDictRoundtrip:

    def test_from_dict(self, sample_config_dict):
        cfg = GenovaConfig.from_dict(sample_config_dict)
        assert cfg.data.seq_length == 256
        assert cfg.model.d_model == 64
        assert cfg.training.lr == 3e-4

    def test_to_dict(self):
        cfg = GenovaConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "data" in d and "model" in d and "training" in d
        assert d["model"]["d_model"] == 768

    def test_roundtrip(self, sample_config_dict):
        cfg = GenovaConfig.from_dict(sample_config_dict)
        d = cfg.to_dict()
        cfg2 = GenovaConfig.from_dict(d)
        assert cfg2.to_dict() == d

    def test_unknown_keys_ignored(self):
        d = {"model": {"d_model": 128, "nonexistent_key": 999}}
        cfg = GenovaConfig.from_dict(d)
        assert cfg.model.d_model == 128


# ---------------------------------------------------------------------------
# YAML load
# ---------------------------------------------------------------------------

class TestYaml:

    def test_load_from_yaml(self, sample_yaml_path):
        cfg = GenovaConfig.from_yaml(sample_yaml_path)
        assert cfg.data.seq_length == 256
        assert cfg.model.n_layers == 2

    def test_missing_yaml_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            GenovaConfig.from_yaml(tmp_path / "missing.yaml")

    def test_save_yaml(self, tmp_path):
        cfg = GenovaConfig()
        path = tmp_path / "out.yaml"
        cfg.save_yaml(path)
        loaded = GenovaConfig.from_yaml(path)
        assert loaded.to_dict() == cfg.to_dict()


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------

class TestOverrides:

    def test_simple_override(self, sample_yaml_path):
        cfg = GenovaConfig.from_yaml(
            sample_yaml_path,
            overrides=["model.d_model=128", "training.lr=5e-4"],
        )
        assert cfg.model.d_model == 128
        assert cfg.training.lr == 5e-4

    def test_bool_override(self, sample_yaml_path):
        cfg = GenovaConfig.from_yaml(
            sample_yaml_path,
            overrides=["model.rotary_emb=false"],
        )
        assert cfg.model.rotary_emb is False

    def test_override_missing_equals_raises(self):
        with pytest.raises(ValueError, match="key=value"):
            _apply_overrides({}, ["bad_override"])


# ---------------------------------------------------------------------------
# _cast_value
# ---------------------------------------------------------------------------

class TestCastValue:

    @pytest.mark.parametrize("raw,expected", [
        ("true", True),
        ("False", False),
        ("none", None),
        ("42", 42),
        ("3.14", 3.14),
        ("hello", "hello"),
        ("1e-4", 1e-4),
    ])
    def test_cast(self, raw, expected):
        assert _cast_value(raw) == expected
