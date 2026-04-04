"""Shared pytest fixtures for Genova tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import yaml

from genova.utils.config import GenovaConfig, ModelConfig, DataConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Sample DNA sequences
# ---------------------------------------------------------------------------

@pytest.fixture
def short_dna_sequences():
    """A handful of short DNA sequences for tokenizer / augmentation tests."""
    return [
        "ACGTACGTACGT",
        "GGCCAATTGGCC",
        "NNACGTNNACGT",
        "TTTTAAAACCCC",
        "ATCGATCGATCG",
    ]


@pytest.fixture
def single_sequence():
    return "ACGTACGTACGTACGT"


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model_config():
    """A very small ModelConfig for fast CPU-only tests."""
    return ModelConfig(
        arch="transformer",
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=128,
        activation="gelu",
        norm_type="layernorm",
        rotary_emb=False,
        flash_attention=False,
        gradient_checkpointing=False,
        tie_word_embeddings=True,
        initializer_range=0.02,
        vocab_size=32,
        pad_token_id=0,
    )


@pytest.fixture
def tiny_training_config():
    """A minimal TrainingConfig for scheduler tests."""
    return TrainingConfig(
        lr=1e-3,
        min_lr=1e-5,
        warmup_steps=100,
        lr_scheduler="cosine",
        weight_decay=0.01,
    )


@pytest.fixture
def sample_config_dict():
    """A plain dict that can round-trip through GenovaConfig.from_dict."""
    return {
        "data": {
            "seq_length": 256,
            "kmer_size": 4,
            "batch_size": 8,
            "vocab_size": 512,
        },
        "model": {
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 128,
            "vocab_size": 512,
        },
        "training": {
            "lr": 3e-4,
            "epochs": 5,
            "warmup_steps": 50,
        },
        "evaluation": {
            "metrics": ["accuracy", "perplexity"],
        },
    }


# ---------------------------------------------------------------------------
# Temporary directories / files
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a clean temporary directory (pytest built-in)."""
    return tmp_path


@pytest.fixture
def sample_yaml_path(tmp_path, sample_config_dict):
    """Write sample_config_dict to a YAML file and return the path."""
    path = tmp_path / "config.yaml"
    with open(path, "w") as fh:
        yaml.dump(sample_config_dict, fh)
    return path
