"""Tests for multi-omics integration modules."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from genova.multiomics.data_fusion import MultiOmicsAligner, OmicsDataFusion
from genova.multiomics.multiomics_model import MultiOmicsGenovaModel
from genova.multiomics.ont_methylation import ONTMethylationProcessor
from genova.utils.config import ModelConfig
from genova.models.transformer import GenovaTransformer
from genova.data.tokenizer import GenomicTokenizer


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
    torch.manual_seed(42)
    return GenovaTransformer(tiny_config)


@pytest.fixture
def tokenizer():
    tok = GenomicTokenizer(mode="nucleotide")
    tok.build_vocab()
    return tok


# ---------------------------------------------------------------------------
# MultiOmicsAligner
# ---------------------------------------------------------------------------


class TestMultiOmicsAligner:

    def test_creation(self):
        aligner = MultiOmicsAligner()
        assert aligner is not None

    def test_attributes(self):
        aligner = MultiOmicsAligner(window_size=256, max_methylation_sites=64)
        assert aligner.window_size == 256
        assert aligner.max_methylation_sites == 64


# ---------------------------------------------------------------------------
# OmicsDataFusion
# ---------------------------------------------------------------------------


class TestOmicsDataFusion:

    def test_creation(self):
        fusion = OmicsDataFusion(d_model=64, num_modalities=2)
        assert fusion is not None

    def test_forward(self):
        fusion = OmicsDataFusion(d_model=64, num_modalities=2)
        # OmicsDataFusion expects list of tensors
        modalities = [torch.randn(2, 16, 64), torch.randn(2, 16, 64)]
        result = fusion(modalities)
        assert result is not None


# ---------------------------------------------------------------------------
# MultiOmicsGenovaModel
# ---------------------------------------------------------------------------


class TestMultiOmicsGenovaModel:

    def test_creation(self, tiny_config):
        model = MultiOmicsGenovaModel(tiny_config)
        assert model is not None

    def test_dna_only_forward(self, tiny_config):
        model = MultiOmicsGenovaModel(tiny_config)
        model.eval()
        input_ids = torch.randint(1, 10, (2, 16))
        with torch.no_grad():
            result = model(input_ids=input_ids)
        assert result is not None


# ---------------------------------------------------------------------------
# ONTMethylationProcessor
# ---------------------------------------------------------------------------


class TestONTMethylationProcessor:

    @pytest.fixture
    def sample_bedmethyl(self, tmp_path):
        lines = [
            "chr1\t1000\t1001\tm\t1000\t+\t1000\t1001\t0,0,0\t20\t80.0",
            "chr1\t2000\t2001\tm\t1000\t+\t2000\t2001\t0,0,0\t30\t50.0",
            "chr1\t3000\t3001\tm\t1000\t+\t3000\t3001\t0,0,0\t2\t90.0",
            "chr1\t4000\t4001\tm\t1000\t+\t4000\t4001\t0,0,0\t10\t0.0",
            "chr1\t5000\t5001\tm\t1000\t+\t5000\t5001\t0,0,0\t25\t100.0",
        ]
        bed_file = tmp_path / "test.bedmethyl"
        bed_file.write_text("\n".join(lines) + "\n")
        return bed_file

    def test_process_file(self, sample_bedmethyl):
        proc = ONTMethylationProcessor(min_coverage=5)
        stats = proc.process_file(sample_bedmethyl)
        assert stats["total_sites"] == 5

    def test_coverage_filtering(self, sample_bedmethyl):
        proc = ONTMethylationProcessor(min_coverage=5)
        proc.process_file(sample_bedmethyl)
        # Site at 3000 has coverage=2 < 5, should be filtered
        assert proc.total_sites < 5

    def test_window_features(self, sample_bedmethyl):
        proc = ONTMethylationProcessor(min_coverage=5)
        proc.process_file(sample_bedmethyl)
        features = proc.get_window_features("chr1", 0, 10000, max_sites=32)
        assert features["beta_values"].shape == (32,)
        assert features["positions"].shape == (32,)
        assert features["mask"].shape == (32,)

    def test_beta_values_in_range(self, sample_bedmethyl):
        proc = ONTMethylationProcessor(min_coverage=5)
        proc.process_file(sample_bedmethyl)
        features = proc.get_window_features("chr1", 0, 10000)
        beta_vals = features["beta_values"]
        mask = features["mask"]
        real_betas = beta_vals[mask]
        if len(real_betas) > 0:
            assert (real_betas >= 0.0).all()
            assert (real_betas <= 1.0).all()
