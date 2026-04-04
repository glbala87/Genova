"""Tests for Genova model components (CPU only)."""

from __future__ import annotations

import pytest
import torch

from genova.utils.config import ModelConfig
from genova.models.transformer import (
    GenovaTransformer,
    GenovaForMLM,
    MultiHeadSelfAttention,
    FeedForward,
    TransformerEncoderLayer,
    MLMHead,
)
from genova.models.embeddings import GenomicEmbedding, SinusoidalPositionalEncoding
from genova.models.model_factory import create_model, count_parameters, model_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return ModelConfig(
        arch="transformer",
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=128,
        flash_attention=False,
        rotary_emb=False,
        gradient_checkpointing=False,
        tie_word_embeddings=True,
        vocab_size=32,
        pad_token_id=0,
    )


@pytest.fixture
def batch():
    """A small (B=2, L=16) batch of random token ids."""
    return torch.randint(1, 32, (2, 16))


# ---------------------------------------------------------------------------
# GenomicEmbedding
# ---------------------------------------------------------------------------

class TestGenomicEmbedding:

    @pytest.mark.parametrize("emb_type", ["learned", "sinusoidal", "sinusoidal+learned"])
    def test_output_shape(self, cfg, batch, emb_type):
        emb = GenomicEmbedding(cfg, embedding_type=emb_type)
        out = emb(batch)
        assert out.shape == (2, 16, 64)

    def test_with_segment_ids(self, cfg, batch):
        emb = GenomicEmbedding(cfg, num_segment_types=2)
        seg = torch.zeros_like(batch)
        out = emb(batch, segment_ids=seg)
        assert out.shape == (2, 16, 64)

    def test_sinusoidal_pe_shape(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=128)
        out = pe(16)
        assert out.shape == (1, 16, 64)


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class TestBlocks:

    def test_mhsa_shape(self, cfg):
        mhsa = MultiHeadSelfAttention(cfg)
        x = torch.randn(2, 16, 64)
        out = mhsa(x)
        assert out.shape == x.shape

    def test_mhsa_with_mask(self, cfg):
        mhsa = MultiHeadSelfAttention(cfg)
        x = torch.randn(2, 16, 64)
        mask = torch.ones(2, 16)
        mask[:, -4:] = 0
        out = mhsa(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_feedforward_shape(self, cfg):
        ff = FeedForward(cfg)
        x = torch.randn(2, 16, 64)
        out = ff(x)
        assert out.shape == x.shape

    def test_encoder_layer_shape(self, cfg):
        layer = TransformerEncoderLayer(cfg)
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == x.shape

    def test_prenorm_layer(self, cfg):
        cfg_pre = ModelConfig(
            d_model=64, n_heads=4, n_layers=1, d_ff=128,
            dropout=0.0, attention_dropout=0.0,
            norm_type="prenorm", vocab_size=32,
        )
        layer = TransformerEncoderLayer(cfg_pre)
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# GenovaTransformer
# ---------------------------------------------------------------------------

class TestGenovaTransformer:

    def test_forward_shape(self, cfg, batch):
        model = GenovaTransformer(cfg)
        model.eval()
        out = model(batch)
        assert out["last_hidden_state"].shape == (2, 16, 64)

    def test_hidden_states(self, cfg, batch):
        model = GenovaTransformer(cfg)
        model.eval()
        out = model(batch, output_hidden_states=True)
        # n_layers + 1 (embedding output)
        assert len(out["hidden_states"]) == cfg.n_layers + 1

    def test_attention_mask(self, cfg, batch):
        model = GenovaTransformer(cfg)
        model.eval()
        mask = torch.ones_like(batch, dtype=torch.float)
        mask[:, -4:] = 0
        out = model(batch, attention_mask=mask)
        assert out["last_hidden_state"].shape == (2, 16, 64)


# ---------------------------------------------------------------------------
# GenovaForMLM
# ---------------------------------------------------------------------------

class TestGenovaForMLM:

    def test_forward_logits_shape(self, cfg, batch):
        model = GenovaForMLM(cfg)
        model.eval()
        out = model(batch)
        assert out["logits"].shape == (2, 16, 32)

    def test_forward_with_loss(self, cfg, batch):
        model = GenovaForMLM(cfg)
        model.train()
        labels = batch.clone()
        labels[:, :8] = -100  # ignore first half
        out = model(batch, labels=labels)
        assert "loss" in out
        assert out["loss"].dim() == 0  # scalar
        assert out["loss"].item() > 0

    def test_loss_backprop(self, cfg, batch):
        model = GenovaForMLM(cfg)
        model.train()
        labels = batch.clone()
        out = model(batch, labels=labels)
        out["loss"].backward()
        # Verify at least one gradient was computed
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_tied_weights(self, cfg):
        model = GenovaForMLM(cfg)
        assert model.mlm_head.decoder.weight is model.transformer.embeddings.token_embeddings.weight

    def test_untied_weights(self, cfg):
        cfg_untied = ModelConfig(
            d_model=64, n_heads=4, n_layers=2, d_ff=128,
            vocab_size=32, tie_word_embeddings=False,
            dropout=0.0, attention_dropout=0.0,
        )
        model = GenovaForMLM(cfg_untied)
        assert model.mlm_head.decoder.weight is not model.transformer.embeddings.token_embeddings.weight


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

class TestModelFactory:

    def test_create_mlm(self, cfg):
        model = create_model(cfg, task="mlm")
        assert isinstance(model, GenovaForMLM)

    def test_create_backbone(self, cfg):
        model = create_model(cfg, task="backbone")
        assert isinstance(model, GenovaTransformer)

    def test_unknown_arch_raises(self):
        bad_cfg = ModelConfig(arch="unknown_arch")
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_model(bad_cfg)

    def test_unknown_task_raises(self, cfg):
        with pytest.raises(ValueError, match="Unknown task"):
            create_model(cfg, task="classification")

    def test_pretrained_missing_raises(self, cfg, tmp_path):
        with pytest.raises(FileNotFoundError):
            create_model(cfg, pretrained_path=tmp_path / "nonexistent.pt")


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

class TestParameterCounting:

    def test_count_parameters(self, cfg):
        model = GenovaForMLM(cfg)
        total = count_parameters(model, trainable_only=True)
        assert total > 0
        all_params = count_parameters(model, trainable_only=False)
        assert all_params >= total

    def test_model_summary(self, cfg):
        model = GenovaForMLM(cfg)
        summary = model_summary(model)
        assert summary["total_params"] > 0
        assert summary["trainable_params"] > 0
        assert isinstance(summary["layer_counts"], dict)
        assert summary["non_trainable_params"] == summary["total_params"] - summary["trainable_params"]
