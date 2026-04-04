"""Integration test: generative pipelines.

Tests autoregressive generation with various sampling strategies and
discrete diffusion forward/reverse passes.
"""

from __future__ import annotations

import pytest
import torch

from genova.utils.config import ModelConfig
from genova.data.tokenizer import GenomicTokenizer
from genova.models.transformer import GenovaForMLM
from genova.generative.autoregressive import AutoregressiveGenerator
from genova.generative.diffusion import DiscreteDiffusion, DiffusionGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tokenizer():
    tok = GenomicTokenizer(mode="kmer", k=3, stride=1)
    tok.build_vocab(["ACGTACGTACGTACGTACGT"])
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
        tie_word_embeddings=False,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=0,
    )


@pytest.fixture
def mlm_model(tiny_config):
    """A tiny GenovaForMLM that returns logits."""
    model = GenovaForMLM(tiny_config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Autoregressive tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAutoregressive:
    """Autoregressive generation tests."""

    def test_greedy_generation(self, mlm_model, tokenizer):
        """Greedy generation (temperature=0) produces valid token ids."""
        generator = AutoregressiveGenerator(
            model=mlm_model,
            tokenizer=tokenizer,
            device="cpu",
        )

        result = generator.generate(
            num_sequences=2,
            max_length=20,
            temperature=0,  # greedy
        )

        assert "token_ids" in result
        assert "sequences" in result
        assert "log_probs" in result
        assert result["token_ids"].shape[0] == 2, "Should generate 2 sequences"
        assert len(result["sequences"]) == 2

        # All token ids should be valid
        max_id = tokenizer.vocab_size
        assert (result["token_ids"] < max_id).all(), (
            "All token ids should be within vocab range"
        )

    def test_topk_generation(self, mlm_model, tokenizer):
        """Top-k sampling generates sequences."""
        generator = AutoregressiveGenerator(
            model=mlm_model,
            tokenizer=tokenizer,
            device="cpu",
        )

        result = generator.generate(
            num_sequences=2,
            max_length=15,
            temperature=0.8,
            top_k=5,
        )

        assert result["token_ids"].shape[0] == 2
        assert all(isinstance(s, str) for s in result["sequences"])

    def test_nucleus_sampling(self, mlm_model, tokenizer):
        """Nucleus (top-p) sampling generates sequences."""
        generator = AutoregressiveGenerator(
            model=mlm_model,
            tokenizer=tokenizer,
            device="cpu",
        )

        result = generator.generate(
            num_sequences=2,
            max_length=15,
            temperature=0.9,
            top_p=0.9,
        )

        assert result["token_ids"].shape[0] == 2

    def test_log_probs_shape(self, mlm_model, tokenizer):
        """Log probs have correct shape."""
        generator = AutoregressiveGenerator(
            model=mlm_model,
            tokenizer=tokenizer,
            device="cpu",
        )

        result = generator.generate(
            num_sequences=1,
            max_length=10,
            temperature=1.0,
            stop_on_eos=False,
        )

        B = result["token_ids"].shape[0]
        gen_len = result["log_probs"].shape[1]
        assert gen_len > 0, "Should have generated some tokens"
        assert result["log_probs"].shape[0] == B

    def test_prompt_continuation(self, mlm_model, tokenizer):
        """Generation can continue from a prompt."""
        generator = AutoregressiveGenerator(
            model=mlm_model,
            tokenizer=tokenizer,
            device="cpu",
        )

        prompt = torch.tensor([[5, 6, 7]], dtype=torch.long)
        result = generator.generate(
            num_sequences=1,
            max_length=15,
            temperature=0.8,
            prompt_ids=prompt,
            stop_on_eos=False,
        )

        # Result should start with the prompt
        generated = result["token_ids"][0]
        assert generated[0] == 5, "Should start with prompt token"
        assert generated[1] == 6
        assert generated[2] == 7


# ---------------------------------------------------------------------------
# Diffusion tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDiffusion:
    """Discrete diffusion forward and reverse tests."""

    @pytest.fixture
    def diffusion_config(self):
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
            vocab_size=32,
            pad_token_id=0,
        )

    def test_forward_loss(self, diffusion_config):
        """Diffusion training forward produces finite loss."""
        model = DiscreteDiffusion(
            diffusion_config,
            num_timesteps=10,
            num_classes=5,
            schedule="cosine",
        )
        model.train()

        B, L = 2, 16
        x_0 = torch.randint(0, 5, (B, L))

        out = model(x_0)

        assert "loss" in out, "Should return loss"
        assert "logits" in out, "Should return logits"
        assert torch.isfinite(out["loss"]), f"Loss should be finite, got {out['loss'].item()}"
        assert out["logits"].shape == (B, L, 5)

    def test_forward_backward(self, diffusion_config):
        """Diffusion training backward produces gradients."""
        model = DiscreteDiffusion(
            diffusion_config,
            num_timesteps=10,
            num_classes=5,
            schedule="linear",
        )
        model.train()

        x_0 = torch.randint(0, 5, (2, 16))
        out = model(x_0)
        out["loss"].backward()

        grad_count = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0, "Should have gradients after backward"

    def test_q_sample_noisy(self, diffusion_config):
        """Forward diffusion q_sample produces noisy versions."""
        model = DiscreteDiffusion(
            diffusion_config,
            num_timesteps=10,
            num_classes=5,
        )

        x_0 = torch.zeros(2, 16, dtype=torch.long)
        t = torch.tensor([9, 9])  # high noise
        x_t = model.q_sample(x_0, t)

        assert x_t.shape == x_0.shape
        assert x_t.dtype == torch.long
        # At high noise, tokens should be spread across classes
        unique_tokens = x_t.unique()
        assert len(unique_tokens) > 1, (
            "At high noise timestep, output should have multiple token types"
        )

    def test_reverse_generation(self, diffusion_config):
        """Reverse diffusion generates valid sequences."""
        model = DiscreteDiffusion(
            diffusion_config,
            num_timesteps=5,  # very few steps for speed
            num_classes=5,
        )
        model.eval()

        gen = DiffusionGenerator(model, num_classes=5, device="cpu")

        result = gen.generate(
            num_sequences=2,
            seq_length=16,
            temperature=1.0,
        )

        assert "token_ids" in result
        assert "sequences" in result
        assert result["token_ids"].shape == (2, 16)
        assert len(result["sequences"]) == 2

        # All tokens should be valid class indices
        assert (result["token_ids"] >= 0).all()
        assert (result["token_ids"] < 5).all()

        # Sequences should be DNA strings
        valid_chars = set("ACGTN")
        for seq in result["sequences"]:
            assert set(seq).issubset(valid_chars), (
                f"Sequence contains invalid characters: {set(seq) - valid_chars}"
            )

    def test_kl_divergence_loss_type(self, diffusion_config):
        """KL divergence loss type works."""
        model = DiscreteDiffusion(
            diffusion_config,
            num_timesteps=5,
            num_classes=5,
            loss_type="kl_divergence",
        )
        model.train()

        x_0 = torch.randint(0, 5, (2, 16))
        out = model(x_0)

        assert torch.isfinite(out["loss"]), "KL loss should be finite"
