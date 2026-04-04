"""Integration test: contrastive learning pipeline.

Tests augmented pair creation, ContrastiveGenovaModel forward pass,
NTXentLoss, and embedding extraction with synthetic sequences.
"""

from __future__ import annotations

import pytest
import torch

from genova.utils.config import ModelConfig
from genova.contrastive.contrastive_model import (
    ContrastiveGenovaModel,
    NTXentLoss,
    ProjectionHead,
)
from genova.contrastive.augmentations import (
    GenomicAugmenter,
    random_mask,
    random_mutation,
    reverse_complement,
    subsequence_crop,
    window_shuffle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_config():
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


@pytest.fixture
def sample_tokens():
    """Batch of synthetic token id tensors."""
    B, L = 4, 20
    # Use token ids in range [1, 8] (avoid 0 which is pad)
    return torch.randint(1, 8, (B, L))


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAugmentations:
    """Test genomic sequence augmentation functions."""

    def test_reverse_complement(self, sample_tokens):
        rc = reverse_complement(sample_tokens)
        assert rc.shape == sample_tokens.shape, "Shape should be preserved"
        # Applying reverse complement twice should give back original
        rc2 = reverse_complement(rc)
        assert torch.equal(rc2, sample_tokens), (
            "Double reverse complement should return original"
        )

    def test_random_mutation(self, sample_tokens):
        mutated = random_mutation(sample_tokens, rate=0.5)
        assert mutated.shape == sample_tokens.shape
        # With 50% rate, at least some tokens should differ
        # (probabilistic but extremely unlikely to be all identical)
        differs = (mutated != sample_tokens).any()
        assert differs, "With 50% mutation rate, sequences should differ"

    def test_random_mask(self, sample_tokens):
        masked = random_mask(sample_tokens, mask_rate=0.5, mask_token_id=5)
        assert masked.shape == sample_tokens.shape
        n_masked = (masked == 5).sum().item()
        assert n_masked > 0, "Some tokens should be masked"

    def test_subsequence_crop(self, sample_tokens):
        cropped = subsequence_crop(sample_tokens, min_frac=0.5, max_frac=0.8)
        assert cropped.shape == sample_tokens.shape, "Shape should be preserved (with padding)"

    def test_window_shuffle(self, sample_tokens):
        shuffled = window_shuffle(sample_tokens, window_size=4)
        assert shuffled.shape == sample_tokens.shape

    def test_augmenter_produces_two_views(self, sample_tokens):
        augmenter = GenomicAugmenter(
            augmentations=["random_mutation", "random_mask"],
            mutation_rate=0.1,
            mask_rate=0.15,
            mask_token_id=5,
        )
        view1, view2 = augmenter(sample_tokens)
        assert view1.shape == sample_tokens.shape
        assert view2.shape == sample_tokens.shape


# ---------------------------------------------------------------------------
# Contrastive model tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestContrastiveLearning:
    """ContrastiveGenovaModel and NTXentLoss integration tests."""

    def test_ntxent_loss_basic(self):
        """NTXentLoss produces a positive finite scalar."""
        loss_fn = NTXentLoss(temperature=0.07)
        N, D = 8, 32

        z1 = torch.nn.functional.normalize(torch.randn(N, D), dim=-1)
        z2 = torch.nn.functional.normalize(torch.randn(N, D), dim=-1)

        loss = loss_fn(z1, z2)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_ntxent_identical_views_low_loss(self):
        """Identical views should give lower loss than random views."""
        loss_fn = NTXentLoss(temperature=0.5)
        N, D = 8, 32

        z = torch.nn.functional.normalize(torch.randn(N, D), dim=-1)
        loss_identical = loss_fn(z, z).item()

        z_random = torch.nn.functional.normalize(torch.randn(N, D), dim=-1)
        loss_random = loss_fn(z, z_random).item()

        assert loss_identical < loss_random, (
            f"Identical views loss ({loss_identical:.4f}) should be lower than "
            f"random views loss ({loss_random:.4f})"
        )

    def test_contrastive_model_forward(self, tiny_config, sample_tokens):
        """Full forward pass through ContrastiveGenovaModel."""
        model = ContrastiveGenovaModel(
            tiny_config,
            projection_dim=32,
            temperature=0.1,
            pooling="mean",
        )
        model.train()

        view1 = sample_tokens
        view2 = random_mutation(sample_tokens, rate=0.1)

        out = model(view1, view2)

        assert "loss" in out, "Output should contain 'loss'"
        assert "z1" in out, "Output should contain 'z1'"
        assert "z2" in out, "Output should contain 'z2'"
        assert torch.isfinite(out["loss"]), "Loss should be finite"
        assert out["z1"].shape == (sample_tokens.size(0), 32), (
            f"z1 should have shape (B, 32), got {out['z1'].shape}"
        )

    def test_contrastive_model_backward(self, tiny_config, sample_tokens):
        """Backward pass produces gradients."""
        model = ContrastiveGenovaModel(
            tiny_config,
            projection_dim=32,
            temperature=0.1,
        )
        model.train()

        view1 = sample_tokens
        view2 = random_mutation(sample_tokens, rate=0.1)

        out = model(view1, view2)
        out["loss"].backward()

        grad_count = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0, "Should have non-zero gradients after backward"

    def test_embedding_extraction(self, tiny_config, sample_tokens):
        """get_embeddings returns vectors without projection head."""
        model = ContrastiveGenovaModel(
            tiny_config,
            projection_dim=32,
            temperature=0.1,
        )
        model.eval()

        with torch.no_grad():
            embeddings = model.get_embeddings(sample_tokens)

        assert embeddings.shape == (sample_tokens.size(0), tiny_config.d_model), (
            f"Embeddings should have shape (B, d_model), got {embeddings.shape}"
        )

    def test_projection_head_normalizes(self):
        """ProjectionHead output should be L2-normalized."""
        head = ProjectionHead(input_dim=64, hidden_dim=64, output_dim=32)
        head.eval()

        x = torch.randn(4, 64)
        with torch.no_grad():
            z = head(x)

        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            f"Projections should be L2-normalized, got norms: {norms}"
        )
