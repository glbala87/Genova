"""Integration test: all model architecture variants.

Tests forward/backward passes, gradient flow, and model factory for
GenovaTransformer, GenovaMamba, and GenovaMultiTask.
"""

from __future__ import annotations

import math

import pytest
import torch

from genova.utils.config import ModelConfig
from genova.models.transformer import GenovaTransformer, GenovaForMLM
from genova.models.mamba_model import GenovaMamba, GenovaMambaForMLM
from genova.models.multi_task import GenovaMultiTask
from genova.models.model_factory import create_model, count_parameters, model_summary


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_config():
    return ModelConfig(
        arch="transformer",
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=128,
        rotary_emb=False,
        flash_attention=False,
        gradient_checkpointing=False,
        tie_word_embeddings=False,
        vocab_size=32,
        pad_token_id=0,
    )


@pytest.fixture
def tiny_mamba_config(tiny_config):
    return ModelConfig(
        arch="mamba",
        d_model=tiny_config.d_model,
        n_heads=tiny_config.n_heads,
        n_layers=tiny_config.n_layers,
        d_ff=tiny_config.d_ff,
        dropout=tiny_config.dropout,
        attention_dropout=tiny_config.attention_dropout,
        max_position_embeddings=tiny_config.max_position_embeddings,
        rotary_emb=False,
        flash_attention=False,
        gradient_checkpointing=False,
        tie_word_embeddings=False,
        vocab_size=tiny_config.vocab_size,
        pad_token_id=0,
    )


@pytest.fixture
def sample_input():
    B, L = 2, 16
    input_ids = torch.randint(5, 30, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    labels = torch.randint(5, 30, (B, L))
    # Set some positions to -100 (ignored)
    labels[:, :3] = -100
    return input_ids, attention_mask, labels


# ---------------------------------------------------------------------------
# Transformer tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGenovaTransformer:
    """GenovaTransformer forward and backward tests."""

    def test_forward_produces_correct_shapes(self, tiny_config, sample_input):
        model = GenovaTransformer(tiny_config)
        input_ids, attention_mask, _ = sample_input

        out = model(input_ids, attention_mask=attention_mask)

        assert "last_hidden_state" in out
        B, L = input_ids.shape
        assert out["last_hidden_state"].shape == (B, L, tiny_config.d_model), (
            f"Expected ({B}, {L}, {tiny_config.d_model}), "
            f"got {out['last_hidden_state'].shape}"
        )

    def test_forward_with_hidden_states(self, tiny_config, sample_input):
        model = GenovaTransformer(tiny_config)
        input_ids, attention_mask, _ = sample_input

        out = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        assert "hidden_states" in out, "Should return hidden_states when requested"
        # n_layers + 1 (embedding output)
        assert len(out["hidden_states"]) == tiny_config.n_layers + 1

    def test_backward_gradient_flow(self, tiny_config, sample_input):
        model = GenovaForMLM(tiny_config)
        model.train()
        input_ids, attention_mask, labels = sample_input

        out = model(input_ids, attention_mask=attention_mask, labels=labels)
        out["loss"].backward()

        # Every parameter with requires_grad should have a gradient
        params_with_grad = [
            (n, p) for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert len(params_with_grad) > 0, "At least some parameters should have gradients"

        for name, param in params_with_grad:
            assert torch.isfinite(param.grad).all(), (
                f"Gradient for {name} contains non-finite values"
            )


# ---------------------------------------------------------------------------
# Mamba tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGenovaMamba:
    """GenovaMamba forward and backward tests (pure PyTorch fallback)."""

    def test_forward_produces_correct_shapes(self, tiny_mamba_config, sample_input):
        model = GenovaMamba(tiny_mamba_config)
        input_ids, attention_mask, _ = sample_input

        out = model(input_ids, attention_mask=attention_mask)

        assert "last_hidden_state" in out
        B, L = input_ids.shape
        assert out["last_hidden_state"].shape == (B, L, tiny_mamba_config.d_model)

    def test_mlm_forward_backward(self, tiny_mamba_config, sample_input):
        model = GenovaMambaForMLM(tiny_mamba_config)
        model.train()
        input_ids, attention_mask, labels = sample_input

        out = model(input_ids, attention_mask=attention_mask, labels=labels)

        assert "loss" in out, "Mamba MLM should return loss"
        assert "logits" in out, "Mamba MLM should return logits"
        assert torch.isfinite(out["loss"]), "Loss should be finite"

        out["loss"].backward()

        params_with_grad = [
            (n, p) for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert len(params_with_grad) > 0, "Should have gradients"

    def test_hidden_states_output(self, tiny_mamba_config, sample_input):
        model = GenovaMamba(tiny_mamba_config)
        input_ids, attention_mask, _ = sample_input

        out = model(input_ids, output_hidden_states=True)

        assert "hidden_states" in out
        assert len(out["hidden_states"]) == tiny_mamba_config.n_layers + 1


# ---------------------------------------------------------------------------
# Multi-task tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGenovaMultiTask:
    """GenovaMultiTask forward with multiple active tasks."""

    def test_multi_task_forward_with_all_tasks(self, tiny_config, sample_input):
        task_configs = {
            "mlm": {"enabled": True, "weight": 1.0},
            "gene_expression": {
                "enabled": True,
                "weight": 0.5,
                "num_targets": 10,
                "pool": "cls",
            },
            "methylation": {
                "enabled": True,
                "weight": 0.5,
                "num_targets": 5,
                "pool": "cls",
            },
        }

        model = GenovaMultiTask(
            tiny_config,
            task_configs=task_configs,
            backbone="transformer",
        )
        input_ids, attention_mask, _ = sample_input
        B, L = input_ids.shape

        labels = {
            "mlm": torch.randint(5, 30, (B, L)),
            "gene_expression": torch.randn(B, 10),
            "methylation": torch.rand(B, 5),
        }
        labels["mlm"][:, :3] = -100

        out = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        assert "total_loss" in out, "Should have total_loss"
        assert "mlm_logits" in out, "Should have mlm_logits"
        assert "gene_expression_logits" in out, "Should have gene_expression logits"
        assert "methylation_logits" in out, "Should have methylation logits"
        assert torch.isfinite(out["total_loss"]), "Total loss should be finite"

        # Check output shapes
        assert out["mlm_logits"].shape == (B, L, tiny_config.vocab_size)
        assert out["gene_expression_logits"].shape == (B, 10)
        assert out["methylation_logits"].shape == (B, 5)

    def test_multi_task_gradient_flow(self, tiny_config, sample_input):
        task_configs = {
            "mlm": {"enabled": True, "weight": 1.0},
            "gene_expression": {"enabled": True, "weight": 0.5, "num_targets": 5},
        }

        model = GenovaMultiTask(
            tiny_config, task_configs=task_configs, backbone="transformer"
        )
        model.train()
        input_ids, attention_mask, _ = sample_input
        B, L = input_ids.shape

        labels = {
            "mlm": torch.randint(5, 30, (B, L)),
            "gene_expression": torch.randn(B, 5),
        }
        labels["mlm"][:, :3] = -100

        out = model(input_ids, attention_mask=attention_mask, labels=labels)
        out["total_loss"].backward()

        # Check gradient flow through backbone and task heads
        backbone_grads = sum(
            1 for n, p in model.backbone.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert backbone_grads > 0, "Backbone should receive gradients from multi-task loss"


# ---------------------------------------------------------------------------
# Model factory tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestModelFactory:
    """Model factory creates correct architecture."""

    def test_create_transformer_mlm(self, tiny_config):
        model = create_model(tiny_config, task="mlm")
        assert isinstance(model, GenovaForMLM), (
            f"Expected GenovaForMLM, got {type(model).__name__}"
        )

    def test_create_transformer_backbone(self, tiny_config):
        model = create_model(tiny_config, task="backbone")
        assert isinstance(model, GenovaTransformer)

    def test_create_mamba_mlm(self, tiny_mamba_config):
        model = create_model(tiny_mamba_config, task="mlm")
        assert isinstance(model, GenovaMambaForMLM)

    def test_create_mamba_backbone(self, tiny_mamba_config):
        model = create_model(tiny_mamba_config, task="backbone")
        assert isinstance(model, GenovaMamba)

    def test_unknown_arch_raises(self, tiny_config):
        bad_config = ModelConfig(arch="unknown_arch", vocab_size=32)
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_model(bad_config)

    def test_count_parameters(self, tiny_config):
        model = create_model(tiny_config, task="mlm")
        total = count_parameters(model, trainable_only=True)
        assert total > 0, "Model should have trainable parameters"

    def test_model_summary(self, tiny_config):
        model = create_model(tiny_config, task="mlm")
        summary = model_summary(model)
        assert summary["total_params"] > 0
        assert summary["trainable_params"] > 0
        assert isinstance(summary["layer_counts"], dict)

    def test_pretrained_path_not_found(self, tiny_config):
        with pytest.raises(FileNotFoundError):
            create_model(tiny_config, pretrained_path="/nonexistent/path.pt")
