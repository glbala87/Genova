"""Tests for Batch D architecture enhancements in Genova.

Covers GQA/MQA, SwiGLU, ALiBi, sliding window attention, RMSNorm,
EMA, distillation, pruning, and advanced LR schedulers.

All tests are CPU-only and require no external data.
"""

import copy
import math

import pytest
import torch
import torch.nn as nn

from genova.utils.config import ModelConfig, TrainingConfig
from genova.models.transformer import (
    GenovaForMLM,
    GenovaTransformer,
    MultiHeadSelfAttention,
    FeedForward,
    RMSNorm,
    SwiGLU,
    _build_norm,
    _build_sliding_window_mask,
)
from genova.models.embeddings import ALiBiPositionalBias, RotaryPositionalEmbedding
from genova.models.pruning import (
    compute_head_importance,
    prune_heads,
    prune_ffn,
    PruningSchedule,
)
from genova.training.ema import EMAModel
from genova.training.distillation import (
    DistillationLoss,
    FeatureDistillationLoss,
    DistillationTrainer,
)
from genova.training.scheduler import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    PolynomialDecay,
    create_scheduler,
)
from genova.models.model_factory import create_model, count_parameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_config(**overrides) -> ModelConfig:
    """Return a minimal ModelConfig for testing."""
    defaults = dict(
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        vocab_size=128,
        max_position_embeddings=256,
        dropout=0.0,
        attention_dropout=0.0,
        norm_type="layernorm",
        pad_token_id=0,
        gradient_checkpointing=False,
        tie_word_embeddings=False,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _dummy_ids(batch=2, seq_len=16):
    """Return random input_ids and labels."""
    ids = torch.randint(1, 128, (batch, seq_len))
    labels = ids.clone()
    labels[:, ::3] = -100  # mask some positions
    return ids, labels


# ===================================================================
# 1. GQA / MQA Tests
# ===================================================================


class TestGQA_MQA:
    """Grouped-Query Attention and Multi-Query Attention tests."""

    @pytest.mark.parametrize("n_kv_heads", [4, 2, 1])
    def test_gqa_forward_shape(self, n_kv_heads):
        """Forward pass output shape is (B, L, D) for various n_kv_heads."""
        cfg = _small_config(n_heads=4, n_kv_heads=n_kv_heads)
        attn = MultiHeadSelfAttention(cfg)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_mha_standard(self):
        """Standard MHA: n_kv_heads == n_heads produces correct output."""
        cfg = _small_config(n_heads=4, n_kv_heads=4)
        attn = MultiHeadSelfAttention(cfg)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_gqa_two_kv_heads(self):
        """GQA with n_kv_heads=2 and n_heads=8 works."""
        cfg = _small_config(d_model=64, n_heads=8, n_kv_heads=2, d_ff=128)
        # head_dim = 64 / 8 = 8
        attn = MultiHeadSelfAttention(cfg)
        assert attn.n_kv_heads == 2
        assert attn.n_groups == 4
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_mqa_single_kv_head(self):
        """MQA with n_kv_heads=1 works."""
        cfg = _small_config(n_heads=4, n_kv_heads=1)
        attn = MultiHeadSelfAttention(cfg)
        assert attn.n_kv_heads == 1
        assert attn.n_groups == 4
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_gqa_fewer_params_than_mha(self):
        """GQA model has fewer parameters than equivalent MHA model."""
        cfg_mha = _small_config(n_heads=4, n_kv_heads=4)
        cfg_gqa = _small_config(n_heads=4, n_kv_heads=1)
        model_mha = GenovaForMLM(cfg_mha)
        model_gqa = GenovaForMLM(cfg_gqa)
        params_mha = count_parameters(model_mha, trainable_only=False)
        params_gqa = count_parameters(model_gqa, trainable_only=False)
        assert params_gqa < params_mha

    def test_gqa_gradient_flow(self):
        """Gradients flow through GQA attention."""
        cfg = _small_config(n_heads=4, n_kv_heads=2)
        attn = MultiHeadSelfAttention(cfg)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_nkv_heads_none_defaults_to_n_heads(self):
        """When n_kv_heads is None, it defaults to n_heads (standard MHA)."""
        cfg = _small_config(n_heads=4, n_kv_heads=None)
        attn = MultiHeadSelfAttention(cfg)
        assert attn.n_kv_heads == 4
        assert attn.n_groups == 1

    def test_gqa_full_model_forward(self):
        """Full model with GQA produces valid logits."""
        cfg = _small_config(n_heads=4, n_kv_heads=2)
        model = GenovaForMLM(cfg)
        ids, labels = _dummy_ids()
        out = model(ids, labels=labels)
        assert out["logits"].shape == (2, 16, 128)
        assert torch.isfinite(out["loss"])


# ===================================================================
# 2. SwiGLU Tests
# ===================================================================


class TestSwiGLU:
    """SwiGLU gated feed-forward network tests."""

    def test_swiglu_forward_shape(self):
        """SwiGLU forward pass produces correct output shape."""
        swiglu = SwiGLU(d_model=64, d_ff=128)
        x = torch.randn(2, 16, 64)
        out = swiglu(x)
        assert out.shape == (2, 16, 64)

    def test_swiglu_has_three_linear_layers(self):
        """SwiGLU has gate, up, and down linear projections."""
        swiglu = SwiGLU(d_model=64, d_ff=128)
        assert hasattr(swiglu, "w_gate")
        assert hasattr(swiglu, "w_up")
        assert hasattr(swiglu, "w_down")
        assert isinstance(swiglu.w_gate, nn.Linear)
        assert isinstance(swiglu.w_up, nn.Linear)
        assert isinstance(swiglu.w_down, nn.Linear)

    def test_model_with_swiglu_activation(self):
        """Model with activation='swiglu' creates correctly and runs."""
        cfg = _small_config(activation="swiglu")
        model = GenovaForMLM(cfg)
        ids, labels = _dummy_ids()
        out = model(ids, labels=labels)
        assert out["logits"].shape == (2, 16, 128)
        assert torch.isfinite(out["loss"])

    def test_default_gelu_activation(self):
        """Default activation='gelu' still works."""
        cfg = _small_config(activation="gelu")
        model = GenovaForMLM(cfg)
        ids, labels = _dummy_ids()
        out = model(ids, labels=labels)
        assert torch.isfinite(out["loss"])

    def test_swiglu_gradient_flow(self):
        """Gradients flow through SwiGLU."""
        swiglu = SwiGLU(d_model=64, d_ff=128)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = swiglu(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ===================================================================
# 3. ALiBi Tests
# ===================================================================


class TestALiBi:
    """Attention with Linear Biases tests."""

    def test_alibi_bias_shape(self):
        """ALiBi bias shape is (1, num_heads, seq_len, seq_len)."""
        alibi = ALiBiPositionalBias(n_heads=4)
        bias = alibi(32)
        assert bias.shape == (1, 4, 32, 32)

    def test_alibi_bias_symmetric_distances(self):
        """Bias uses absolute distances: bias[h, i, j] == bias[h, j, i]."""
        alibi = ALiBiPositionalBias(n_heads=4)
        bias = alibi(16)
        # Since bias = -slope * |i - j|, it should be symmetric per head
        for h in range(4):
            head_bias = bias[0, h]
            assert torch.allclose(head_bias, head_bias.T)

    def test_alibi_slopes_geometric_series(self):
        """Slopes follow geometric series 2^(-8*(i+1)/n_heads)."""
        n_heads = 4
        alibi = ALiBiPositionalBias(n_heads=n_heads)
        expected_slopes = torch.tensor(
            [2.0 ** (-8.0 * (i + 1) / n_heads) for i in range(n_heads)]
        )
        assert torch.allclose(alibi.slopes, expected_slopes)

    def test_alibi_no_learnable_parameters(self):
        """ALiBi has no learnable parameters."""
        alibi = ALiBiPositionalBias(n_heads=4)
        params = list(alibi.parameters())
        assert len(params) == 0

    def test_model_with_alibi(self):
        """Model with pos_encoding='alibi' creates and runs."""
        cfg = _small_config()
        model = GenovaForMLM(cfg, embedding_type="alibi")
        ids, labels = _dummy_ids()
        out = model(ids, labels=labels)
        assert out["logits"].shape == (2, 16, 128)
        assert torch.isfinite(out["loss"])


# ===================================================================
# 4. Sliding Window Tests
# ===================================================================


class TestSlidingWindow:
    """Sliding window attention tests."""

    def test_no_sliding_window_default(self):
        """With window_size=None, no sliding window mask is applied."""
        cfg = _small_config(sliding_window_size=None)
        attn = MultiHeadSelfAttention(cfg)
        assert attn.window_size is None

    def test_sliding_window_mask_shape(self):
        """Sliding window mask has shape (1, 1, L, L)."""
        mask = _build_sliding_window_mask(32, window_size=8, device=torch.device("cpu"))
        assert mask.shape == (1, 1, 32, 32)

    def test_sliding_window_blocks_distant_tokens(self):
        """With window_size=4, tokens more than 2 positions away are masked."""
        mask = _build_sliding_window_mask(16, window_size=4, device=torch.device("cpu"))
        # Position 0 attending to position 10 should be -inf
        assert mask[0, 0, 0, 10] == float("-inf")
        # Position 0 attending to position 1 should be 0 (allowed)
        assert mask[0, 0, 0, 1] == 0.0

    def test_sliding_window_forward_pass(self):
        """Forward pass works with sliding window attention."""
        cfg = _small_config(sliding_window_size=8)
        model = GenovaForMLM(cfg)
        ids, labels = _dummy_ids(batch=2, seq_len=32)
        out = model(ids, labels=labels)
        assert out["logits"].shape == (2, 32, 128)
        assert torch.isfinite(out["loss"])


# ===================================================================
# 5. RMSNorm Tests
# ===================================================================


class TestRMSNorm:
    """Root Mean Square Layer Normalization tests."""

    def test_rmsnorm_output_shape(self):
        """RMSNorm output shape matches input shape."""
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_rmsnorm_no_bias(self):
        """RMSNorm has weight but no bias parameter (unlike LayerNorm)."""
        norm = RMSNorm(64)
        assert hasattr(norm, "weight")
        # RMSNorm should not have a bias attribute as a Parameter
        param_names = [name for name, _ in norm.named_parameters()]
        assert "weight" in param_names
        assert "bias" not in param_names

    def test_norm_type_rmsnorm_in_model(self):
        """norm_type='rmsnorm' creates RMSNorm in the model."""
        cfg = _small_config(norm_type="rmsnorm")
        model = GenovaForMLM(cfg)
        # Check that at least one RMSNorm module exists
        has_rmsnorm = any(isinstance(m, RMSNorm) for m in model.modules())
        assert has_rmsnorm

    def test_norm_type_layernorm_default(self):
        """norm_type='layernorm' (default) uses nn.LayerNorm."""
        cfg = _small_config(norm_type="layernorm")
        model = GenovaForMLM(cfg)
        has_layernorm = any(
            isinstance(m, nn.LayerNorm) for m in model.modules()
        )
        assert has_layernorm
        has_rmsnorm = any(isinstance(m, RMSNorm) for m in model.modules())
        assert not has_rmsnorm


# ===================================================================
# 6. EMA Tests
# ===================================================================


class TestEMA:
    """Exponential Moving Average tests."""

    def _make_model_and_ema(self, decay=0.999):
        cfg = _small_config()
        model = GenovaForMLM(cfg)
        ema = EMAModel(model, decay=decay)
        return model, ema

    def test_ema_shadow_differs_after_update(self):
        """After a training step + EMA update, shadow differs from model."""
        model, ema = self._make_model_and_ema()
        # Simulate a parameter change
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ema.update()
        # Shadow should now differ from the current model params
        differs = False
        for name, param in model.named_parameters():
            if name in ema.shadow and param.requires_grad:
                if not torch.allclose(param.data, ema.shadow[name]):
                    differs = True
                    break
        assert differs

    def test_apply_shadow_swaps_weights(self):
        """apply_shadow replaces model weights with shadow weights."""
        model, ema = self._make_model_and_ema()
        # Change model weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update()

        # Record a shadow value
        first_name = next(iter(ema.shadow.keys()))
        shadow_val = ema.shadow[first_name].clone()

        ema.apply_shadow()
        # Model param should now match shadow
        param_dict = dict(model.named_parameters())
        assert torch.allclose(param_dict[first_name].data, shadow_val)

    def test_restore_brings_back_original(self):
        """restore() brings back original training weights."""
        model, ema = self._make_model_and_ema()
        first_name = next(
            n for n, p in model.named_parameters() if p.requires_grad
        )
        original = dict(model.named_parameters())[first_name].data.clone()

        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update()

        modified = dict(model.named_parameters())[first_name].data.clone()
        ema.apply_shadow()
        ema.restore()
        restored = dict(model.named_parameters())[first_name].data.clone()
        assert torch.allclose(restored, modified)

    def test_state_dict_roundtrip(self):
        """state_dict save/load roundtrip preserves shadow weights."""
        model, ema = self._make_model_and_ema()
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.5)
        ema.update()

        state = ema.state_dict()
        ema2 = EMAModel(model, decay=0.5)  # different decay
        ema2.load_state_dict(state)

        assert ema2.decay == ema.decay
        for name in ema.shadow:
            if name in ema2.shadow:
                assert torch.allclose(ema.shadow[name], ema2.shadow[name])

    def test_decay_affects_update_rate(self):
        """Higher decay means shadow moves slower toward current weights."""
        cfg = _small_config()
        model_fast = GenovaForMLM(cfg)
        model_slow = copy.deepcopy(model_fast)

        ema_fast = EMAModel(model_fast, decay=0.5)
        ema_slow = EMAModel(model_slow, decay=0.999)

        with torch.no_grad():
            for p in model_fast.parameters():
                p.add_(1.0)
            for p in model_slow.parameters():
                p.add_(1.0)

        ema_fast.update()
        ema_slow.update()

        # Fast EMA should have moved more toward the new weights
        first_name = next(iter(ema_fast.shadow.keys()))
        fast_shadow = ema_fast.shadow[first_name]
        slow_shadow = ema_slow.shadow[first_name]
        current = dict(model_fast.named_parameters())[first_name].data

        # Distance from current weights: fast should be closer
        dist_fast = (current - fast_shadow).abs().sum()
        dist_slow = (current - slow_shadow).abs().sum()
        assert dist_fast < dist_slow


# ===================================================================
# 7. Distillation Tests
# ===================================================================


class TestDistillation:
    """Knowledge distillation tests."""

    def test_distillation_loss_finite(self):
        """DistillationLoss produces a finite loss value."""
        criterion = DistillationLoss(temperature=4.0, alpha=0.5)
        student_logits = torch.randn(2, 16, 128)
        teacher_logits = torch.randn(2, 16, 128)
        labels = torch.randint(0, 128, (2, 16))
        result = criterion(student_logits, teacher_logits, labels=labels)
        assert torch.isfinite(result["loss"])
        assert "distill_loss" in result
        assert "task_loss" in result

    def test_temperature_scaling(self):
        """Different temperatures produce different soft targets."""
        student_logits = torch.randn(2, 16, 128)
        teacher_logits = torch.randn(2, 16, 128)

        crit_low = DistillationLoss(temperature=1.0, alpha=1.0)
        crit_high = DistillationLoss(temperature=10.0, alpha=1.0)

        loss_low = crit_low(student_logits, teacher_logits)["loss"]
        loss_high = crit_high(student_logits, teacher_logits)["loss"]
        # Different temperatures should give different loss values
        assert not torch.allclose(loss_low, loss_high)

    def test_alpha_zero_pure_task_loss(self):
        """alpha=0 means the combined loss equals the task loss."""
        criterion = DistillationLoss(temperature=4.0, alpha=0.0)
        student_logits = torch.randn(2, 16, 128)
        teacher_logits = torch.randn(2, 16, 128)
        labels = torch.randint(0, 128, (2, 16))
        result = criterion(student_logits, teacher_logits, labels=labels)
        # With alpha=0: loss = 0 * distill + 1.0 * task = task_loss
        assert torch.allclose(result["loss"], result["task_loss"])

    def test_feature_distillation_mse(self):
        """FeatureDistillationLoss computes MSE between hidden states."""
        feat_loss = FeatureDistillationLoss(student_dim=64, teacher_dim=64)
        student_feats = torch.randn(2, 16, 64)
        teacher_feats = torch.randn(2, 16, 64)
        loss = feat_loss(student_feats, teacher_feats)
        assert torch.isfinite(loss)
        assert loss.ndim == 0  # scalar

    def test_distillation_trainer_one_step(self):
        """DistillationTrainer completes one train_step without error."""
        cfg_teacher = _small_config(d_model=64, n_heads=4, n_layers=2, d_ff=128)
        cfg_student = _small_config(d_model=64, n_heads=4, n_layers=1, d_ff=128)

        teacher = GenovaForMLM(cfg_teacher)
        student = GenovaForMLM(cfg_student)

        criterion = DistillationLoss(temperature=4.0, alpha=0.5)
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

        trainer = DistillationTrainer(
            teacher=teacher,
            student=student,
            criterion=criterion,
            optimizer=optimizer,
        )

        ids, labels = _dummy_ids()
        result = trainer.train_step(ids, labels=labels)
        assert "loss" in result
        assert torch.isfinite(result["loss"])


# ===================================================================
# 8. Pruning Tests
# ===================================================================


class TestPruning:
    """Structured pruning tests."""

    def test_compute_head_importance_shape(self):
        """compute_head_importance returns (n_layers, n_heads) scores."""
        cfg = _small_config(n_heads=4, n_layers=2)
        model = GenovaForMLM(cfg)
        ids, labels = _dummy_ids()
        data = [{"input_ids": ids, "labels": labels}]
        importance = compute_head_importance(model, data, method="taylor")
        assert importance.shape == (2, 4)

    def test_prune_heads_zeros_out(self):
        """prune_heads zeros out selected heads' Q projections."""
        cfg = _small_config(n_heads=4, n_layers=2)
        model = GenovaForMLM(cfg)
        heads_map = {0: [0, 1]}  # prune heads 0,1 from layer 0
        prune_heads(model, num_heads_to_prune=2, heads_to_prune=heads_map)

        # Check that head 0 of layer 0 Q projection is zeroed
        attn = model.transformer.layers[0].attn
        head_dim = attn.head_dim
        q_weight_head0 = attn.q_proj.weight[0:head_dim, :]
        assert q_weight_head0.abs().sum() == 0.0

    def test_prune_ffn_reduces_neurons(self):
        """prune_ffn zeros out neurons and returns nonzero count."""
        cfg = _small_config()
        model = GenovaForMLM(cfg)
        total_pruned = prune_ffn(model, pruning_ratio=0.3)
        assert total_pruned > 0

    def test_pruning_schedule_sparsity(self):
        """PruningSchedule returns correct sparsity at different steps."""
        schedule = PruningSchedule(
            initial_ratio=0.0, final_ratio=0.5, total_steps=1000
        )
        # At step 0: should be close to initial
        r0 = schedule.get_ratio(0)
        assert r0 == pytest.approx(0.0, abs=0.01)
        # At step 1000: should equal final
        r_end = schedule.get_ratio(1000)
        assert r_end == pytest.approx(0.5, abs=0.01)
        # Intermediate should be between initial and final
        r_mid = schedule.get_ratio(500)
        assert 0.0 < r_mid < 0.5

    def test_pruned_model_still_works(self):
        """A pruned model still produces valid finite outputs."""
        cfg = _small_config()
        model = GenovaForMLM(cfg)
        prune_heads(model, num_heads_to_prune=2, heads_to_prune={0: [0], 1: [1]})
        prune_ffn(model, pruning_ratio=0.2)
        ids, labels = _dummy_ids()
        out = model(ids, labels=labels)
        assert torch.isfinite(out["logits"]).all()
        assert torch.isfinite(out["loss"])


# ===================================================================
# 9. Advanced Scheduler Tests
# ===================================================================


class TestAdvancedSchedulers:
    """Tests for new LR scheduler types."""

    def _make_optimizer(self, lr=0.1):
        model = nn.Linear(10, 10)
        return torch.optim.SGD(model.parameters(), lr=lr)

    def test_cosine_warm_restarts_lr_resets(self):
        """CosineAnnealingWarmRestarts: LR resets after T_0 steps."""
        opt = self._make_optimizer(lr=0.1)
        scheduler = CosineAnnealingWarmRestarts(
            opt, warmup_steps=0, total_steps=200, t_0=50
        )
        lrs = []
        for _ in range(120):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            scheduler.step()

        # After warmup=0 and t_0=50, LR should reset around step 50.
        # LR at step 1 should be high, at step ~25 mid, at step 49 low,
        # and at step 51 high again.
        lr_at_1 = lrs[1]
        lr_at_49 = lrs[49]
        lr_at_51 = lrs[51]
        # After reset, LR should be higher than trough
        assert lr_at_51 > lr_at_49

    def test_one_cycle_lr_increases_then_decreases(self):
        """OneCycleLR: LR increases in first half, decreases in second."""
        opt = self._make_optimizer(lr=0.1)
        scheduler = OneCycleLR(
            opt, warmup_steps=0, total_steps=100
        )
        lrs = []
        for _ in range(100):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            scheduler.step()

        # LR at midpoint should be higher than at start and end
        lr_start = lrs[0]
        lr_mid = lrs[50]
        lr_end = lrs[99]
        assert lr_mid > lr_start
        assert lr_mid > lr_end

    def test_polynomial_decay(self):
        """PolynomialDecay: LR decreases over training."""
        opt = self._make_optimizer(lr=0.1)
        scheduler = PolynomialDecay(
            opt, warmup_steps=5, total_steps=100, power=2.0
        )
        lrs = []
        for _ in range(100):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            scheduler.step()

        # After warmup, LR should generally decrease
        lr_after_warmup = lrs[10]
        lr_near_end = lrs[90]
        assert lr_after_warmup > lr_near_end

    def test_factory_creates_all_new_types(self):
        """create_scheduler factory creates all new scheduler types."""
        for sched_name in ["cosine_warm_restarts", "one_cycle", "polynomial"]:
            cfg = TrainingConfig(
                lr=0.1, min_lr=0.001, warmup_steps=10, lr_scheduler=sched_name
            )
            opt = self._make_optimizer(lr=0.1)
            scheduler = create_scheduler(opt, cfg, num_training_steps=100)
            assert scheduler is not None

    def test_all_schedulers_finite_lr(self):
        """All new schedulers produce finite LR values over their range."""
        for sched_name in ["cosine_warm_restarts", "one_cycle", "polynomial"]:
            cfg = TrainingConfig(
                lr=0.1, min_lr=0.001, warmup_steps=10, lr_scheduler=sched_name
            )
            opt = self._make_optimizer(lr=0.1)
            scheduler = create_scheduler(opt, cfg, num_training_steps=100)
            for _ in range(100):
                lr = opt.param_groups[0]["lr"]
                assert math.isfinite(lr), f"{sched_name} produced non-finite LR: {lr}"
                opt.step()
                scheduler.step()
