"""Tests for LR schedulers."""

from __future__ import annotations

import pytest
import torch
from torch.optim import AdamW

from genova.utils.config import TrainingConfig
from genova.training.scheduler import (
    CosineWithWarmup,
    LinearWithWarmup,
    create_scheduler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_optimizer():
    """A simple optimizer with a single parameter."""
    param = torch.nn.Parameter(torch.randn(4))
    return AdamW([param], lr=1e-3)


def collect_lr_factors(scheduler, steps):
    """Step the scheduler and collect LR values."""
    lrs = []
    for _ in range(steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    return lrs


# ---------------------------------------------------------------------------
# CosineWithWarmup
# ---------------------------------------------------------------------------

class TestCosineWithWarmup:

    def test_warmup_ramp(self, dummy_optimizer):
        sched = CosineWithWarmup(dummy_optimizer, warmup_steps=10, total_steps=100)
        lrs = collect_lr_factors(sched, 11)
        # LR should increase during warmup
        for i in range(1, 10):
            assert lrs[i] > lrs[i - 1] or i == 0

    def test_starts_at_zero(self, dummy_optimizer):
        sched = CosineWithWarmup(dummy_optimizer, warmup_steps=10, total_steps=100)
        # Step 0 -> factor 0/10 = 0
        assert sched.get_last_lr()[0] == pytest.approx(0.0)

    def test_peak_at_warmup_end(self, dummy_optimizer):
        base_lr = 1e-3
        sched = CosineWithWarmup(dummy_optimizer, warmup_steps=10, total_steps=100)
        lrs = collect_lr_factors(sched, 11)
        # At step 10 the factor should be 1.0
        assert lrs[10] == pytest.approx(base_lr, rel=1e-5)

    def test_decay_after_warmup(self, dummy_optimizer):
        sched = CosineWithWarmup(dummy_optimizer, warmup_steps=10, total_steps=100)
        lrs = collect_lr_factors(sched, 100)
        # LRs after warmup should generally decrease
        assert lrs[50] < lrs[11]
        assert lrs[99] < lrs[50]

    def test_min_lr_ratio(self, dummy_optimizer):
        sched = CosineWithWarmup(
            dummy_optimizer, warmup_steps=10, total_steps=100, min_lr_ratio=0.1
        )
        lrs = collect_lr_factors(sched, 101)
        # Final LR should approach min_lr_ratio * base_lr
        assert lrs[-1] >= 0.1 * 1e-3 * 0.99  # allow small tolerance


# ---------------------------------------------------------------------------
# LinearWithWarmup
# ---------------------------------------------------------------------------

class TestLinearWithWarmup:

    def test_warmup_ramp(self, dummy_optimizer):
        sched = LinearWithWarmup(dummy_optimizer, warmup_steps=10, total_steps=100)
        lrs = collect_lr_factors(sched, 11)
        for i in range(1, 10):
            assert lrs[i] > lrs[i - 1] or i == 0

    def test_linear_decay(self, dummy_optimizer):
        sched = LinearWithWarmup(dummy_optimizer, warmup_steps=10, total_steps=100)
        lrs = collect_lr_factors(sched, 100)
        # After warmup, LR should decrease linearly
        assert lrs[50] < lrs[11]
        assert lrs[99] < lrs[50]

    def test_reaches_zero(self, dummy_optimizer):
        sched = LinearWithWarmup(dummy_optimizer, warmup_steps=10, total_steps=100)
        lrs = collect_lr_factors(sched, 101)
        assert lrs[-1] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestCreateScheduler:

    def test_cosine(self, dummy_optimizer):
        cfg = TrainingConfig(lr_scheduler="cosine", warmup_steps=5, lr=1e-3, min_lr=1e-5)
        sched = create_scheduler(dummy_optimizer, cfg, num_training_steps=50)
        assert isinstance(sched, CosineWithWarmup)

    def test_linear(self, dummy_optimizer):
        cfg = TrainingConfig(lr_scheduler="linear", warmup_steps=5)
        sched = create_scheduler(dummy_optimizer, cfg, num_training_steps=50)
        assert isinstance(sched, LinearWithWarmup)

    def test_unknown_raises(self, dummy_optimizer):
        cfg = TrainingConfig(lr_scheduler="nonexistent_scheduler")
        with pytest.raises(ValueError, match="Unknown scheduler"):
            create_scheduler(dummy_optimizer, cfg, num_training_steps=50)
