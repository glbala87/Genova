"""Learning rate schedulers for Genova training.

Provides cosine, linear, cosine-annealing-warm-restarts, one-cycle, and
polynomial-decay warmup schedulers, plus a factory that selects the correct
scheduler from a :class:`~genova.utils.config.TrainingConfig`.

Example::

    from genova.training.scheduler import create_scheduler

    scheduler = create_scheduler(optimizer, config, num_training_steps=100_000)
    for step in range(100_000):
        optimizer.step()
        scheduler.step()
"""

from __future__ import annotations

import math
from typing import Any, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from genova.utils.config import TrainingConfig


# ---------------------------------------------------------------------------
# Cosine with warmup
# ---------------------------------------------------------------------------


class CosineWithWarmup(LambdaLR):
    """Cosine-decay learning rate schedule with linear warmup.

    The learning rate increases linearly from 0 to *base_lr* during the
    first *warmup_steps*, then decays following a cosine curve to *min_lr*
    over the remaining steps.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        min_lr_ratio: Ratio of minimum LR to base LR (``min_lr / base_lr``).
            Defaults to 0.0.
        last_epoch: Index of last epoch (for resuming).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        # LambdaLR calls the lambda immediately, so attributes must exist first.
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, current_step: int) -> float:
        """Compute multiplicative factor for the learning rate."""
        if current_step < self.warmup_steps:
            return current_step / max(1, self.warmup_steps)
        progress = (current_step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay


# ---------------------------------------------------------------------------
# Linear with warmup
# ---------------------------------------------------------------------------


class LinearWithWarmup(LambdaLR):
    """Linear-decay learning rate schedule with linear warmup.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        last_epoch: Index of last epoch (for resuming).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, current_step: int) -> float:
        """Compute multiplicative factor for the learning rate."""
        if current_step < self.warmup_steps:
            return current_step / max(1, self.warmup_steps)
        remaining = max(0, self.total_steps - current_step)
        total_decay_steps = max(1, self.total_steps - self.warmup_steps)
        return remaining / total_decay_steps


# ---------------------------------------------------------------------------
# Cosine Annealing with Warm Restarts
# ---------------------------------------------------------------------------


class CosineAnnealingWarmRestarts(LambdaLR):
    """Cosine annealing with periodic warm restarts (Loshchilov & Hutter, 2017).

    After each restart period of ``T_0`` steps, the learning rate is reset
    to the base value.  The period can optionally grow by a factor of
    ``T_mult`` after each restart.  A linear warmup phase of
    ``warmup_steps`` is applied only at the very beginning.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Linear warmup steps at the start of training.
        total_steps: Total number of training steps.
        t_0: Initial restart period length (in steps).  Defaults to
            ``total_steps`` (single cosine cycle).
        t_mult: Multiplicative factor applied to the period after each
            restart.  Defaults to 1 (constant period).
        min_lr_ratio: Ratio of minimum LR to base LR at the trough of
            each cosine cycle.
        last_epoch: For resumption; forwarded to the scheduler.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        t_0: Optional[int] = None,
        t_mult: float = 1.0,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.t_0 = t_0 if t_0 is not None else max(1, total_steps - warmup_steps)
        self.t_mult = t_mult
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, current_step: int) -> float:
        """Compute multiplicative factor for the learning rate."""
        if current_step < self.warmup_steps:
            return current_step / max(1, self.warmup_steps)

        step = current_step - self.warmup_steps
        t_cur = self.t_0
        # Walk through restart periods to find where *step* falls.
        while step >= t_cur and self.t_mult > 0:
            step -= t_cur
            t_cur = int(t_cur * self.t_mult)
            if t_cur <= 0:
                break

        progress = step / max(1, t_cur)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay


# ---------------------------------------------------------------------------
# OneCycleLR wrapper
# ---------------------------------------------------------------------------


class OneCycleLR(LambdaLR):
    """One-cycle learning rate policy (Smith & Topin, 2018).

    The schedule consists of two symmetric cosine phases:

    1. **Warm-up** (first half): LR ramps from ``initial_lr_ratio * base_lr``
       up to ``base_lr``.
    2. **Decay** (second half): LR decays from ``base_lr`` down to
       ``min_lr_ratio * base_lr``.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Ignored (present for factory compatibility).  The
            warm-up fraction is fixed at 50 % of *total_steps*.
        total_steps: Total number of training steps.
        initial_lr_ratio: Starting LR as a fraction of base LR.
        min_lr_ratio: Final LR as a fraction of base LR.
        last_epoch: For resumption.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        initial_lr_ratio: float = 0.1,
        min_lr_ratio: float = 0.01,
        last_epoch: int = -1,
    ) -> None:
        self.total_steps = max(1, total_steps)
        self.initial_lr_ratio = initial_lr_ratio
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, current_step: int) -> float:
        """Compute multiplicative factor for the learning rate."""
        pct = current_step / self.total_steps
        if pct <= 0.5:
            # Warm-up phase: cosine ramp from initial_lr_ratio to 1.0
            progress = pct / 0.5
            return self.initial_lr_ratio + (1.0 - self.initial_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * (1.0 - progress))
            )
        # Decay phase: cosine ramp from 1.0 to min_lr_ratio
        progress = (pct - 0.5) / 0.5
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )


# ---------------------------------------------------------------------------
# Polynomial Decay
# ---------------------------------------------------------------------------


class PolynomialDecay(LambdaLR):
    """Polynomial learning rate decay with linear warmup.

    After warmup, the learning rate decays as::

        lr = (base_lr - min_lr) * (1 - progress)^power + min_lr

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total number of training steps.
        power: Exponent of the polynomial decay.  1.0 gives linear decay,
            2.0 gives quadratic, etc.
        min_lr_ratio: Ratio of minimum LR to base LR.
        last_epoch: For resumption.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        power: float = 2.0,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, current_step: int) -> float:
        """Compute multiplicative factor for the learning rate."""
        if current_step < self.warmup_steps:
            return current_step / max(1, self.warmup_steps)
        progress = (current_step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)
        decay = (1.0 - progress) ** self.power
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * decay


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SCHEDULER_REGISTRY = {
    "cosine": CosineWithWarmup,
    "linear": LinearWithWarmup,
    "cosine_warm_restarts": CosineAnnealingWarmRestarts,
    "one_cycle": OneCycleLR,
    "polynomial": PolynomialDecay,
}


def create_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a learning rate scheduler from a training config.

    The scheduler type is determined by ``config.lr_scheduler``.  Supported
    values: ``"cosine"``, ``"linear"``, ``"cosine_warm_restarts"``,
    ``"one_cycle"``, ``"polynomial"``.

    Args:
        optimizer: The wrapped optimizer.
        config: Training configuration with ``lr_scheduler``,
            ``warmup_steps``, ``lr``, and ``min_lr`` fields.
        num_training_steps: Total number of optimisation steps.
        last_epoch: For resumption; forwarded to the scheduler constructor.

    Returns:
        A configured :class:`LambdaLR` scheduler instance.

    Raises:
        ValueError: If ``config.lr_scheduler`` is not recognised.
    """
    name = config.lr_scheduler.lower()
    if name not in _SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler {config.lr_scheduler!r}. "
            f"Choose from {list(_SCHEDULER_REGISTRY.keys())}."
        )

    cls = _SCHEDULER_REGISTRY[name]

    kwargs: dict[str, Any] = {
        "optimizer": optimizer,
        "warmup_steps": config.warmup_steps,
        "total_steps": num_training_steps,
        "last_epoch": last_epoch,
    }

    base_lr = config.lr
    min_lr = config.min_lr
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    # CosineWithWarmup accepts min_lr_ratio
    if name == "cosine":
        kwargs["min_lr_ratio"] = min_lr_ratio

    # CosineAnnealingWarmRestarts accepts min_lr_ratio (t_0/t_mult use defaults)
    elif name == "cosine_warm_restarts":
        kwargs["min_lr_ratio"] = min_lr_ratio

    # OneCycleLR accepts min_lr_ratio
    elif name == "one_cycle":
        kwargs["min_lr_ratio"] = min_lr_ratio

    # PolynomialDecay accepts min_lr_ratio
    elif name == "polynomial":
        kwargs["min_lr_ratio"] = min_lr_ratio

    return cls(**kwargs)
