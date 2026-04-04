"""Fully Sharded Data Parallel (FSDP) utilities for Genova training.

Provides FSDP wrapping, mixed-precision policies, auto-wrap policies,
checkpoint saving/loading, and optional CPU offloading.

Example::

    from genova.training.fsdp import FSDPConfig, setup_fsdp

    fsdp_cfg = FSDPConfig(sharding_strategy="FULL_SHARD", mixed_precision="bf16")
    model = setup_fsdp(model, fsdp_cfg)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set, Type, Union

import torch
import torch.nn as nn
from loguru import logger

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        CPUOffload,
        StateDictType,
        FullStateDictConfig,
        BackwardPrefetch,
    )
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )

    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Sharding strategy mapping
# ---------------------------------------------------------------------------

_SHARDING_STRATEGY_MAP: Dict[str, Any] = {}
if FSDP_AVAILABLE:
    _SHARDING_STRATEGY_MAP = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class FSDPConfig:
    """Configuration for Fully Sharded Data Parallel wrapping.

    Attributes:
        enabled: Whether FSDP is enabled.
        sharding_strategy: One of ``"FULL_SHARD"``, ``"SHARD_GRAD_OP"``,
            or ``"NO_SHARD"``.
        mixed_precision: Mixed precision mode for FSDP. One of ``"fp16"``,
            ``"bf16"``, or ``"none"`` / ``""``.
        auto_wrap_min_params: Minimum number of parameters for a sub-module
            to be automatically wrapped as its own FSDP unit.  Set to 0 to
            disable size-based auto-wrapping.
        transformer_layer_cls_names: Optional set of transformer layer class
            names to use with transformer auto-wrap policy.  When provided,
            this takes precedence over *auto_wrap_min_params*.
        cpu_offload: If ``True``, offload parameters and gradients to CPU
            when not in use (saves GPU memory at the cost of speed).
        backward_prefetch: Backward prefetch strategy.  One of
            ``"BACKWARD_PRE"`` or ``"BACKWARD_POST"``.
        sync_module_states: If ``True``, broadcast module states from rank 0
            at FSDP init so all ranks start with identical weights.
        use_orig_params: If ``True``, use the original parameter interface
            (required for ``torch.compile`` compatibility).
        limit_all_gathers: If ``True``, limit in-flight all-gathers to
            reduce memory pressure.
    """

    enabled: bool = True
    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: str = "bf16"
    auto_wrap_min_params: int = 100_000
    transformer_layer_cls_names: Optional[Set[str]] = None
    cpu_offload: bool = False
    backward_prefetch: str = "BACKWARD_PRE"
    sync_module_states: bool = True
    use_orig_params: bool = False
    limit_all_gathers: bool = True


# ---------------------------------------------------------------------------
# Mixed precision policy
# ---------------------------------------------------------------------------


def _build_mixed_precision(precision: str) -> Optional[Any]:
    """Build an FSDP :class:`MixedPrecision` policy.

    Args:
        precision: ``"fp16"``, ``"bf16"``, or ``"none"``/``""``.

    Returns:
        A :class:`MixedPrecision` instance, or ``None`` if disabled.
    """
    if not FSDP_AVAILABLE:
        return None

    if precision == "bf16":
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif precision == "fp16":
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    return None


# ---------------------------------------------------------------------------
# Auto-wrap policy
# ---------------------------------------------------------------------------


def _build_auto_wrap_policy(
    config: FSDPConfig,
    model: nn.Module,
) -> Optional[Any]:
    """Build an FSDP auto-wrap policy from *config*.

    If *transformer_layer_cls_names* is specified, a transformer-based
    policy is used; otherwise a size-based policy is used when
    *auto_wrap_min_params* > 0.

    Args:
        config: FSDP configuration.
        model: The model (used to resolve class names).

    Returns:
        A callable auto-wrap policy, or ``None`` if auto-wrap is disabled.
    """
    if not FSDP_AVAILABLE:
        return None

    # Transformer-class-based wrapping
    if config.transformer_layer_cls_names:
        cls_set: set = set()
        # Resolve class names to actual classes from the model's module tree
        for name, mod in model.named_modules():
            if type(mod).__name__ in config.transformer_layer_cls_names:
                cls_set.add(type(mod))
        if cls_set:
            return transformer_auto_wrap_policy(
                transformer_layer_cls=cls_set,
            )
        logger.warning(
            "transformer_layer_cls_names={} specified but no matching "
            "modules found; falling back to size-based policy.",
            config.transformer_layer_cls_names,
        )

    # Size-based wrapping
    if config.auto_wrap_min_params > 0:
        return size_based_auto_wrap_policy(
            min_num_params=config.auto_wrap_min_params,
        )

    return None


# ---------------------------------------------------------------------------
# Setup function
# ---------------------------------------------------------------------------


def setup_fsdp(
    model: nn.Module,
    config: FSDPConfig,
    device_id: Optional[int] = None,
) -> nn.Module:
    """Wrap *model* with FSDP according to *config*.

    Args:
        model: The unwrapped :class:`nn.Module` to shard.
        config: FSDP configuration dataclass.
        device_id: CUDA device id for this rank.  Defaults to the current
            CUDA device.

    Returns:
        The FSDP-wrapped model.

    Raises:
        RuntimeError: If FSDP is not available (requires PyTorch >= 1.12).
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError(
            "FSDP is not available. Upgrade to PyTorch >= 1.12 with "
            "torch.distributed.fsdp support."
        )

    if not config.enabled:
        logger.info("FSDP config present but disabled; returning unwrapped model.")
        return model

    if device_id is None:
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else None

    # Sharding strategy
    sharding = _SHARDING_STRATEGY_MAP.get(config.sharding_strategy)
    if sharding is None:
        raise ValueError(
            f"Unknown sharding_strategy {config.sharding_strategy!r}. "
            f"Must be one of {list(_SHARDING_STRATEGY_MAP.keys())}."
        )

    # Mixed precision
    mp_policy = _build_mixed_precision(config.mixed_precision)

    # Auto-wrap policy
    auto_wrap = _build_auto_wrap_policy(config, model)

    # CPU offload
    cpu_offload = CPUOffload(offload_params=True) if config.cpu_offload else None

    # Backward prefetch
    backward_prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
    }
    backward_prefetch = backward_prefetch_map.get(
        config.backward_prefetch, BackwardPrefetch.BACKWARD_PRE
    )

    fsdp_model = FSDP(
        model,
        sharding_strategy=sharding,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap,
        cpu_offload=cpu_offload,
        backward_prefetch=backward_prefetch,
        device_id=device_id,
        sync_module_states=config.sync_module_states,
        use_orig_params=config.use_orig_params,
        limit_all_gathers=config.limit_all_gathers,
    )

    logger.info(
        "Model wrapped with FSDP: sharding={}, mixed_precision={}, "
        "cpu_offload={}, auto_wrap_min_params={}",
        config.sharding_strategy,
        config.mixed_precision,
        config.cpu_offload,
        config.auto_wrap_min_params,
    )

    return fsdp_model


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_fsdp_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Union[str, Path],
    *,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: float = float("inf"),
    config: Optional[Dict[str, Any]] = None,
    scheduler_state: Optional[Dict[str, Any]] = None,
    scaler_state: Optional[Dict[str, Any]] = None,
    rank: int = 0,
) -> None:
    """Save a full-state FSDP checkpoint.

    Uses ``StateDictType.FULL_STATE_DICT`` so that a single consolidated
    checkpoint is saved on rank 0, compatible with non-FSDP loading.

    Args:
        model: The FSDP-wrapped model.
        optimizer: The optimizer.
        path: Destination file path.
        epoch: Current epoch number.
        global_step: Current global step.
        best_metric: Best validation metric so far.
        config: Optional config dict to embed in the checkpoint.
        scheduler_state: Optional scheduler state dict.
        scaler_state: Optional GradScaler state dict.
        rank: Current process rank.
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available.")

    path = Path(path)

    # Configure full state dict gathering
    full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)

    # Only rank 0 actually writes the file
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: Dict[str, Any] = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "fsdp": True,
        }
        if config is not None:
            checkpoint["config"] = config
        if scheduler_state is not None:
            checkpoint["scheduler_state_dict"] = scheduler_state
        if scaler_state is not None:
            checkpoint["scaler_state_dict"] = scaler_state

        torch.save(checkpoint, path)
        logger.info("FSDP checkpoint saved to {} (rank 0).", path)


def load_fsdp_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a checkpoint previously saved with :func:`save_fsdp_checkpoint`.

    The model and optimizer states are loaded in-place.  The remaining
    metadata (epoch, global_step, best_metric, etc.) is returned as a dict.

    Args:
        model: The FSDP-wrapped model.
        optimizer: The optimizer (may be ``None`` to skip optimizer loading).
        path: Checkpoint file path.
        device: Device to map tensors to during loading.

    Returns:
        Dictionary containing ``epoch``, ``global_step``, ``best_metric``,
        and any other metadata stored in the checkpoint.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available.")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FSDP checkpoint not found: {path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Load model state
    full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
        model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optim_state = FSDP.optim_state_dict_to_load(
            model, optimizer, checkpoint["optimizer_state_dict"]
        )
        optimizer.load_state_dict(optim_state)

    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "best_metric": checkpoint.get("best_metric", float("inf")),
        "config": checkpoint.get("config"),
        "scheduler_state_dict": checkpoint.get("scheduler_state_dict"),
        "scaler_state_dict": checkpoint.get("scaler_state_dict"),
    }

    logger.info(
        "FSDP checkpoint loaded from {} (epoch={}, step={}).",
        path,
        metadata["epoch"],
        metadata["global_step"],
    )

    return metadata
