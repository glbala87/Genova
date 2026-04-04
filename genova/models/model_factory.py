"""Model factory for Genova.

Provides a single entry-point :func:`create_model` that builds the
appropriate model (Transformer or Mamba) from a :class:`ModelConfig`
and optionally loads pretrained weights.  Supports optional
``torch.compile`` wrapping for PyTorch 2.0+ acceleration.

Example::

    from genova.utils.config import ModelConfig
    cfg = ModelConfig(arch="transformer", d_model=256, n_layers=6,
                      n_heads=4, d_ff=1024, vocab_size=4096)
    model = create_model(cfg)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from genova.utils.config import ModelConfig
from genova.models.transformer import GenovaTransformer, GenovaForMLM
from genova.models.mamba_model import GenovaMamba, GenovaMambaForMLM


# ---------------------------------------------------------------------------
# torch.compile availability
# ---------------------------------------------------------------------------

_HAS_TORCH_COMPILE = hasattr(torch, "compile")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Dict[str, type]] = {
    "transformer": {
        "backbone": GenovaTransformer,
        "mlm": GenovaForMLM,
    },
    "mamba": {
        "backbone": GenovaMamba,
        "mlm": GenovaMambaForMLM,
    },
}


def _maybe_compile(
    model: nn.Module,
    compile_model: bool = False,
    compile_backend: str = "inductor",
) -> nn.Module:
    """Optionally wrap *model* with ``torch.compile``.

    This is a no-op when:
    - ``compile_model`` is ``False``
    - ``torch.compile`` is not available (PyTorch < 2.0)

    Args:
        model: The model to compile.
        compile_model: Whether to apply ``torch.compile``.
        compile_backend: Backend name passed to ``torch.compile``.
            Common values: ``"inductor"`` (default, optimised),
            ``"eager"`` (no compilation, useful for debugging).

    Returns:
        The original or compiled model.
    """
    if not compile_model:
        return model

    if not _HAS_TORCH_COMPILE:
        warnings.warn(
            "torch.compile requested but not available (requires PyTorch >= 2.0). "
            "Falling back to eager mode.",
            RuntimeWarning,
            stacklevel=2,
        )
        return model

    try:
        compiled = torch.compile(model, backend=compile_backend)  # type: ignore[attr-defined]
        return compiled
    except Exception as exc:
        warnings.warn(
            f"torch.compile failed ({exc}); falling back to eager mode.",
            RuntimeWarning,
            stacklevel=2,
        )
        return model


def create_model(
    config: ModelConfig,
    *,
    task: str = "mlm",
    pretrained_path: Optional[Union[str, Path]] = None,
    strict: bool = True,
    embedding_type: str = "learned",
    compile_model: Optional[bool] = None,
    compile_backend: str = "inductor",
    **kwargs: Any,
) -> nn.Module:
    """Instantiate a Genova model from configuration.

    Args:
        config: A :class:`ModelConfig` with at least ``arch`` set to
            ``"transformer"`` or ``"mamba"``.
        task: Model variant to build.  ``"mlm"`` wraps the backbone with
            an MLM head; ``"backbone"`` returns the encoder only.
        pretrained_path: Optional path to a ``state_dict`` checkpoint
            (``.pt`` / ``.bin``).  If provided, weights are loaded after
            construction.
        strict: Passed to :meth:`nn.Module.load_state_dict`.
        embedding_type: Positional encoding style forwarded to the model.
        compile_model: If ``True``, wrap the model with
            ``torch.compile``.  If ``None`` (default), falls back to
            ``config.compile_model`` (from :class:`TrainingConfig`) if
            it exists, otherwise ``False``.
        compile_backend: Backend for ``torch.compile``.  Defaults to
            ``"inductor"``.
        **kwargs: Extra keyword arguments forwarded to the model
            constructor (e.g. ``d_state``, ``d_conv`` for Mamba).

    Returns:
        An :class:`nn.Module` ready for training or inference.

    Raises:
        ValueError: If ``config.arch`` or *task* is not recognised.
    """
    arch = config.arch.lower()
    if arch not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture {config.arch!r}. "
            f"Choose from {list(_REGISTRY.keys())}."
        )

    variant_map = _REGISTRY[arch]
    if task not in variant_map:
        raise ValueError(
            f"Unknown task {task!r} for architecture {arch!r}. "
            f"Choose from {list(variant_map.keys())}."
        )

    model_cls = variant_map[task]
    model = model_cls(config, embedding_type=embedding_type, **kwargs)

    if pretrained_path is not None:
        pretrained_path = Path(pretrained_path)
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        # Handle nested checkpoint format (e.g. {"model_state_dict": ...})
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=strict)

    # --- torch.compile ---
    should_compile = compile_model
    if should_compile is None:
        # Check if config has a compile_model attribute (may come from TrainingConfig)
        should_compile = getattr(config, "compile_model", False)
    model = _maybe_compile(model, compile_model=should_compile, compile_backend=compile_backend)

    return model


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the total number of (optionally trainable) parameters.

    Args:
        model: Any PyTorch module.
        trainable_only: If ``True`` (default), count only parameters
            with ``requires_grad=True``.

    Returns:
        Integer parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_summary(model: nn.Module) -> Dict[str, Any]:
    """Return a summary dict with parameter counts and layer info.

    Args:
        model: Any PyTorch module.

    Returns:
        Dict with ``total_params``, ``trainable_params``,
        ``non_trainable_params``, and ``layer_counts``.
    """
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)

    layer_counts: Dict[str, int] = {}
    for name, _ in model.named_modules():
        parts = name.split(".")
        top = parts[0] if parts[0] else "root"
        layer_counts[top] = layer_counts.get(top, 0) + 1

    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": total - trainable,
        "layer_counts": layer_counts,
    }
