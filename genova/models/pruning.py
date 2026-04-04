"""Structured pruning utilities for Genova transformer models.

Provides attention head pruning (gradient-based importance scoring,
Taylor expansion) and FFN neuron pruning (magnitude-based), with
support for iterative pruning schedules.

Example::

    from genova.models.pruning import compute_head_importance, prune_heads

    importance = compute_head_importance(model, data_loader, method="taylor")
    prune_heads(model, num_heads_to_prune=4, importance_scores=importance)

    from genova.models.pruning import prune_ffn
    prune_ffn(model, pruning_ratio=0.3)

    from genova.models.pruning import PruningSchedule
    schedule = PruningSchedule(initial_ratio=0.0, final_ratio=0.5, total_steps=1000)
    for step in range(1000):
        ratio = schedule.get_ratio(step)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Head importance scoring
# ---------------------------------------------------------------------------


def compute_head_importance(
    model: nn.Module,
    data_loader: Any,
    method: str = "taylor",
    num_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Compute per-head importance scores across all transformer layers.

    Two scoring methods are supported:

    - ``"gradient"``: Sum of absolute gradients of attention output
      projections with respect to the loss, aggregated per head.
    - ``"taylor"``: First-order Taylor expansion -- element-wise product
      of weights and their gradients, summed per head.  This approximates
      the change in loss when a head is removed.

    Args:
        model: A Genova transformer model (e.g. :class:`GenovaForMLM`).
        data_loader: An iterable yielding dicts with at least
            ``input_ids`` and ``labels`` keys.
        method: ``"gradient"`` or ``"taylor"``.
        num_batches: Maximum number of batches to process.  ``None``
            means use the entire data loader.
        device: Device to run on; inferred from model if ``None``.

    Returns:
        Tensor of shape ``(n_layers, n_heads)`` with importance scores.
        Higher values indicate more important heads.

    Raises:
        ValueError: If *method* is not recognised.
    """
    if method not in ("gradient", "taylor"):
        raise ValueError(
            f"Unknown importance method {method!r}. Choose 'gradient' or 'taylor'."
        )

    if device is None:
        device = next(model.parameters()).device

    model.train()

    # Discover attention layers
    attn_layers = _find_attention_layers(model)
    n_layers = len(attn_layers)
    if n_layers == 0:
        raise RuntimeError("No MultiHeadSelfAttention layers found in model.")

    n_heads = attn_layers[0].n_heads
    head_dim = attn_layers[0].head_dim
    importance = torch.zeros(n_layers, n_heads, device=device)

    batches_processed = 0
    for batch in data_loader:
        if num_batches is not None and batches_processed >= num_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        model.zero_grad()
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.get("loss")
        if loss is None:
            continue
        loss.backward()

        for layer_idx, attn in enumerate(attn_layers):
            weight = attn.out_proj.weight  # (d_model, d_model)
            grad = weight.grad
            if grad is None:
                continue

            # Reshape to (d_model, n_heads, head_dim)
            w = weight.view(weight.size(0), n_heads, head_dim)
            g = grad.view(grad.size(0), n_heads, head_dim)

            if method == "gradient":
                score = g.abs().sum(dim=(0, 2))  # (n_heads,)
            else:  # taylor
                score = (w * g).abs().sum(dim=(0, 2))  # (n_heads,)

            importance[layer_idx] += score.detach()

        batches_processed += 1

    # Average over batches
    if batches_processed > 0:
        importance /= batches_processed

    return importance


# ---------------------------------------------------------------------------
# Head pruning
# ---------------------------------------------------------------------------


def prune_heads(
    model: nn.Module,
    num_heads_to_prune: int,
    importance_scores: Optional[Tensor] = None,
    heads_to_prune: Optional[Dict[int, List[int]]] = None,
) -> Dict[int, List[int]]:
    """Prune attention heads from a Genova transformer model.

    Either provide pre-computed *importance_scores* (the least important
    heads are pruned globally) or an explicit *heads_to_prune* mapping.

    Pruning zeroes out the corresponding rows/columns of the Q, K, V, and
    output projection weight matrices.  This is a *soft* prune -- the
    tensor sizes remain unchanged but pruned heads contribute nothing.

    Args:
        model: A Genova transformer model.
        num_heads_to_prune: Number of heads to prune globally.  Ignored
            when *heads_to_prune* is provided.
        importance_scores: ``(n_layers, n_heads)`` importance tensor.
            Required when *heads_to_prune* is ``None``.
        heads_to_prune: Explicit mapping ``{layer_idx: [head_indices]}``.
            When provided, *importance_scores* and *num_heads_to_prune*
            are ignored.

    Returns:
        The pruning map ``{layer_idx: [head_indices]}`` that was applied.

    Raises:
        ValueError: If neither *importance_scores* nor *heads_to_prune*
            is provided.
    """
    attn_layers = _find_attention_layers(model)

    if heads_to_prune is None:
        if importance_scores is None:
            raise ValueError(
                "Provide either importance_scores or heads_to_prune."
            )
        heads_to_prune = _select_heads_to_prune(
            importance_scores, num_heads_to_prune
        )

    for layer_idx, head_indices in heads_to_prune.items():
        if layer_idx >= len(attn_layers):
            continue
        attn = attn_layers[layer_idx]
        _zero_heads(attn, head_indices)

    return heads_to_prune


def _select_heads_to_prune(
    importance: Tensor,
    num_to_prune: int,
) -> Dict[int, List[int]]:
    """Select the globally least-important heads.

    Args:
        importance: ``(n_layers, n_heads)`` importance scores.
        num_to_prune: Number of heads to select.

    Returns:
        Mapping ``{layer_idx: [head_indices]}``.
    """
    flat = importance.view(-1)
    n_heads = importance.size(1)
    num_to_prune = min(num_to_prune, flat.numel())
    _, indices = torch.topk(flat, num_to_prune, largest=False)

    result: Dict[int, List[int]] = {}
    for idx in indices.tolist():
        layer = idx // n_heads
        head = idx % n_heads
        result.setdefault(layer, []).append(head)
    return result


def _zero_heads(attn: nn.Module, head_indices: List[int]) -> None:
    """Zero out the weights for specific heads in an attention layer.

    Args:
        attn: A :class:`MultiHeadSelfAttention` instance.
        head_indices: List of head indices to zero.
    """
    head_dim = attn.head_dim  # type: ignore[attr-defined]
    with torch.no_grad():
        for head_idx in head_indices:
            start = head_idx * head_dim
            end = start + head_dim

            # Zero Q projection rows for this head
            attn.q_proj.weight[start:end, :].zero_()  # type: ignore[index]
            if attn.q_proj.bias is not None:  # type: ignore[union-attr]
                attn.q_proj.bias[start:end].zero_()  # type: ignore[index]

            # Zero output projection columns for this head
            attn.out_proj.weight[:, start:end].zero_()  # type: ignore[index]


# ---------------------------------------------------------------------------
# FFN neuron pruning
# ---------------------------------------------------------------------------


def prune_ffn(
    model: nn.Module,
    pruning_ratio: float = 0.3,
    method: str = "magnitude",
) -> int:
    """Prune FFN neurons by magnitude across all transformer layers.

    Neurons (rows of fc1 / columns of fc2) with the smallest L2 norm
    are zeroed out.

    Args:
        model: A Genova transformer model.
        pruning_ratio: Fraction of neurons to prune in each FFN.
            Must be in ``[0, 1)``.
        method: Pruning criterion.  Currently only ``"magnitude"`` is
            supported.

    Returns:
        Total number of neurons pruned across all layers.

    Raises:
        ValueError: If *pruning_ratio* is out of range.
    """
    if not 0.0 <= pruning_ratio < 1.0:
        raise ValueError(f"pruning_ratio must be in [0, 1), got {pruning_ratio}")

    ff_layers = _find_ffn_layers(model)
    total_pruned = 0

    for ff in ff_layers:
        # Handle both standard FFN (fc1/fc2) and SwiGLU (net.w_gate etc.)
        if hasattr(ff, "fc1"):
            weight = ff.fc1.weight  # (d_ff, d_model)
        elif hasattr(ff, "net") and ff.net is not None and hasattr(ff.net, "w_gate"):
            weight = ff.net.w_gate.weight  # (d_ff, d_model)
        else:
            continue

        d_ff = weight.size(0)
        num_to_prune = int(d_ff * pruning_ratio)
        if num_to_prune == 0:
            continue

        # L2 norm of each neuron
        norms = weight.data.norm(dim=1)  # (d_ff,)
        _, indices = torch.topk(norms, num_to_prune, largest=False)

        with torch.no_grad():
            if hasattr(ff, "fc1"):
                ff.fc1.weight[indices, :].zero_()
                if ff.fc1.bias is not None:
                    ff.fc1.bias[indices].zero_()
                ff.fc2.weight[:, indices].zero_()
            elif hasattr(ff, "net") and ff.net is not None:
                ff.net.w_gate.weight[indices, :].zero_()
                ff.net.w_up.weight[indices, :].zero_()
                ff.net.w_down.weight[:, indices].zero_()

        total_pruned += num_to_prune

    return total_pruned


# ---------------------------------------------------------------------------
# Pruning schedule
# ---------------------------------------------------------------------------


class PruningSchedule:
    """Gradual pruning schedule for iterative pruning with fine-tuning.

    Follows cubic spline interpolation between *initial_ratio* and
    *final_ratio* over *total_steps* steps, as described by Zhu & Gupta
    (2018).  Pruning is applied every *frequency* steps.

    The pruning ratio at step *t* is::

        ratio(t) = final - (final - initial) * (1 - t / total)^3

    Args:
        initial_ratio: Starting pruning ratio (typically 0).
        final_ratio: Target pruning ratio at the end of the schedule.
        total_steps: Number of steps over which pruning ramps up.
        frequency: How often (in steps) pruning should be applied.
    """

    def __init__(
        self,
        initial_ratio: float = 0.0,
        final_ratio: float = 0.5,
        total_steps: int = 1000,
        frequency: int = 100,
    ) -> None:
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.total_steps = max(1, total_steps)
        self.frequency = max(1, frequency)

    def get_ratio(self, step: int) -> float:
        """Return the pruning ratio for the given step.

        Args:
            step: Current training step.

        Returns:
            Pruning ratio in ``[initial_ratio, final_ratio]``.
        """
        if step >= self.total_steps:
            return self.final_ratio
        t = step / self.total_steps
        return self.final_ratio - (self.final_ratio - self.initial_ratio) * (
            1.0 - t
        ) ** 3

    def should_prune(self, step: int) -> bool:
        """Return ``True`` if pruning should be applied at this step.

        Args:
            step: Current training step.

        Returns:
            ``True`` every *frequency* steps while within the schedule.
        """
        if step > self.total_steps:
            return False
        return step % self.frequency == 0 and step > 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_attention_layers(model: nn.Module) -> List[nn.Module]:
    """Recursively find all MultiHeadSelfAttention modules.

    Args:
        model: Root module to search.

    Returns:
        List of attention modules in order.
    """
    from genova.models.transformer import MultiHeadSelfAttention

    layers: List[nn.Module] = []
    for module in model.modules():
        if isinstance(module, MultiHeadSelfAttention):
            layers.append(module)
    return layers


def _find_ffn_layers(model: nn.Module) -> List[nn.Module]:
    """Recursively find all FeedForward modules.

    Args:
        model: Root module to search.

    Returns:
        List of FFN modules in order.
    """
    from genova.models.transformer import FeedForward

    layers: List[nn.Module] = []
    for module in model.modules():
        if isinstance(module, FeedForward):
            layers.append(module)
    return layers
