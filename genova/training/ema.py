"""Exponential Moving Average (EMA) of model weights for Genova.

Maintains a shadow copy of model parameters updated with an exponential
moving average.  Useful for stabilising evaluation metrics during training
and often leads to better generalisation.

Example::

    from genova.training.ema import EMAModel

    ema = EMAModel(model, decay=0.999)

    for batch in train_loader:
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
        ema.update()

    # Evaluate with EMA weights
    ema.apply_shadow()
    evaluate(model)
    ema.restore()
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class EMAModel:
    """Exponential Moving Average wrapper for any :class:`nn.Module`.

    Keeps a *shadow* copy of every parameter in the wrapped model.  After
    each training step the caller invokes :meth:`update` to blend the
    current model weights into the shadow::

        shadow = decay * shadow + (1 - decay) * param

    For evaluation, :meth:`apply_shadow` swaps the shadow weights into the
    model (saving the training weights internally).  :meth:`restore` swaps
    them back.

    Args:
        model: The model whose parameters will be tracked.
        decay: EMA decay factor.  Values close to 1 (e.g. 0.999 or 0.9999)
            produce a slower-moving average.
        device: Optional device to store the shadow parameters on.  If
            ``None``, they live on the same device as the model params.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.decay = decay
        self.device = device

        # Shadow parameters -- deep copy of current model params
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                shadow_val = param.data.clone()
                if device is not None:
                    shadow_val = shadow_val.to(device)
                self.shadow[name] = shadow_val

        # Backup storage for training weights when shadow is applied
        self._backup: Dict[str, torch.Tensor] = {}

        # Keep a reference to the model (not a copy)
        self._model = model

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, model: Optional[nn.Module] = None) -> None:
        """Update the shadow parameters with the current model weights.

        ``shadow = decay * shadow + (1 - decay) * param``

        Args:
            model: Model to read current weights from.  If ``None``, uses
                the model passed at construction time.
        """
        model = model or self._model
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow and param.requires_grad:
                    new_val = param.data
                    if self.device is not None:
                        new_val = new_val.to(self.device)
                    self.shadow[name].mul_(self.decay).add_(
                        new_val, alpha=1.0 - self.decay
                    )

    def apply_shadow(self, model: Optional[nn.Module] = None) -> None:
        """Swap EMA shadow weights into the model for evaluation.

        The current training weights are saved internally so they can be
        restored later with :meth:`restore`.

        Args:
            model: Target model.  If ``None``, uses the construction-time
                model.
        """
        model = model or self._model
        self._backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow and param.requires_grad:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device))

    def restore(self, model: Optional[nn.Module] = None) -> None:
        """Restore the original training weights after :meth:`apply_shadow`.

        Args:
            model: Target model.  If ``None``, uses the construction-time
                model.

        Raises:
            RuntimeError: If called before :meth:`apply_shadow`.
        """
        model = model or self._model
        if not self._backup:
            raise RuntimeError(
                "restore() called before apply_shadow(). Nothing to restore."
            )
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    # ------------------------------------------------------------------
    # State dict support
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return the EMA state as a serialisable dictionary.

        Returns:
            Dictionary with ``decay`` and ``shadow`` entries.
        """
        return {
            "decay": self.decay,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load EMA state from a dictionary produced by :meth:`state_dict`.

        Args:
            state: Dictionary with ``decay`` and ``shadow`` keys.
        """
        self.decay = state["decay"]
        for name, tensor in state["shadow"].items():
            if name in self.shadow:
                target_device = self.shadow[name].device
                self.shadow[name] = tensor.to(target_device)
