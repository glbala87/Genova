"""Differential privacy training for Genova models.

Implements DP-SGD (Differentially Private Stochastic Gradient Descent) with
privacy budget tracking via Renyi Differential Privacy (RDP) accountant.

If ``opacus`` is installed it is used as the backend.  Otherwise a lightweight
built-in implementation provides the core DP-SGD primitives.

Usage::

    from genova.training.differential_privacy import DPTrainer

    dp_trainer = DPTrainer(model, optimizer, data_loader)
    dp_trainer.configure(epsilon=8.0, delta=1e-5, max_grad_norm=1.0, noise_multiplier=1.1)
    for epoch in range(10):
        loss = dp_trainer.train_step()
    eps, delta = dp_trainer.get_privacy_spent()
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from opacus import PrivacyEngine
    from opacus.accountants import RDPAccountant
    from opacus.validators import ModuleValidator

    _HAS_OPACUS = True
except ImportError:
    _HAS_OPACUS = False


# ---------------------------------------------------------------------------
# RDP Accountant (built-in fallback)
# ---------------------------------------------------------------------------

class SimpleRDPAccountant:
    """Lightweight Renyi Differential Privacy accountant.

    Tracks privacy spent via the moments accountant method when ``opacus``
    is not available.

    Parameters
    ----------
    orders : list of float, optional
        RDP orders (alpha values) to track.
    """

    def __init__(
        self,
        orders: Optional[List[float]] = None,
    ) -> None:
        self.orders = orders or [
            1 + x / 10.0 for x in range(1, 100)
        ] + list(range(12, 64))
        self._rdp = np.zeros(len(self.orders))
        self._steps = 0

    def step(
        self,
        noise_multiplier: float,
        sample_rate: float,
    ) -> None:
        """Record one DP-SGD step.

        Parameters
        ----------
        noise_multiplier : float
            Ratio of noise standard deviation to clipping norm.
        sample_rate : float
            Probability that each sample is included in a mini-batch
            (batch_size / dataset_size).
        """
        self._rdp += self._compute_rdp(noise_multiplier, sample_rate)
        self._steps += 1

    def get_privacy_spent(
        self,
        delta: float = 1e-5,
    ) -> Tuple[float, float]:
        """Convert accumulated RDP to (epsilon, delta).

        Parameters
        ----------
        delta : float
            Target delta.

        Returns
        -------
        tuple of (epsilon, delta)
        """
        eps = self._rdp_to_eps(self._rdp, delta)
        return (float(eps), delta)

    def _compute_rdp(
        self,
        sigma: float,
        q: float,
    ) -> np.ndarray:
        """Compute RDP guarantee for a single step of subsampled Gaussian."""
        rdp = np.zeros(len(self.orders))
        for i, alpha in enumerate(self.orders):
            if alpha <= 1:
                continue
            if sigma == 0:
                rdp[i] = float("inf")
            else:
                rdp[i] = alpha / (2 * sigma ** 2)
                # Subsampling amplification (simplified)
                if q < 1.0:
                    rdp[i] = (1.0 / (alpha - 1)) * math.log(
                        1 + q ** 2 * (math.exp((alpha - 1) * alpha / (2 * sigma ** 2)) - 1)
                    )
        return rdp

    def _rdp_to_eps(
        self,
        rdp: np.ndarray,
        delta: float,
    ) -> float:
        """Convert RDP values to epsilon using the optimal conversion."""
        eps_candidates = []
        for i, alpha in enumerate(self.orders):
            if alpha <= 1:
                continue
            eps = rdp[i] - math.log(delta) / (alpha - 1) + math.log(
                (alpha - 1) / alpha
            )
            eps_candidates.append(eps)
        return min(eps_candidates) if eps_candidates else float("inf")


# ---------------------------------------------------------------------------
# Per-sample gradient clipping
# ---------------------------------------------------------------------------

def clip_gradients_per_sample(
    model: "nn.Module",
    max_grad_norm: float,
) -> float:
    """Clip per-parameter gradients to *max_grad_norm* (L2).

    This is a simplified version; true per-sample clipping requires
    per-sample gradient computation (e.g., via ``functorch`` or ``opacus``).

    Returns
    -------
    float
        The total gradient norm before clipping.
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.0

    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    return float(total_norm)


def add_noise_to_gradients(
    model: "nn.Module",
    noise_multiplier: float,
    max_grad_norm: float,
) -> None:
    """Add calibrated Gaussian noise to gradients for DP-SGD.

    Parameters
    ----------
    model : nn.Module
        Model whose ``.grad`` tensors receive noise.
    noise_multiplier : float
        Ratio of noise std to clipping norm.
    max_grad_norm : float
        The clipping norm (noise std = noise_multiplier * max_grad_norm).
    """
    noise_std = noise_multiplier * max_grad_norm
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_std
            param.grad.add_(noise)


# ---------------------------------------------------------------------------
# DPTrainer
# ---------------------------------------------------------------------------

class DPTrainer:
    """Differentially-private trainer wrapping a standard training loop.

    Uses ``opacus`` if available; otherwise falls back to a manual
    implementation of gradient clipping + noise addition + RDP accounting.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    optimizer : torch.optim.Optimizer
        The optimizer.
    data_loader : DataLoader
        Training data loader.
    loss_fn : callable, optional
        Loss function. Defaults to ``nn.CrossEntropyLoss()``.
    device : str
        Device for training.
    """

    def __init__(
        self,
        model: Any = None,
        optimizer: Any = None,
        data_loader: Any = None,
        loss_fn: Any = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.device = device
        self._loss_fn = loss_fn

        # DP parameters
        self.epsilon: float = 8.0
        self.delta: float = 1e-5
        self.max_grad_norm: float = 1.0
        self.noise_multiplier: float = 1.1

        # State
        self._configured = False
        self._using_opacus = False
        self._accountant: Optional[Any] = None
        self._privacy_engine: Optional[Any] = None
        self._dataset_size: int = 0
        self._batch_size: int = 1
        self._step_count: int = 0

    def configure(
        self,
        epsilon: float = 8.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        target_epochs: Optional[int] = None,
    ) -> None:
        """Configure differential privacy parameters.

        Parameters
        ----------
        epsilon : float
            Target privacy budget epsilon.
        delta : float
            Target delta (should be < 1/N where N is dataset size).
        max_grad_norm : float
            Maximum L2 norm for per-sample gradient clipping.
        noise_multiplier : float, optional
            Noise multiplier (sigma). If None, it is automatically calibrated
            from epsilon/delta/target_epochs.
        target_epochs : int, optional
            Number of training epochs (used for noise calibration).
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

        if noise_multiplier is not None:
            self.noise_multiplier = noise_multiplier
        elif target_epochs is not None and self.data_loader is not None:
            # Rough calibration
            self._dataset_size = len(self.data_loader.dataset) if hasattr(self.data_loader, "dataset") else 1000
            self._batch_size = getattr(self.data_loader, "batch_size", 32) or 32
            steps = target_epochs * (self._dataset_size // self._batch_size)
            q = self._batch_size / self._dataset_size
            # Heuristic: sigma ~ sqrt(2 * steps * q^2 * log(1/delta)) / epsilon
            self.noise_multiplier = max(
                0.1,
                math.sqrt(2 * steps * q ** 2 * math.log(1.0 / delta)) / epsilon,
            )

        if self.data_loader is not None:
            self._dataset_size = len(self.data_loader.dataset) if hasattr(self.data_loader, "dataset") else 1000
            self._batch_size = getattr(self.data_loader, "batch_size", 32) or 32

        # Try opacus
        if _HAS_OPACUS and self.model is not None and self.optimizer is not None and self.data_loader is not None:
            try:
                # Validate model
                self.model = ModuleValidator.fix(self.model)

                self._privacy_engine = PrivacyEngine()
                self.model, self.optimizer, self.data_loader = self._privacy_engine.make_private_with_epsilon(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.data_loader,
                    epochs=target_epochs or 1,
                    target_epsilon=epsilon,
                    target_delta=delta,
                    max_grad_norm=max_grad_norm,
                )
                self._using_opacus = True
                logger.info(
                    "DP training configured with Opacus (eps={}, delta={}, sigma={:.4f})",
                    epsilon, delta, self.noise_multiplier,
                )
            except Exception as e:
                logger.warning("Opacus setup failed, falling back to manual DP: {}", e)
                self._using_opacus = False

        if not self._using_opacus:
            self._accountant = SimpleRDPAccountant()
            logger.info(
                "DP training configured (manual): eps={}, delta={}, "
                "sigma={:.4f}, clip={:.4f}",
                epsilon, delta, self.noise_multiplier, max_grad_norm,
            )

        self._configured = True

    def train_step(self) -> float:
        """Execute one training step with private gradients.

        Returns
        -------
        float
            The training loss for this step.
        """
        if not self._configured:
            raise RuntimeError("Call configure() before train_step().")
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for training.")

        self.model.train()

        if self._loss_fn is None:
            self._loss_fn = nn.CrossEntropyLoss()

        # Get a batch
        if not hasattr(self, "_data_iter"):
            self._data_iter = iter(self.data_loader)
        try:
            batch = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.data_loader)
            batch = next(self._data_iter)

        # Unpack batch (expecting dict or tuple)
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)
            attention_mask = batch.get("attention_mask", None)
        elif isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(self.device)
            labels = batch[1].to(self.device) if len(batch) > 1 else input_ids
            attention_mask = batch[2].to(self.device) if len(batch) > 2 else None
        else:
            input_ids = batch.to(self.device)
            labels = input_ids
            attention_mask = None

        # Forward
        self.optimizer.zero_grad()

        kwargs: Dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        outputs = self.model(**kwargs)

        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("last_hidden_state"))
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = getattr(outputs, "logits", outputs)

        if logits.dim() == 3:
            loss = self._loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            loss = self._loss_fn(logits, labels)

        loss.backward()

        # DP mechanisms (only for manual mode)
        if not self._using_opacus:
            clip_gradients_per_sample(self.model, self.max_grad_norm)
            add_noise_to_gradients(self.model, self.noise_multiplier, self.max_grad_norm)

            # Account privacy
            sample_rate = self._batch_size / max(self._dataset_size, 1)
            self._accountant.step(self.noise_multiplier, sample_rate)

        self.optimizer.step()
        self._step_count += 1

        return float(loss.item())

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Return the current privacy budget consumed.

        Returns
        -------
        tuple of (epsilon, delta)
        """
        if self._using_opacus and self._privacy_engine is not None:
            return self._privacy_engine.get_epsilon(self.delta), self.delta

        if self._accountant is not None:
            return self._accountant.get_privacy_spent(self.delta)

        return (float("inf"), self.delta)

    @property
    def privacy_budget_remaining(self) -> float:
        """Estimated remaining epsilon budget."""
        spent_eps, _ = self.get_privacy_spent()
        return max(0.0, self.epsilon - spent_eps)

    @property
    def is_budget_exhausted(self) -> bool:
        """Whether the privacy budget has been fully consumed."""
        return self.privacy_budget_remaining <= 0

    def state_dict(self) -> Dict[str, Any]:
        """Serialise DP state for checkpointing."""
        state = {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "max_grad_norm": self.max_grad_norm,
            "noise_multiplier": self.noise_multiplier,
            "step_count": self._step_count,
            "using_opacus": self._using_opacus,
        }
        if self._accountant is not None:
            state["rdp_values"] = self._accountant._rdp.tolist()
            state["rdp_steps"] = self._accountant._steps
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore DP state from a checkpoint."""
        self.epsilon = state.get("epsilon", self.epsilon)
        self.delta = state.get("delta", self.delta)
        self.max_grad_norm = state.get("max_grad_norm", self.max_grad_norm)
        self.noise_multiplier = state.get("noise_multiplier", self.noise_multiplier)
        self._step_count = state.get("step_count", 0)
        if "rdp_values" in state and self._accountant is not None:
            self._accountant._rdp = np.array(state["rdp_values"])
            self._accountant._steps = state.get("rdp_steps", 0)
