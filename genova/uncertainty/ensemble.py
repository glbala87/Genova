"""Ensemble methods for uncertainty estimation in genomic models.

Provides Deep Ensemble (Lakshminarayanan et al., 2017) and Snapshot Ensemble
(Huang et al., 2017) implementations for obtaining calibrated uncertainty
estimates from multiple model predictions.

Example::

    from genova.uncertainty.ensemble import DeepEnsemble, SnapshotEnsemble

    # Deep Ensemble
    ensemble = DeepEnsemble(device="cuda")
    ensemble.add_model(model_1)
    ensemble.add_model(model_2)
    result = ensemble.predict_with_uncertainty(input_ids)

    # Snapshot Ensemble
    snap = SnapshotEnsemble(base_model, device="cuda")
    snap.collect_snapshot()  # call at each LR cycle restart
    result = snap.predict_with_uncertainty(input_ids)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from loguru import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute predictive entropy from probability arrays.

    Args:
        probs: ``(N,)`` or ``(N, C)`` predicted probabilities.

    Returns:
        Entropy values of shape ``(N,)``.
    """
    probs = np.clip(probs, 1e-10, 1.0)
    if probs.ndim == 1:
        probs_2d = np.stack([1 - probs, probs], axis=-1)
    else:
        probs_2d = probs
    return -np.sum(probs_2d * np.log(probs_2d), axis=-1)


def _extract_probs(
    model: nn.Module,
    input_ids: Tensor,
    batch_size: int,
    device: torch.device,
    temperature: float = 1.0,
) -> np.ndarray:
    """Run a model and extract probabilities.

    Args:
        model: PyTorch model.
        input_ids: ``(N, L)`` input tensor.
        batch_size: Batch size for inference.
        device: Device.
        temperature: Temperature scaling parameter.

    Returns:
        Probabilities array.
    """
    model.eval()
    model.to(device)
    all_probs: List[np.ndarray] = []

    for start in range(0, len(input_ids), batch_size):
        batch = input_ids[start: start + batch_size].to(device)
        with torch.no_grad():
            output = model(batch)

        if isinstance(output, dict):
            logits = output.get("logits", output.get("output"))
        elif isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output

        if temperature != 1.0:
            logits = logits / temperature

        if logits.dim() >= 2 and logits.size(-1) > 1:
            probs = F.softmax(logits, dim=-1)
        else:
            probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


# ---------------------------------------------------------------------------
# DeepEnsemble
# ---------------------------------------------------------------------------


class DeepEnsemble:
    """Deep Ensemble uncertainty estimation.

    Aggregates predictions from *N* independently trained models to obtain
    mean predictions and epistemic uncertainty via the variance across
    ensemble members.

    Args:
        device: Inference device.
        batch_size: Batch size for inference.
    """

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self._models: List[nn.Module] = []
        self._temperature: float = 1.0

        logger.info("DeepEnsemble initialised on {}", self.device)

    @property
    def n_models(self) -> int:
        """Number of models in the ensemble."""
        return len(self._models)

    def add_model(self, model: nn.Module) -> None:
        """Add a trained model to the ensemble.

        The model is deep-copied to avoid shared state.

        Args:
            model: Trained PyTorch model.
        """
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        self._models.append(model_copy)
        logger.info(
            "Added model to ensemble (total: {})", len(self._models)
        )

    def remove_model(self, index: int) -> None:
        """Remove a model from the ensemble by index.

        Args:
            index: Index of the model to remove.
        """
        if 0 <= index < len(self._models):
            self._models.pop(index)
            logger.info(
                "Removed model {} from ensemble (remaining: {})",
                index,
                len(self._models),
            )

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        input_ids: Tensor,
        return_all_predictions: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Run ensemble inference and compute uncertainty estimates.

        Args:
            input_ids: ``(N, L)`` input tensor.
            return_all_predictions: If ``True``, include per-model predictions.

        Returns:
            Dictionary with:
                - ``"mean"``: Mean prediction across ensemble members.
                - ``"variance"``: Predictive variance (epistemic uncertainty).
                - ``"std"``: Standard deviation.
                - ``"entropy"``: Predictive entropy of the mean prediction.
                - ``"all_predictions"`` (optional): ``(M, N, ...)`` array.
        """
        if not self._models:
            raise RuntimeError("No models in ensemble. Call add_model() first.")

        all_outputs: List[np.ndarray] = []

        for i, model in enumerate(self._models):
            probs = _extract_probs(
                model, input_ids, self.batch_size, self.device, self._temperature
            )
            all_outputs.append(probs)

        stacked = np.stack(all_outputs, axis=0)  # (M, N, ...)
        mean_pred = stacked.mean(axis=0)
        variance = stacked.var(axis=0)
        std = np.sqrt(variance)
        entropy = _predictive_entropy(mean_pred)

        result: Dict[str, np.ndarray] = {
            "mean": mean_pred,
            "variance": variance,
            "std": std,
            "entropy": entropy,
        }

        if return_all_predictions:
            result["all_predictions"] = stacked

        logger.debug(
            "DeepEnsemble: {} models, mean entropy={:.4f}",
            len(self._models),
            float(entropy.mean()),
        )
        return result

    def calibrate(
        self,
        val_input_ids: Tensor,
        val_labels: np.ndarray,
    ) -> float:
        """Calibrate the ensemble using temperature scaling.

        Finds the optimal temperature parameter by minimizing NLL on
        validation data.

        Args:
            val_input_ids: Validation input tensor.
            val_labels: Ground truth labels.

        Returns:
            Optimal temperature value.
        """
        logger.info("Calibrating DeepEnsemble with temperature scaling...")

        # Get ensemble mean predictions at temperature=1
        self._temperature = 1.0
        result = self.predict_with_uncertainty(val_input_ids)
        mean_probs = result["mean"]
        val_labels = np.asarray(val_labels).ravel()

        def _nll(temp: float) -> float:
            if mean_probs.ndim == 1 or (
                mean_probs.ndim == 2 and mean_probs.shape[1] == 1
            ):
                probs = mean_probs.ravel()
                logits = np.log(
                    np.clip(probs, 1e-10, 1.0)
                    / np.clip(1 - probs, 1e-10, 1.0)
                )
                scaled = 1.0 / (1.0 + np.exp(-logits / temp))
                scaled = np.clip(scaled, 1e-10, 1 - 1e-10)
                return -float(
                    np.mean(
                        val_labels * np.log(scaled)
                        + (1 - val_labels) * np.log(1 - scaled)
                    )
                )
            else:
                logits = np.log(np.clip(mean_probs, 1e-10, 1.0))
                scaled = logits / temp
                exp_s = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
                sp = exp_s / exp_s.sum(axis=-1, keepdims=True)
                sp = np.clip(sp, 1e-10, 1.0)
                labs = val_labels.astype(int)
                return -float(
                    np.mean(np.log(sp[np.arange(len(labs)), labs]))
                )

        best_temp = 1.0
        best_loss = _nll(1.0)
        for t in np.linspace(0.1, 5.0, 100):
            loss = _nll(float(t))
            if loss < best_loss:
                best_loss = loss
                best_temp = float(t)

        # Refine
        for t in np.linspace(
            max(0.05, best_temp - 0.3), best_temp + 0.3, 100
        ):
            loss = _nll(float(t))
            if loss < best_loss:
                best_loss = loss
                best_temp = float(t)

        self._temperature = best_temp
        logger.info(
            "Calibration complete: temperature={:.4f}, NLL={:.4f}",
            best_temp,
            best_loss,
        )
        return best_temp


# ---------------------------------------------------------------------------
# SnapshotEnsemble
# ---------------------------------------------------------------------------


class SnapshotEnsemble:
    """Snapshot Ensemble: ensemble from a single training run.

    Collects model snapshots at cosine learning rate cycle restarts and
    aggregates their predictions for uncertainty estimation.

    Args:
        base_model: Model being trained (snapshots are captured from it).
        device: Inference device.
        batch_size: Batch size for inference.
        max_snapshots: Maximum number of snapshots to retain.
    """

    def __init__(
        self,
        base_model: nn.Module,
        device: str = "cpu",
        batch_size: int = 32,
        max_snapshots: int = 10,
    ) -> None:
        self.base_model = base_model
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_snapshots = max_snapshots
        self._snapshots: List[Dict[str, Tensor]] = []
        self._temperature: float = 1.0

        logger.info(
            "SnapshotEnsemble initialised (max_snapshots={})", max_snapshots
        )

    @property
    def n_snapshots(self) -> int:
        """Number of collected snapshots."""
        return len(self._snapshots)

    def collect_snapshot(self) -> int:
        """Capture a snapshot of the current model state.

        Should be called at each cosine LR cycle restart during training.

        Returns:
            Total number of snapshots after collection.
        """
        state = copy.deepcopy(self.base_model.state_dict())
        self._snapshots.append(state)

        # Enforce max snapshots (keep most recent)
        if len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(0)

        logger.info(
            "Collected snapshot {} (total: {})",
            len(self._snapshots),
            len(self._snapshots),
        )
        return len(self._snapshots)

    def clear_snapshots(self) -> None:
        """Remove all collected snapshots."""
        self._snapshots.clear()
        logger.info("Cleared all snapshots.")

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        input_ids: Tensor,
        return_all_predictions: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Run snapshot ensemble inference.

        Loads each snapshot into the base model, runs inference, and
        aggregates predictions.

        Args:
            input_ids: ``(N, L)`` input tensor.
            return_all_predictions: If ``True``, include per-snapshot
                predictions.

        Returns:
            Dictionary with ``"mean"``, ``"variance"``, ``"std"``,
            ``"entropy"``, and optionally ``"all_predictions"``.
        """
        if not self._snapshots:
            raise RuntimeError(
                "No snapshots collected. Call collect_snapshot() first."
            )

        original_state = copy.deepcopy(self.base_model.state_dict())
        all_outputs: List[np.ndarray] = []

        try:
            for i, state in enumerate(self._snapshots):
                self.base_model.load_state_dict(state)
                probs = _extract_probs(
                    self.base_model,
                    input_ids,
                    self.batch_size,
                    self.device,
                    self._temperature,
                )
                all_outputs.append(probs)
        finally:
            # Restore original state
            self.base_model.load_state_dict(original_state)

        stacked = np.stack(all_outputs, axis=0)
        mean_pred = stacked.mean(axis=0)
        variance = stacked.var(axis=0)
        std = np.sqrt(variance)
        entropy = _predictive_entropy(mean_pred)

        result: Dict[str, np.ndarray] = {
            "mean": mean_pred,
            "variance": variance,
            "std": std,
            "entropy": entropy,
        }

        if return_all_predictions:
            result["all_predictions"] = stacked

        logger.debug(
            "SnapshotEnsemble: {} snapshots, mean entropy={:.4f}",
            len(self._snapshots),
            float(entropy.mean()),
        )
        return result

    @staticmethod
    def cosine_annealing_schedule(
        epoch: int,
        total_epochs: int,
        n_cycles: int,
        initial_lr: float,
    ) -> float:
        """Compute learning rate for cosine annealing with warm restarts.

        Utility function for training with snapshot collection. Snapshots
        should be collected when the LR reaches its minimum (cycle restart).

        Args:
            epoch: Current epoch (0-indexed).
            total_epochs: Total number of training epochs.
            n_cycles: Number of cosine cycles.
            initial_lr: Initial (maximum) learning rate.

        Returns:
            Learning rate for the current epoch.
        """
        import math

        cycle_length = total_epochs / n_cycles
        cycle_pos = (epoch % cycle_length) / cycle_length
        return initial_lr * (1 + math.cos(math.pi * cycle_pos)) / 2
