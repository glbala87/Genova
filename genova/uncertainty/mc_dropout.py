"""Monte Carlo Dropout for uncertainty estimation in genomic models.

Enables dropout at inference time to obtain a distribution of predictions,
yielding uncertainty estimates (variance, entropy) alongside mean predictions.

Example::

    from genova.uncertainty.mc_dropout import MCDropoutPredictor

    predictor = MCDropoutPredictor(model, n_forward_passes=50)
    mean, variance, entropy = predictor.predict_with_uncertainty(input_ids)
"""

from __future__ import annotations

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


def _enable_dropout(model: nn.Module) -> int:
    """Set all Dropout layers to training mode (active) while keeping
    other layers in eval mode.

    Args:
        model: PyTorch model.

    Returns:
        Number of dropout layers activated.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()
            count += 1
    return count


def _disable_dropout(model: nn.Module) -> None:
    """Set all Dropout layers back to eval mode.

    Args:
        model: PyTorch model.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.eval()


def _predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute predictive entropy from probability arrays.

    Args:
        probs: Array of shape ``(N,)`` or ``(N, C)`` with predicted
            probabilities.

    Returns:
        Entropy values of shape ``(N,)``.
    """
    probs = np.clip(probs, 1e-10, 1.0)
    if probs.ndim == 1:
        # Binary: convert to two-class
        probs_2d = np.stack([1 - probs, probs], axis=-1)
    else:
        probs_2d = probs
    return -np.sum(probs_2d * np.log(probs_2d), axis=-1)


# ---------------------------------------------------------------------------
# MCDropoutPredictor
# ---------------------------------------------------------------------------


class MCDropoutPredictor:
    """Monte Carlo Dropout uncertainty estimation.

    Runs multiple stochastic forward passes with dropout active to
    approximate Bayesian inference.  Returns mean predictions, variance,
    and predictive entropy.

    Args:
        model: PyTorch model with dropout layers.
        n_forward_passes: Number of stochastic forward passes.
        device: Torch device.
        batch_size: Inference batch size.
    """

    def __init__(
        self,
        model: nn.Module,
        n_forward_passes: int = 30,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.n_forward_passes = n_forward_passes
        self.device = torch.device(device)
        self.batch_size = batch_size

        # Calibration parameters
        self._temperature: float = 1.0

        # Validate model has dropout
        n_dropout = sum(
            1
            for m in model.modules()
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
        )
        if n_dropout == 0:
            logger.warning(
                "Model has no Dropout layers. MC Dropout will produce "
                "deterministic outputs."
            )
        else:
            logger.info(
                "MCDropoutPredictor initialised with {} dropout layers, "
                "{} forward passes",
                n_dropout,
                n_forward_passes,
            )

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        input_ids: Tensor,
        return_all_passes: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Run MC Dropout inference and return uncertainty estimates.

        Args:
            input_ids: Input tensor of shape ``(N, seq_len)``.
            return_all_passes: If ``True``, include all individual forward
                pass predictions in the result.

        Returns:
            Dictionary with keys:

            - ``"mean"``: Mean prediction, shape ``(N,)`` or ``(N, C)``.
            - ``"variance"``: Predictive variance, same shape.
            - ``"entropy"``: Predictive entropy, shape ``(N,)``.
            - ``"std"``: Standard deviation, same shape as variance.
            - ``"all_passes"`` (optional): ``(T, N, ...)`` array of all
              forward-pass outputs.
        """
        self.model.to(self.device)
        self.model.eval()
        n_dropout = _enable_dropout(self.model)

        all_outputs: List[np.ndarray] = []

        for t in range(self.n_forward_passes):
            batch_outputs: List[np.ndarray] = []

            for start in range(0, len(input_ids), self.batch_size):
                batch = input_ids[start : start + self.batch_size].to(self.device)
                output = self.model(batch)

                if isinstance(output, dict):
                    logits = output.get("logits", output.get("output"))
                elif isinstance(output, (tuple, list)):
                    logits = output[0]
                else:
                    logits = output

                # Apply temperature scaling
                if self._temperature != 1.0:
                    logits = logits / self._temperature

                # Convert to probabilities
                if logits.dim() >= 2 and logits.size(-1) > 1:
                    probs = F.softmax(logits, dim=-1)
                else:
                    probs = torch.sigmoid(logits)

                batch_outputs.append(probs.cpu().numpy())

            pass_output = np.concatenate(batch_outputs, axis=0)
            all_outputs.append(pass_output)

        _disable_dropout(self.model)

        # Stack: shape (T, N, ...) where T = n_forward_passes
        stacked = np.stack(all_outputs, axis=0)

        # Compute statistics
        mean_pred = stacked.mean(axis=0)
        variance = stacked.var(axis=0)
        std = np.sqrt(variance)

        # Predictive entropy from mean predictions
        entropy = _predictive_entropy(mean_pred)

        result: Dict[str, np.ndarray] = {
            "mean": mean_pred,
            "variance": variance,
            "std": std,
            "entropy": entropy,
        }

        if return_all_passes:
            result["all_passes"] = stacked

        logger.debug(
            "MC Dropout: {} passes, mean entropy={:.4f}",
            self.n_forward_passes,
            float(entropy.mean()),
        )
        return result

    def calibrate(
        self,
        val_input_ids: Tensor,
        val_labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Learn a temperature parameter to calibrate predictions.

        Uses the NLL on validation data to optimise temperature scaling
        via gradient-free line search.

        Args:
            val_input_ids: Validation input tensor.
            val_labels: Ground truth labels, shape ``(N,)``.
            lr: Learning rate for temperature optimisation.
            max_iter: Maximum optimisation iterations.

        Returns:
            Optimised temperature value.
        """
        logger.info("Calibrating MC Dropout with temperature scaling...")

        # Get uncalibrated mean predictions
        self._temperature = 1.0
        result = self.predict_with_uncertainty(val_input_ids)
        mean_probs = result["mean"]

        val_labels = np.asarray(val_labels).ravel()

        # Binary case
        if mean_probs.ndim == 1 or (mean_probs.ndim == 2 and mean_probs.shape[1] == 1):
            probs = mean_probs.ravel()

            def _nll(temp: float) -> float:
                # Re-scale logits
                logits = np.log(np.clip(probs, 1e-10, 1.0) / np.clip(1 - probs, 1e-10, 1.0))
                scaled_logits = logits / temp
                scaled_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
                scaled_probs = np.clip(scaled_probs, 1e-10, 1 - 1e-10)
                loss = -np.mean(
                    val_labels * np.log(scaled_probs)
                    + (1 - val_labels) * np.log(1 - scaled_probs)
                )
                return loss
        else:
            # Multi-class case
            def _nll(temp: float) -> float:
                logits = np.log(np.clip(mean_probs, 1e-10, 1.0))
                scaled = logits / temp
                # Softmax
                exp_s = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
                scaled_probs = exp_s / exp_s.sum(axis=-1, keepdims=True)
                scaled_probs = np.clip(scaled_probs, 1e-10, 1.0)
                labels_int = val_labels.astype(int)
                loss = -np.mean(
                    np.log(scaled_probs[np.arange(len(labels_int)), labels_int])
                )
                return loss

        # Grid + refinement search
        best_temp = 1.0
        best_loss = _nll(1.0)

        for temp_candidate in np.linspace(0.1, 5.0, 50):
            loss = _nll(temp_candidate)
            if loss < best_loss:
                best_loss = loss
                best_temp = float(temp_candidate)

        # Refine around best
        for temp_candidate in np.linspace(
            max(0.05, best_temp - 0.5), best_temp + 0.5, 100
        ):
            loss = _nll(temp_candidate)
            if loss < best_loss:
                best_loss = loss
                best_temp = float(temp_candidate)

        self._temperature = best_temp
        logger.info(
            "Calibration complete: temperature={:.4f}, NLL={:.4f}",
            best_temp,
            best_loss,
        )
        return best_temp

    @property
    def temperature(self) -> float:
        """Current temperature scaling parameter."""
        return self._temperature
