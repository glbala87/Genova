"""Bayesian neural network layers for uncertainty quantification.

Provides a Bayesian linear layer using the reparameterization trick
(Blundell et al., 2015 -- "Weight Uncertainty in Neural Networks") and
a wrapper to convert standard models to approximate Bayesian variants.

Example::

    from genova.uncertainty.bayesian import BayesianLinear, BayesianWrapper

    layer = BayesianLinear(256, 64)
    output = layer(x)  # Sampled weights each forward pass
    kl = layer.kl_divergence()

    # Wrap an existing model
    wrapper = BayesianWrapper(model, target_modules=["classifier"])
    output, kl_loss = wrapper(x)
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from loguru import logger


# ---------------------------------------------------------------------------
# Bayesian Linear layer
# ---------------------------------------------------------------------------


class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty.

    Learns a posterior distribution ``q(w) = N(mu, sigma^2)`` over weights
    using the local reparameterization trick.  The prior is a standard
    normal ``N(0, prior_sigma^2)``.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``True``, adds a learnable bias with its own posterior.
        prior_sigma: Standard deviation of the prior distribution.
        init_rho: Initial value for the rho parameter (``sigma =
            log(1 + exp(rho))``).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prior_sigma: float = 1.0,
        init_rho: float = -3.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.prior_sigma = prior_sigma

        # Weight posterior parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_rho = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        # Prior parameters (fixed)
        self.register_buffer(
            "prior_weight_mu", torch.zeros(out_features, in_features)
        )
        self.register_buffer(
            "prior_weight_sigma",
            torch.full((out_features, in_features), prior_sigma),
        )

        self._init_rho = init_rho
        self.reset_parameters()

        # Store last sampled log-probabilities for KL computation
        self._log_prior: float = 0.0
        self._log_posterior: float = 0.0

    def reset_parameters(self) -> None:
        """Initialise posterior parameters."""
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, self._init_rho)
        if self.use_bias:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, self._init_rho)

    @staticmethod
    def _softplus(x: Tensor) -> Tensor:
        """Numerically stable softplus: ``log(1 + exp(x))``."""
        return F.softplus(x)

    def _sample_weight(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Sample weights from the variational posterior using the
        reparameterization trick.

        Returns:
            Tuple of ``(sampled_weight, sampled_bias)``.
        """
        weight_sigma = self._softplus(self.weight_rho)
        weight_epsilon = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_sigma * weight_epsilon

        bias: Optional[Tensor] = None
        if self.use_bias:
            bias_sigma = self._softplus(self.bias_rho)
            bias_epsilon = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_sigma * bias_epsilon

        return weight, bias

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with sampled weights.

        Args:
            x: Input tensor of shape ``(*, in_features)``.

        Returns:
            Output tensor of shape ``(*, out_features)``.
        """
        weight, bias = self._sample_weight()
        return F.linear(x, weight, bias)

    def kl_divergence(self) -> Tensor:
        """Compute KL divergence ``KL(q(w) || p(w))`` between the
        variational posterior and the prior.

        Returns:
            Scalar KL divergence tensor.
        """
        weight_sigma = self._softplus(self.weight_rho)
        prior_sigma = self.prior_weight_sigma

        # KL for weights: KL(N(mu_q, sigma_q^2) || N(0, sigma_p^2))
        kl = (
            torch.log(prior_sigma / weight_sigma)
            + (weight_sigma ** 2 + self.weight_mu ** 2) / (2 * prior_sigma ** 2)
            - 0.5
        ).sum()

        if self.use_bias:
            bias_sigma = self._softplus(self.bias_rho)
            kl += (
                torch.log(torch.tensor(self.prior_sigma) / bias_sigma)
                + (bias_sigma ** 2 + self.bias_mu ** 2)
                / (2 * self.prior_sigma ** 2)
                - 0.5
            ).sum()

        return kl

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.use_bias}, prior_sigma={self.prior_sigma}"
        )


# ---------------------------------------------------------------------------
# BayesianWrapper
# ---------------------------------------------------------------------------


class BayesianWrapper(nn.Module):
    """Convert a standard model to an approximate Bayesian model.

    Replaces specified ``nn.Linear`` layers with :class:`BayesianLinear`
    layers, enabling weight uncertainty and KL divergence computation.

    Args:
        model: Base PyTorch model to wrap.
        target_modules: List of module name substrings to replace.
            If ``None``, replaces all ``nn.Linear`` layers.
        prior_sigma: Prior standard deviation for Bayesian layers.
        freeze_non_bayesian: If ``True``, freeze parameters not in
            target modules.
    """

    def __init__(
        self,
        model: nn.Module,
        target_modules: Optional[List[str]] = None,
        prior_sigma: float = 1.0,
        freeze_non_bayesian: bool = False,
    ) -> None:
        super().__init__()
        self.model = copy.deepcopy(model)
        self._bayesian_layers: List[BayesianLinear] = []
        self._prior_sigma = prior_sigma

        replaced = self._replace_layers(self.model, target_modules or [])
        logger.info("BayesianWrapper: replaced {} Linear layers", replaced)

        if freeze_non_bayesian:
            self._freeze_non_bayesian()

    def _replace_layers(
        self,
        module: nn.Module,
        target_modules: List[str],
        prefix: str = "",
    ) -> int:
        """Recursively replace Linear layers with BayesianLinear.

        Args:
            module: Current module.
            target_modules: Name substrings to match.  Empty list = all.
            prefix: Current module path prefix.

        Returns:
            Number of layers replaced.
        """
        count = 0
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                should_replace = (
                    not target_modules
                    or any(t in full_name for t in target_modules)
                )
                if should_replace:
                    bayesian_layer = BayesianLinear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        prior_sigma=self._prior_sigma,
                    )
                    # Copy pretrained mean weights
                    bayesian_layer.weight_mu.data.copy_(child.weight.data)
                    if child.bias is not None and bayesian_layer.use_bias:
                        bayesian_layer.bias_mu.data.copy_(child.bias.data)

                    setattr(module, name, bayesian_layer)
                    self._bayesian_layers.append(bayesian_layer)
                    count += 1
            else:
                count += self._replace_layers(child, target_modules, full_name)
        return count

    def _freeze_non_bayesian(self) -> None:
        """Freeze all parameters except those in Bayesian layers."""
        bayesian_params = set()
        for bl in self._bayesian_layers:
            for p in bl.parameters():
                bayesian_params.add(id(p))

        for param in self.model.parameters():
            if id(param) not in bayesian_params:
                param.requires_grad = False

    def forward(self, *args: Any, **kwargs: Any) -> Tuple[Any, Tensor]:
        """Forward pass with KL divergence.

        Returns:
            Tuple of ``(model_output, kl_divergence)``.
        """
        output = self.model(*args, **kwargs)
        kl = self.total_kl_divergence()
        return output, kl

    def total_kl_divergence(self) -> Tensor:
        """Sum KL divergence across all Bayesian layers.

        Returns:
            Scalar tensor.
        """
        kl = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for layer in self._bayesian_layers:
            kl = kl + layer.kl_divergence()
        return kl

    def elbo_loss(
        self,
        nll: Tensor,
        n_samples: int,
        kl_weight: float = 1.0,
    ) -> Tensor:
        """Compute the Evidence Lower Bound (ELBO) loss.

        ``ELBO = NLL + (kl_weight / n_samples) * KL``

        Args:
            nll: Negative log-likelihood from the task loss.
            n_samples: Total number of training samples (for scaling KL).
            kl_weight: Weighting factor for the KL term.

        Returns:
            ELBO loss tensor.
        """
        kl = self.total_kl_divergence()
        return nll + (kl_weight / n_samples) * kl

    @torch.no_grad()
    def posterior_predictive(
        self,
        input_ids: Tensor,
        n_samples: int = 30,
    ) -> Dict[str, np.ndarray]:
        """Compute posterior predictive distribution via multiple
        forward passes.

        Args:
            input_ids: Input tensor.
            n_samples: Number of weight samples.

        Returns:
            Dictionary with ``"mean"``, ``"variance"``, ``"std"``,
            and ``"entropy"`` arrays.
        """
        self.model.eval()
        all_outputs: List[np.ndarray] = []

        for _ in range(n_samples):
            output = self.model(input_ids)
            if isinstance(output, dict):
                logits = output.get("logits", output.get("output"))
            elif isinstance(output, (tuple, list)):
                logits = output[0]
            else:
                logits = output

            if logits.dim() >= 2 and logits.size(-1) > 1:
                probs = F.softmax(logits, dim=-1)
            else:
                probs = torch.sigmoid(logits)

            all_outputs.append(probs.cpu().numpy())

        stacked = np.stack(all_outputs, axis=0)
        mean_pred = stacked.mean(axis=0)
        variance = stacked.var(axis=0)

        # Predictive entropy
        probs_clipped = np.clip(mean_pred, 1e-10, 1.0)
        if probs_clipped.ndim == 1 or (
            probs_clipped.ndim == 2 and probs_clipped.shape[-1] == 1
        ):
            p = probs_clipped.ravel()
            entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p + 1e-10))
        else:
            entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=-1)

        return {
            "mean": mean_pred,
            "variance": variance,
            "std": np.sqrt(variance),
            "entropy": entropy,
        }
