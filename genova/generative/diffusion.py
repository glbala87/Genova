"""Discrete diffusion for DNA sequence generation (D3PM-style).

Implements a Discrete Denoising Diffusion Probabilistic Model for generating
DNA sequences.  The forward process gradually corrupts nucleotide tokens by
transitioning them towards a uniform distribution, and the reverse process
learns to denoise corrupted sequences back to valid DNA.

Reference: Austin et al., "Structured Denoising Diffusion Models in Discrete
State-Spaces" (D3PM), NeurIPS 2021.

Example::

    from genova.utils.config import ModelConfig
    from genova.generative.diffusion import DiscreteDiffusion, DiffusionGenerator

    cfg = ModelConfig(d_model=256, n_heads=4, n_layers=6, d_ff=1024,
                      vocab_size=4096)
    model = DiscreteDiffusion(cfg, num_timesteps=1000, num_classes=5)
    generator = DiffusionGenerator(model, num_classes=5, device="cuda")
    samples = generator.generate(num_sequences=8, seq_length=256)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from genova.models.transformer import GenovaTransformer
from genova.models.mamba_model import GenovaMamba
from genova.utils.config import ModelConfig


# ---------------------------------------------------------------------------
# Noise schedule utilities
# ---------------------------------------------------------------------------


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    """Cosine noise schedule (Nichol & Dhariwal, 2021).

    Args:
        timesteps: Number of diffusion steps.
        s: Small offset to prevent beta from being too small near t=0.

    Returns:
        ``(timesteps,)`` tensor of beta values.
    """
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(0.0001, 0.9999).float()


def _linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> Tensor:
    """Linear noise schedule.

    Args:
        timesteps: Number of diffusion steps.
        beta_start: Starting beta.
        beta_end: Ending beta.

    Returns:
        ``(timesteps,)`` tensor of beta values.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# ---------------------------------------------------------------------------
# Discrete transition matrices
# ---------------------------------------------------------------------------


def _uniform_transition_matrix(beta_t: float, num_classes: int) -> Tensor:
    """Build a transition matrix for the uniform-noise forward process.

    With probability ``beta_t`` a token transitions to a uniformly random
    class; with probability ``1 - beta_t`` it stays the same.

    Args:
        beta_t: Noise level at timestep *t*.
        num_classes: Number of discrete token classes (e.g. 5 for ACGTN).

    Returns:
        ``(num_classes, num_classes)`` transition matrix Q_t.
    """
    mat = torch.full((num_classes, num_classes), beta_t / num_classes)
    mat += torch.eye(num_classes) * (1.0 - beta_t)
    return mat


# ---------------------------------------------------------------------------
# Denoising network
# ---------------------------------------------------------------------------


class DenoisingNetwork(nn.Module):
    """Transformer-based denoising network for the discrete diffusion model.

    Takes noisy token ids and a timestep embedding and predicts the clean
    token distribution at each position.

    Args:
        config: Model configuration.
        num_classes: Number of discrete classes (nucleotide tokens).
        num_timesteps: Maximum number of diffusion timesteps.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_classes: int = 5,
        num_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.d_model = config.d_model

        # Timestep embedding
        self.time_embedding = nn.Sequential(
            nn.Embedding(num_timesteps, config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # Token embedding (for noisy inputs)
        self.token_embedding = nn.Embedding(num_classes, config.d_model)

        # Positional encoding
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.d_model
        )

        # Backbone
        if config.arch.lower() in ("transformer", "bert"):
            self.backbone = GenovaTransformer(config)
        elif config.arch.lower() in ("mamba", "ssm"):
            self.backbone = GenovaMamba(config)
        else:
            raise ValueError(f"Unsupported arch '{config.arch}'")

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, num_classes),
        )

    def forward(
        self,
        noisy_ids: Tensor,
        timesteps: Tensor,
        attention_mask: Optional[Tensor] = None,
        condition_emb: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict clean token logits from noisy inputs.

        Args:
            noisy_ids: ``(B, L)`` noisy token ids.
            timesteps: ``(B,)`` integer timestep for each sample.
            attention_mask: ``(B, L)`` attention mask.
            condition_emb: ``(B, D)`` optional conditioning embedding.

        Returns:
            ``(B, L, num_classes)`` logits predicting clean tokens.
        """
        B, L = noisy_ids.shape

        # Embed tokens
        x = self.token_embedding(noisy_ids)  # (B, L, D)

        # Add positional encoding
        positions = torch.arange(L, device=noisy_ids.device).unsqueeze(0)
        x = x + self.position_embedding(positions)

        # Add timestep embedding
        t_emb = self.time_embedding(timesteps)  # (B, D)
        x = x + t_emb.unsqueeze(1)  # broadcast over L

        # Add optional condition
        if condition_emb is not None:
            x = x + condition_emb.unsqueeze(1)

        # Backbone
        out = self.backbone(noisy_ids, attention_mask=attention_mask)
        hidden = out["last_hidden_state"] + x  # residual from embeddings

        # Predict clean logits
        logits = self.output_head(hidden)  # (B, L, num_classes)
        return logits


# ---------------------------------------------------------------------------
# Discrete diffusion model
# ---------------------------------------------------------------------------


class DiscreteDiffusion(nn.Module):
    """Discrete Denoising Diffusion Probabilistic Model (D3PM) for DNA.

    Implements the full forward and reverse diffusion process for discrete
    token sequences.

    Args:
        config: Model configuration for the denoising backbone.
        num_timesteps: Number of diffusion steps.
        num_classes: Number of discrete token classes.
        schedule: Noise schedule type (``"cosine"`` or ``"linear"``).
        loss_type: ``"cross_entropy"`` or ``"kl_divergence"``.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_timesteps: int = 1000,
        num_classes: int = 5,
        schedule: str = "cosine",
        loss_type: str = "cross_entropy",
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.loss_type = loss_type

        # Denoising network
        self.denoise_net = DenoisingNetwork(
            config, num_classes=num_classes, num_timesteps=num_timesteps
        )

        # Noise schedule
        if schedule == "cosine":
            betas = _cosine_beta_schedule(num_timesteps)
        elif schedule == "linear":
            betas = _linear_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule '{schedule}'")

        self.register_buffer("betas", betas)

        # Pre-compute cumulative transition matrices
        # Q_t[i, j] = P(x_t = j | x_{t-1} = i)
        q_matrices = []
        q_bar = torch.eye(num_classes)  # cumulative product
        q_bar_list = [q_bar.clone()]

        for t in range(num_timesteps):
            q_t = _uniform_transition_matrix(betas[t].item(), num_classes)
            q_matrices.append(q_t)
            q_bar = q_bar @ q_t
            q_bar_list.append(q_bar.clone())

        self.register_buffer(
            "q_matrices",
            torch.stack(q_matrices),  # (T, C, C)
        )
        self.register_buffer(
            "q_bar",
            torch.stack(q_bar_list),  # (T+1, C, C)
        )

        logger.info(
            "DiscreteDiffusion: T={}, C={}, schedule={}, loss={}",
            num_timesteps,
            num_classes,
            schedule,
            loss_type,
        )

    # -- forward process (noise) --------------------------------------------

    def q_sample(
        self,
        x_0: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Sample from the forward process q(x_t | x_0).

        Args:
            x_0: ``(B, L)`` clean token ids.
            t: ``(B,)`` timesteps.

        Returns:
            ``(B, L)`` noisy token ids at timestep *t*.
        """
        B, L = x_0.shape

        # Get cumulative transition probs for each sample
        # q_bar[t] is (C, C): q_bar[t][i, j] = P(x_t = j | x_0 = i)
        log_probs = torch.zeros(B, L, self.num_classes, device=x_0.device)

        for b in range(B):
            # One-hot encode x_0
            x_0_onehot = F.one_hot(x_0[b], self.num_classes).float()  # (L, C)
            # Multiply by Q_bar to get marginal distribution
            probs = x_0_onehot @ self.q_bar[t[b]]  # (L, C)
            log_probs[b] = probs

        # Sample from the categorical distribution
        flat_probs = log_probs.reshape(-1, self.num_classes)
        sampled = torch.multinomial(flat_probs.clamp(min=1e-8), num_samples=1)
        return sampled.reshape(B, L)

    # -- training loss -------------------------------------------------------

    def forward(
        self,
        x_0: Tensor,
        attention_mask: Optional[Tensor] = None,
        condition_emb: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute the diffusion training loss.

        Args:
            x_0: ``(B, L)`` clean token ids.
            attention_mask: ``(B, L)`` attention mask.
            condition_emb: ``(B, D)`` optional conditioning embedding.

        Returns:
            Dict with ``"loss"`` (scalar) and ``"logits"`` (B, L, C).
        """
        B, L = x_0.shape
        device = x_0.device

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # Forward diffuse
        x_t = self.q_sample(x_0, t)

        # Predict clean distribution
        logits = self.denoise_net(
            x_t, t,
            attention_mask=attention_mask,
            condition_emb=condition_emb,
        )  # (B, L, C)

        # Loss
        if self.loss_type == "cross_entropy":
            loss = F.cross_entropy(
                logits.reshape(-1, self.num_classes),
                x_0.reshape(-1),
                reduction="none",
            ).reshape(B, L)

            if attention_mask is not None:
                loss = (loss * attention_mask.float()).sum() / attention_mask.float().sum().clamp(min=1.0)
            else:
                loss = loss.mean()

        elif self.loss_type == "kl_divergence":
            # KL divergence between predicted and true posterior
            pred_probs = F.softmax(logits, dim=-1)
            target_onehot = F.one_hot(x_0, self.num_classes).float()
            kl = (target_onehot * (target_onehot.clamp(min=1e-8).log() - pred_probs.clamp(min=1e-8).log())).sum(-1)

            if attention_mask is not None:
                loss = (kl * attention_mask.float()).sum() / attention_mask.float().sum().clamp(min=1.0)
            else:
                loss = kl.mean()
        else:
            raise ValueError(f"Unknown loss_type '{self.loss_type}'")

        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# Diffusion generator (reverse sampling)
# ---------------------------------------------------------------------------


class DiffusionGenerator:
    """Generate DNA sequences by running the reverse diffusion process.

    Args:
        model: Trained :class:`DiscreteDiffusion` model.
        num_classes: Number of discrete classes.
        device: Inference device.
    """

    def __init__(
        self,
        model: DiscreteDiffusion,
        num_classes: int = 5,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.model = model
        self.num_classes = num_classes
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        # Nucleotide mapping for decoding
        self._id_to_nuc = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}

        logger.info(
            "DiffusionGenerator ready (T={}, C={}, device={})",
            model.num_timesteps,
            num_classes,
            self.device,
        )

    @torch.no_grad()
    def generate(
        self,
        num_sequences: int = 1,
        seq_length: int = 256,
        temperature: float = 1.0,
        condition_emb: Optional[Tensor] = None,
        return_trajectory: bool = False,
    ) -> Dict[str, Any]:
        """Generate sequences via the reverse diffusion process.

        Starts from pure noise (uniform random tokens) and iteratively
        denoises for ``T`` steps.

        Args:
            num_sequences: Number of sequences to generate.
            seq_length: Length of each generated sequence.
            temperature: Sampling temperature for the reverse process.
            condition_emb: ``(B, D)`` conditioning embedding.
            return_trajectory: If ``True``, store the sequence at every
                timestep.

        Returns:
            Dict with:
                - ``"token_ids"``: ``(B, L)`` final generated token ids.
                - ``"sequences"``: list of decoded DNA strings.
                - ``"trajectory"``: list of ``(B, L)`` tensors (if requested).
        """
        B, L = num_sequences, seq_length
        T = self.model.num_timesteps

        # Start from pure noise (uniform random tokens)
        x_t = torch.randint(0, self.num_classes, (B, L), device=self.device)

        trajectory: List[Tensor] = []
        if return_trajectory:
            trajectory.append(x_t.clone())

        # Reverse process: t = T-1, T-2, ..., 0
        for t_val in reversed(range(T)):
            t = torch.full((B,), t_val, dtype=torch.long, device=self.device)

            # Predict clean distribution
            logits = self.model.denoise_net(
                x_t, t, condition_emb=condition_emb
            )  # (B, L, C)

            if t_val > 0:
                # Sample from predicted distribution with temperature
                probs = F.softmax(logits / temperature, dim=-1)
                flat_probs = probs.reshape(-1, self.num_classes)
                x_t = torch.multinomial(flat_probs, num_samples=1).reshape(B, L)
            else:
                # Final step: take argmax for deterministic output
                x_t = logits.argmax(dim=-1)

            if return_trajectory:
                trajectory.append(x_t.clone())

        sequences = self._decode_batch(x_t)

        result: Dict[str, Any] = {
            "token_ids": x_t,
            "sequences": sequences,
        }
        if return_trajectory:
            result["trajectory"] = trajectory

        return result

    @torch.no_grad()
    def conditional_generate(
        self,
        condition_emb: Tensor,
        num_sequences: int = 1,
        seq_length: int = 256,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Generate sequences conditioned on an embedding.

        Args:
            condition_emb: ``(B, D)`` or ``(1, D)`` conditioning embedding.
            num_sequences: Number of sequences to generate.
            seq_length: Length of each generated sequence.
            temperature: Sampling temperature.

        Returns:
            Same as :meth:`generate`.
        """
        cond = condition_emb.to(self.device)
        if cond.size(0) == 1 and num_sequences > 1:
            cond = cond.expand(num_sequences, -1)
        return self.generate(
            num_sequences=num_sequences,
            seq_length=seq_length,
            temperature=temperature,
            condition_emb=cond,
        )

    @torch.no_grad()
    def inpaint(
        self,
        partial_sequence: Tensor,
        mask: Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Inpaint masked positions in a partial DNA sequence.

        Runs the reverse process but fixes known positions at each step.

        Args:
            partial_sequence: ``(B, L)`` token ids with known positions.
            mask: ``(B, L)`` binary mask.  ``1`` = position to generate,
                ``0`` = known (fixed) position.
            temperature: Sampling temperature.

        Returns:
            Dict with ``"token_ids"`` and ``"sequences"``.
        """
        B, L = partial_sequence.shape
        T = self.model.num_timesteps

        partial_sequence = partial_sequence.to(self.device)
        mask = mask.to(self.device).bool()

        # Initialise: known positions from input, unknown from noise
        x_t = partial_sequence.clone()
        x_t[mask] = torch.randint(0, self.num_classes, (mask.sum().item(),), device=self.device)

        for t_val in reversed(range(T)):
            t = torch.full((B,), t_val, dtype=torch.long, device=self.device)

            logits = self.model.denoise_net(x_t, t)

            if t_val > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                flat_probs = probs.reshape(-1, self.num_classes)
                new_tokens = torch.multinomial(flat_probs, num_samples=1).reshape(B, L)
            else:
                new_tokens = logits.argmax(dim=-1)

            # Only update masked (unknown) positions
            x_t[mask] = new_tokens[mask]

        sequences = self._decode_batch(x_t)
        return {"token_ids": x_t, "sequences": sequences}

    def _decode_batch(self, token_ids: Tensor) -> List[str]:
        """Decode token id tensor to DNA strings."""
        sequences: List[str] = []
        for i in range(token_ids.size(0)):
            ids = token_ids[i].cpu().tolist()
            seq = "".join(self._id_to_nuc.get(t, "N") for t in ids)
            sequences.append(seq)
        return sequences
