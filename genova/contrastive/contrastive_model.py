"""Contrastive learning framework for genomic representations.

Implements SimCLR-style contrastive learning with an NT-Xent (InfoNCE) loss
on top of the Genova encoder backbone (transformer or mamba).

Example::

    from genova.utils.config import ModelConfig
    from genova.contrastive.contrastive_model import ContrastiveGenovaModel

    cfg = ModelConfig(d_model=256, n_heads=4, n_layers=6, d_ff=1024,
                      vocab_size=4096)
    model = ContrastiveGenovaModel(cfg, projection_dim=128, temperature=0.07)

    # view1, view2: (B, L) token ids
    loss = model(view1, view2)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from genova.utils.config import ModelConfig
from genova.models.transformer import GenovaTransformer
from genova.models.mamba_model import GenovaMamba


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------


class ProjectionHead(nn.Module):
    """Non-linear projection head mapping encoder outputs to the
    contrastive embedding space.

    Architecture: ``Linear -> BN -> ReLU -> Linear -> BN``.

    Args:
        input_dim: Dimensionality of encoder hidden states.
        hidden_dim: Width of the hidden layer.
        output_dim: Dimensionality of the contrastive embedding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project encoder pooled representation.

        Args:
            x: ``(B, D_in)`` tensor.

        Returns:
            ``(B, D_out)`` L2-normalised projection.
        """
        return F.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# NT-Xent (InfoNCE) loss
# ---------------------------------------------------------------------------


class NTXentLoss(nn.Module):
    """Normalised temperature-scaled cross-entropy loss (NT-Xent).

    Computes the InfoNCE / SimCLR contrastive loss using in-batch negatives.
    For a batch of *N* positive pairs, each sample has 1 positive and
    ``2N - 2`` negatives.

    Args:
        temperature: Scaling temperature (tau). Lower values make the
            distribution sharper.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """Compute NT-Xent loss for a batch of positive pairs.

        Args:
            z_i: ``(N, D)`` L2-normalised projections of view 1.
            z_j: ``(N, D)`` L2-normalised projections of view 2.

        Returns:
            Scalar loss tensor.
        """
        N = z_i.size(0)
        device = z_i.device

        # Concatenate both views: [z_i; z_j] => (2N, D)
        z = torch.cat([z_i, z_j], dim=0)  # (2N, D)

        # Full pairwise similarity matrix: (2N, 2N)
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # Mask out self-similarity on the diagonal
        mask_self = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim.masked_fill_(mask_self, float("-inf"))

        # Positive pairs: (i, i+N) and (i+N, i)
        # Labels: for row i in [0, N), positive is at column i+N
        #         for row i in [N, 2N), positive is at column i-N
        labels = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N, device=device),
        ])  # (2N,)

        loss = F.cross_entropy(sim, labels)
        return loss

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}"


# ---------------------------------------------------------------------------
# Contrastive model
# ---------------------------------------------------------------------------


class ContrastiveGenovaModel(nn.Module):
    """SimCLR-style contrastive wrapper around a Genova encoder backbone.

    Encodes two augmented views of each genomic sequence, pools the
    representations, projects them through a non-linear head, and computes
    the NT-Xent contrastive loss.

    Args:
        config: Model configuration (determines encoder architecture).
        projection_dim: Output dimensionality of the projection head.
        projection_hidden_dim: Hidden width of the projection head.
            Defaults to ``config.d_model``.
        temperature: Temperature for the NT-Xent loss.
        pooling: Pooling strategy over sequence positions.
            ``"mean"`` (default) or ``"cls"`` (first-token pooling).
        backbone: Which encoder architecture to use.
            ``None`` auto-selects from ``config.arch``.
    """

    def __init__(
        self,
        config: ModelConfig,
        projection_dim: int = 128,
        projection_hidden_dim: Optional[int] = None,
        temperature: float = 0.07,
        pooling: str = "mean",
        backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.pooling = pooling

        # Encoder backbone
        if backbone is not None:
            self.encoder = backbone
        elif config.arch.lower() in ("transformer", "bert"):
            self.encoder = GenovaTransformer(config)
        elif config.arch.lower() in ("mamba", "ssm"):
            self.encoder = GenovaMamba(config)
        else:
            raise ValueError(
                f"Unsupported arch '{config.arch}'. Use 'transformer' or 'mamba'."
            )

        # Projection head
        hidden_dim = projection_hidden_dim or config.d_model
        self.projection_head = ProjectionHead(
            input_dim=config.d_model,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )

        # Contrastive loss
        self.criterion = NTXentLoss(temperature=temperature)

    def pool(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Pool sequence-level representations into a single vector.

        Args:
            hidden_states: ``(B, L, D)`` encoder outputs.
            attention_mask: ``(B, L)`` mask (1 = real, 0 = pad).

        Returns:
            ``(B, D)`` pooled representation.
        """
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
                summed = (hidden_states * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                return summed / counts
            return hidden_states.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling '{self.pooling}'. Use 'mean' or 'cls'.")

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode token ids and return pooled, projected embeddings.

        Args:
            input_ids: ``(B, L)`` token ids.
            attention_mask: ``(B, L)`` attention mask.

        Returns:
            ``(B, projection_dim)`` L2-normalised embeddings.
        """
        enc_out = self.encoder(input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
        hidden = self._extract_hidden(enc_out)
        pooled = self.pool(hidden, attention_mask)
        return self.projection_head(pooled)

    @staticmethod
    def _extract_hidden(enc_out: Dict[str, Tensor]) -> Tensor:
        """Extract last hidden state from encoder output dict.

        Handles both backbone (``last_hidden_state``) and MLM wrapper
        (``hidden_states[-1]``) output formats.
        """
        if "last_hidden_state" in enc_out:
            return enc_out["last_hidden_state"]
        if "hidden_states" in enc_out:
            hs = enc_out["hidden_states"]
            return hs[-1] if isinstance(hs, (list, tuple)) else hs
        raise KeyError(
            f"Encoder output missing 'last_hidden_state' or 'hidden_states'. "
            f"Got keys: {list(enc_out.keys())}"
        )

    def get_embeddings(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return pooled encoder embeddings *without* the projection head.

        Useful for downstream evaluation and transfer learning.

        Args:
            input_ids: ``(B, L)`` token ids.
            attention_mask: ``(B, L)`` attention mask.

        Returns:
            ``(B, d_model)`` embeddings.
        """
        enc_out = self.encoder(input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
        hidden = self._extract_hidden(enc_out)
        return self.pool(hidden, attention_mask)

    def forward(
        self,
        view1_ids: Tensor,
        view2_ids: Tensor,
        view1_mask: Optional[Tensor] = None,
        view2_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass: encode both views and compute contrastive loss.

        Args:
            view1_ids: ``(B, L)`` token ids for augmented view 1.
            view2_ids: ``(B, L)`` token ids for augmented view 2.
            view1_mask: Optional attention mask for view 1.
            view2_mask: Optional attention mask for view 2.

        Returns:
            Dict with keys:
                - ``loss``: scalar NT-Xent contrastive loss.
                - ``z1``: ``(B, projection_dim)`` projections for view 1.
                - ``z2``: ``(B, projection_dim)`` projections for view 2.
        """
        z1 = self.encode(view1_ids, view1_mask)
        z2 = self.encode(view2_ids, view2_mask)
        loss = self.criterion(z1, z2)
        return {"loss": loss, "z1": z1, "z2": z2}
