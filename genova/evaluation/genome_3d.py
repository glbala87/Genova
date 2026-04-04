"""3D genome structure prediction for Genova.

Predicts higher-order chromatin organisation from DNA sequence:
- TAD (Topologically Associating Domain) boundaries
- Hi-C-style contact maps
- A/B compartment assignments
- Insulator / boundary element detection

Example::

    from genova.evaluation.genome_3d import Genome3DPredictor

    predictor = Genome3DPredictor(encoder, d_model=768)
    boundaries = predictor.predict_tad_boundaries(input_ids)
    contact_map = predictor.predict_contact_map(input_ids, resolution=10000)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TADBoundaryPrediction:
    """TAD boundary prediction results.

    Attributes:
        scores: Per-position boundary scores in ``[0, 1]``.
        boundaries: List of positions called as boundaries.
        resolution: Resolution in base pairs.
    """

    scores: np.ndarray
    boundaries: List[int]
    resolution: int = 1


@dataclass
class ContactMapPrediction:
    """Hi-C-style contact map prediction.

    Attributes:
        matrix: Symmetric contact probability matrix of shape ``(N, N)``.
        resolution: Resolution in base pairs per bin.
        num_bins: Number of bins along each axis.
    """

    matrix: np.ndarray
    resolution: int = 10000
    num_bins: int = 0

    def __post_init__(self) -> None:
        if self.num_bins == 0 and self.matrix.size > 0:
            self.num_bins = self.matrix.shape[0]


@dataclass
class CompartmentPrediction:
    """A/B compartment prediction.

    Attributes:
        labels: Per-bin compartment labels (``"A"`` or ``"B"``).
        scores: Raw scores (positive = A, negative = B).
        resolution: Resolution in base pairs.
    """

    labels: List[str]
    scores: np.ndarray
    resolution: int = 100000


# ---------------------------------------------------------------------------
# Helper layers
# ---------------------------------------------------------------------------


class _PairwiseScoring(nn.Module):
    """Compute pairwise interaction scores from per-position embeddings.

    Given embeddings ``(B, N, D)`` produces a symmetric score matrix
    ``(B, N, N)`` via outer-product + learned projection.

    Args:
        d_model: Input embedding dimension.
        d_pair: Hidden dimension for pairwise features.
    """

    def __init__(self, d_model: int, d_pair: int = 128) -> None:
        super().__init__()
        self.proj_row = nn.Linear(d_model, d_pair)
        self.proj_col = nn.Linear(d_model, d_pair)
        self.out_proj = nn.Sequential(
            nn.Linear(d_pair, d_pair // 2),
            nn.GELU(),
            nn.Linear(d_pair // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute pairwise scores.

        Args:
            x: ``(B, N, D)``

        Returns:
            ``(B, N, N)`` symmetric score matrix.
        """
        row = self.proj_row(x)  # (B, N, d_pair)
        col = self.proj_col(x)  # (B, N, d_pair)

        # Outer combination: (B, N, 1, d_pair) + (B, 1, N, d_pair)
        pair = row.unsqueeze(2) + col.unsqueeze(1)  # (B, N, N, d_pair)
        scores = self.out_proj(pair).squeeze(-1)  # (B, N, N)

        # Symmetrise
        scores = (scores + scores.transpose(-1, -2)) / 2.0
        return scores


class _BinPool(nn.Module):
    """Pool per-position embeddings into bins of a given resolution.

    Args:
        bin_size: Number of positions per bin.
    """

    def __init__(self, bin_size: int) -> None:
        super().__init__()
        self.bin_size = bin_size

    def forward(self, x: Tensor) -> Tensor:
        """Pool positions into bins.

        Args:
            x: ``(B, L, D)``

        Returns:
            ``(B, num_bins, D)``
        """
        B, L, D = x.shape
        n_bins = max(1, (L + self.bin_size - 1) // self.bin_size)
        pad_len = n_bins * self.bin_size - L
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        return x.reshape(B, n_bins, self.bin_size, D).mean(dim=2)


# ---------------------------------------------------------------------------
# Genome3DPredictor
# ---------------------------------------------------------------------------


class Genome3DPredictor(nn.Module):
    """Predict 3D genome organisation from DNA sequence.

    Uses encoder embeddings with task-specific heads for TAD boundary
    detection, contact map prediction, compartment calling, and
    insulator element detection.

    Args:
        encoder: Pretrained Genova encoder returning a dict with
            ``"last_hidden_state"`` of shape ``(B, L, D)``.
        d_model: Hidden dimension of encoder output.
        d_pair: Hidden dimension for pairwise scoring (contact maps).
        dropout: Dropout probability for heads.
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_model: int = 768,
        d_pair: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model

        # TAD boundary head: per-position binary classification
        self.tad_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Contact map head: pairwise scoring
        self.contact_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(dropout),
        )
        self.pairwise = _PairwiseScoring(d_model, d_pair)

        # Compartment head: per-bin binary classification (A vs B)
        self.compartment_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Insulator / boundary element head: per-position score
        self.insulator_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _get_hidden(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run encoder and return hidden states ``(B, L, D)``."""
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(enc_out, dict):
            hidden = enc_out.get("last_hidden_state", enc_out.get("hidden_states"))
        elif isinstance(enc_out, Tensor):
            hidden = enc_out
        else:
            hidden = getattr(enc_out, "last_hidden_state", enc_out)

        if isinstance(hidden, (list, tuple)):
            hidden = hidden[-1]
        return hidden

    # ------------------------------------------------------------------
    # TAD boundaries
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_tad_boundaries(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        threshold: float = 0.5,
    ) -> TADBoundaryPrediction:
        """Predict TAD boundary positions.

        Args:
            input_ids: ``(B, L)`` or ``(L,)``.  Only the first sample
                in the batch is returned.
            attention_mask: Optional mask.
            threshold: Probability threshold for calling a boundary.

        Returns:
            :class:`TADBoundaryPrediction` with scores and boundary
            positions.
        """
        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        hidden = self._get_hidden(input_ids, attention_mask)
        logits = self.tad_head(hidden)[0].squeeze(-1)  # (L,)
        scores = torch.sigmoid(logits).cpu().numpy()
        boundaries = [int(i) for i in np.where(scores >= threshold)[0]]

        return TADBoundaryPrediction(scores=scores, boundaries=boundaries)

    # ------------------------------------------------------------------
    # Contact map
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_contact_map(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        resolution: int = 10000,
    ) -> ContactMapPrediction:
        """Predict a Hi-C-style contact probability matrix.

        Positions are pooled into bins at the given resolution before
        computing pairwise scores.

        Args:
            input_ids: ``(B, L)`` or ``(L,)``.
            attention_mask: Optional mask.
            resolution: Bin size for the contact map.

        Returns:
            :class:`ContactMapPrediction` with the ``(N, N)`` matrix.
        """
        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        hidden = self._get_hidden(input_ids, attention_mask)
        features = self.contact_transform(hidden)

        # Pool into bins
        pool = _BinPool(resolution)
        binned = pool(features)  # (B, num_bins, D)

        scores = self.pairwise(binned)  # (B, num_bins, num_bins)
        probs = torch.sigmoid(scores[0]).cpu().numpy()

        return ContactMapPrediction(matrix=probs, resolution=resolution)

    # ------------------------------------------------------------------
    # Compartments
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_compartments(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        resolution: int = 100000,
    ) -> CompartmentPrediction:
        """Predict A/B compartment labels.

        Positions are binned at the given resolution.  Positive scores
        correspond to A compartment; negative to B.

        Args:
            input_ids: ``(B, L)`` or ``(L,)``.
            attention_mask: Optional mask.
            resolution: Bin size for compartment calling.

        Returns:
            :class:`CompartmentPrediction` with labels and scores.
        """
        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        hidden = self._get_hidden(input_ids, attention_mask)

        pool = _BinPool(resolution)
        binned = pool(hidden)  # (B, num_bins, D)

        logits = self.compartment_head(binned)[0].squeeze(-1)  # (num_bins,)
        scores = logits.cpu().numpy()
        labels = ["A" if s > 0 else "B" for s in scores]

        return CompartmentPrediction(
            labels=labels, scores=scores, resolution=resolution
        )

    # ------------------------------------------------------------------
    # Insulator detection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_insulators(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, List[int]]:
        """Detect insulator / boundary elements.

        Args:
            input_ids: ``(B, L)`` or ``(L,)``.
            attention_mask: Optional mask.
            threshold: Probability threshold for calling an insulator.

        Returns:
            Tuple of (per-position scores, list of insulator positions).
        """
        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        hidden = self._get_hidden(input_ids, attention_mask)
        logits = self.insulator_head(hidden)[0].squeeze(-1)  # (L,)
        scores = torch.sigmoid(logits).cpu().numpy()
        positions = [int(i) for i in np.where(scores >= threshold)[0]]

        return scores, positions

    # ------------------------------------------------------------------
    # Training losses
    # ------------------------------------------------------------------

    def compute_tad_loss(
        self,
        input_ids: Tensor,
        labels: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute TAD boundary prediction loss.

        Args:
            input_ids: ``(B, L)``
            labels: Binary boundary labels ``(B, L)``.
            attention_mask: ``(B, L)``

        Returns:
            Scalar loss.
        """
        hidden = self._get_hidden(input_ids, attention_mask)
        logits = self.tad_head(hidden).squeeze(-1)  # (B, L)
        return F.binary_cross_entropy_with_logits(logits, labels.float())

    def compute_contact_loss(
        self,
        input_ids: Tensor,
        contact_labels: Tensor,
        attention_mask: Optional[Tensor] = None,
        resolution: int = 10000,
    ) -> Tensor:
        """Compute contact map prediction loss.

        Args:
            input_ids: ``(B, L)``
            contact_labels: Binary contact labels ``(B, N, N)`` where
                ``N = L // resolution``.
            attention_mask: ``(B, L)``
            resolution: Bin size.

        Returns:
            Scalar loss.
        """
        hidden = self._get_hidden(input_ids, attention_mask)
        features = self.contact_transform(hidden)
        pool = _BinPool(resolution)
        binned = pool(features)
        scores = self.pairwise(binned)

        # Truncate to match label size
        n = min(scores.size(1), contact_labels.size(1))
        scores = scores[:, :n, :n]
        contact_labels = contact_labels[:, :n, :n]

        return F.binary_cross_entropy_with_logits(scores, contact_labels.float())

    def compute_compartment_loss(
        self,
        input_ids: Tensor,
        labels: Tensor,
        attention_mask: Optional[Tensor] = None,
        resolution: int = 100000,
    ) -> Tensor:
        """Compute compartment prediction loss.

        Args:
            input_ids: ``(B, L)``
            labels: Binary labels ``(B, N)`` where 1=A, 0=B and
                ``N = L // resolution``.
            attention_mask: ``(B, L)``
            resolution: Bin size.

        Returns:
            Scalar loss.
        """
        hidden = self._get_hidden(input_ids, attention_mask)
        pool = _BinPool(resolution)
        binned = pool(hidden)
        logits = self.compartment_head(binned).squeeze(-1)

        n = min(logits.size(1), labels.size(1))
        return F.binary_cross_entropy_with_logits(
            logits[:, :n], labels[:, :n].float()
        )
