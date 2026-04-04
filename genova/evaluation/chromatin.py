"""Chromatin state prediction for Genova.

Predicts chromatin accessibility (open/closed) and histone modification
marks from DNA sequence.  Supports per-position or binned predictions
at configurable resolution (128bp, 256bp).

Example::

    from genova.evaluation.chromatin import ChromatinStatePredictor

    predictor = ChromatinStatePredictor(encoder, num_marks=5, d_model=768)
    accessibility = predictor.predict_accessibility(input_ids)
    marks = predictor.predict_histone_marks(input_ids)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Standard histone marks predicted by default.
DEFAULT_HISTONE_MARKS: List[str] = [
    "H3K4me3",
    "H3K27ac",
    "H3K27me3",
    "H3K36me3",
    "H3K9me3",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ChromatinPrediction:
    """Container for chromatin state predictions at a locus.

    Attributes:
        accessibility: Per-bin accessibility scores in ``[0, 1]``.
        histone_marks: Dict mapping mark name to per-bin prediction array.
        bin_size: Resolution in base pairs.
        num_bins: Number of bins in the prediction.
    """

    accessibility: np.ndarray  # (num_bins,)
    histone_marks: Dict[str, np.ndarray]  # mark -> (num_bins,)
    bin_size: int = 128
    num_bins: int = 0

    def __post_init__(self) -> None:
        if self.num_bins == 0 and self.accessibility.size > 0:
            self.num_bins = len(self.accessibility)


# ---------------------------------------------------------------------------
# Helper layers
# ---------------------------------------------------------------------------


class _BinningLayer(nn.Module):
    """Reduce per-position features to per-bin features via pooling.

    Args:
        bin_size: Number of positions per bin.
        mode: ``"mean"`` or ``"max"`` pooling.
    """

    def __init__(self, bin_size: int = 128, mode: str = "mean") -> None:
        super().__init__()
        self.bin_size = bin_size
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        """Bin a per-position tensor.

        Args:
            x: Input of shape ``(B, L, D)``.

        Returns:
            Binned tensor of shape ``(B, L // bin_size, D)``.
            If ``L`` is not divisible by ``bin_size`` the last partial
            bin is included.
        """
        B, L, D = x.shape
        n_bins = (L + self.bin_size - 1) // self.bin_size

        # Pad to make L divisible by bin_size
        pad_len = n_bins * self.bin_size - L
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))  # pad length dim

        x = x.reshape(B, n_bins, self.bin_size, D)
        if self.mode == "max":
            return x.max(dim=2).values
        return x.mean(dim=2)


# ---------------------------------------------------------------------------
# ChromatinStatePredictor
# ---------------------------------------------------------------------------


class ChromatinStatePredictor(nn.Module):
    """Multi-task chromatin state predictor.

    Simultaneously predicts:
    - Chromatin accessibility (open/closed per bin)
    - Multiple histone modification marks per bin

    Uses encoder embeddings pooled into fixed-resolution bins.

    Args:
        encoder: Pretrained Genova encoder returning a dict with
            ``"last_hidden_state"`` of shape ``(B, L, D)``.
        num_marks: Number of histone marks to predict.
        d_model: Hidden dimension of encoder output.
        bin_size: Resolution in base pairs / positions for binning.
            Default 128.
        mark_names: Optional list of histone mark names.  If ``None``,
            uses :data:`DEFAULT_HISTONE_MARKS` (truncated or padded to
            ``num_marks``).
        dropout: Dropout probability for heads.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_marks: int = 5,
        d_model: int = 768,
        bin_size: int = 128,
        mark_names: Optional[List[str]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_marks = num_marks
        self.d_model = d_model
        self.bin_size = bin_size

        # Set mark names
        if mark_names is not None:
            if len(mark_names) != num_marks:
                raise ValueError(
                    f"Expected {num_marks} mark names, got {len(mark_names)}"
                )
            self._mark_names = list(mark_names)
        else:
            self._mark_names = DEFAULT_HISTONE_MARKS[:num_marks]
            # Pad with generic names if num_marks > len(DEFAULT_HISTONE_MARKS)
            while len(self._mark_names) < num_marks:
                self._mark_names.append(f"mark_{len(self._mark_names)}")

        # Binning layer
        self.binning = _BinningLayer(bin_size=bin_size, mode="mean")

        # Shared feature extractor after binning
        self.shared_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(dropout),
        )

        # Accessibility head (binary per bin)
        self.accessibility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Histone marks head (num_marks outputs per bin)
        self.histone_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_marks),
        )

    @property
    def mark_names(self) -> List[str]:
        """Names of the histone marks being predicted."""
        return self._mark_names

    def _encode(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run encoder and return binned, transformed features.

        Args:
            input_ids: ``(B, L)``
            attention_mask: ``(B, L)``

        Returns:
            ``(B, num_bins, D)``
        """
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(enc_out, dict):
            hidden = enc_out.get("last_hidden_state", enc_out.get("hidden_states"))
        elif isinstance(enc_out, Tensor):
            hidden = enc_out
        else:
            hidden = getattr(enc_out, "last_hidden_state", enc_out)

        if isinstance(hidden, (list, tuple)):
            hidden = hidden[-1]

        binned = self.binning(hidden)  # (B, num_bins, D)
        return self.shared_transform(binned)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Full forward pass returning accessibility and histone logits.

        Args:
            input_ids: ``(B, L)``
            attention_mask: ``(B, L)``

        Returns:
            Dict with keys:
            - ``"accessibility_logits"``: ``(B, num_bins, 1)``
            - ``"histone_logits"``: ``(B, num_bins, num_marks)``
        """
        features = self._encode(input_ids, attention_mask)
        return {
            "accessibility_logits": self.accessibility_head(features),
            "histone_logits": self.histone_head(features),
        }

    @torch.no_grad()
    def predict(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> ChromatinPrediction:
        """Predict chromatin state for a single sequence.

        Args:
            input_ids: ``(1, L)`` or ``(L,)``.
            attention_mask: Optional mask.

        Returns:
            :class:`ChromatinPrediction` with accessibility and histone
            mark predictions.
        """
        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        out = self.forward(input_ids, attention_mask)

        acc_probs = torch.sigmoid(out["accessibility_logits"][0]).squeeze(-1)  # (num_bins,)
        hist_probs = torch.sigmoid(out["histone_logits"][0])  # (num_bins, num_marks)

        acc_np = acc_probs.cpu().numpy()
        hist_np = hist_probs.cpu().numpy()

        mark_dict: Dict[str, np.ndarray] = {}
        for i, name in enumerate(self._mark_names):
            mark_dict[name] = hist_np[:, i]

        return ChromatinPrediction(
            accessibility=acc_np,
            histone_marks=mark_dict,
            bin_size=self.bin_size,
        )

    @torch.no_grad()
    def predict_accessibility(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict chromatin accessibility scores.

        Args:
            input_ids: ``(B, L)``
            attention_mask: ``(B, L)``

        Returns:
            Accessibility probabilities of shape ``(B, num_bins)``.
        """
        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        features = self._encode(input_ids, attention_mask)
        logits = self.accessibility_head(features).squeeze(-1)  # (B, num_bins)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_histone_marks(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Predict histone modification marks.

        Args:
            input_ids: ``(B, L)``
            attention_mask: ``(B, L)``

        Returns:
            Dict mapping mark name to probability tensor of shape
            ``(B, num_bins)``.
        """
        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        features = self._encode(input_ids, attention_mask)
        logits = self.histone_head(features)  # (B, num_bins, num_marks)
        probs = torch.sigmoid(logits)

        result: Dict[str, Tensor] = {}
        for i, name in enumerate(self._mark_names):
            result[name] = probs[:, :, i]

        return result

    def compute_loss(
        self,
        input_ids: Tensor,
        accessibility_labels: Optional[Tensor] = None,
        histone_labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        accessibility_weight: float = 1.0,
        histone_weight: float = 1.0,
    ) -> Dict[str, Tensor]:
        """Compute combined loss for accessibility and histone marks.

        Args:
            input_ids: ``(B, L)``
            accessibility_labels: Binary labels ``(B, num_bins)``.
            histone_labels: Binary labels ``(B, num_bins, num_marks)``.
            attention_mask: ``(B, L)``
            accessibility_weight: Loss weight for accessibility task.
            histone_weight: Loss weight for histone task.

        Returns:
            Dict with ``"total_loss"``, ``"accessibility_loss"``, and
            ``"histone_loss"`` keys.
        """
        out = self.forward(input_ids, attention_mask)
        losses: Dict[str, Tensor] = {}
        total = torch.tensor(0.0, device=input_ids.device)

        if accessibility_labels is not None:
            acc_logits = out["accessibility_logits"].squeeze(-1)  # (B, num_bins)
            acc_loss = F.binary_cross_entropy_with_logits(
                acc_logits, accessibility_labels.float()
            )
            losses["accessibility_loss"] = acc_loss
            total = total + accessibility_weight * acc_loss

        if histone_labels is not None:
            hist_logits = out["histone_logits"]  # (B, num_bins, num_marks)
            hist_loss = F.binary_cross_entropy_with_logits(
                hist_logits, histone_labels.float()
            )
            losses["histone_loss"] = hist_loss
            total = total + histone_weight * hist_loss

        losses["total_loss"] = total
        return losses
