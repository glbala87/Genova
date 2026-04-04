"""Enhancer-Promoter interaction prediction for Genova.

Predicts whether an enhancer-promoter pair physically interacts
based on their sequence embeddings and optional genomic distance
features.

Example::

    from genova.evaluation.epi_interaction import EPInteractionPredictor

    predictor = EPInteractionPredictor(encoder, d_model=768)
    score = predictor.predict_interaction(enhancer_ids, promoter_ids)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EPPair:
    """An enhancer-promoter pair for interaction prediction.

    Attributes:
        enhancer_ids: Token IDs for the enhancer sequence.
        promoter_ids: Token IDs for the promoter sequence.
        distance: Genomic distance between the pair in base pairs.
            ``None`` if unknown.
        label: Ground-truth interaction label (1=interacts, 0=not),
            ``None`` if unlabelled.
    """

    enhancer_ids: Tensor
    promoter_ids: Tensor
    distance: Optional[float] = None
    label: Optional[int] = None


@dataclass
class EPInteractionResult:
    """Result of an enhancer-promoter interaction prediction.

    Attributes:
        score: Interaction probability in ``[0, 1]``.
        label: Predicted label (``"interacting"`` or ``"non-interacting"``).
        enhancer_embedding: Mean-pooled enhancer embedding (optional).
        promoter_embedding: Mean-pooled promoter embedding (optional).
    """

    score: float
    label: str
    enhancer_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    promoter_embedding: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Distance encoder
# ---------------------------------------------------------------------------


class _DistanceEncoder(nn.Module):
    """Encode genomic distance as a fixed-size feature vector.

    Uses log-scaled distance with a small MLP.

    Args:
        d_out: Output feature dimension.
    """

    def __init__(self, d_out: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, distance: Tensor) -> Tensor:
        """Encode distances.

        Args:
            distance: ``(B,)`` or ``(B, 1)`` raw distances in base pairs.

        Returns:
            ``(B, d_out)`` distance features.
        """
        if distance.dim() == 1:
            distance = distance.unsqueeze(-1)
        # Log-scale to handle wide range of genomic distances
        log_dist = torch.log1p(distance.float())
        return self.net(log_dist)


# ---------------------------------------------------------------------------
# EPInteractionPredictor
# ---------------------------------------------------------------------------


class EPInteractionPredictor(nn.Module):
    """Predict enhancer-promoter interactions from sequence.

    Encodes enhancer and promoter sequences separately through the
    shared encoder, pools the representations, and combines them
    (with optional distance features) to predict interaction probability.

    Args:
        encoder: Pretrained Genova encoder returning a dict with
            ``"last_hidden_state"`` of shape ``(B, L, D)``.
        d_model: Hidden dimension of encoder output.
        use_distance: Whether to include genomic distance features.
        d_distance: Dimension of the distance encoding.
        dropout: Dropout probability.
        threshold: Decision threshold for binary interaction calls.
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_model: int = 768,
        use_distance: bool = True,
        d_distance: int = 64,
        dropout: float = 0.1,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model
        self.use_distance = use_distance
        self.threshold = threshold

        # Sequence-level pooling projections
        self.enhancer_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model, eps=1e-12),
        )
        self.promoter_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model, eps=1e-12),
        )

        # Distance encoder
        self.distance_encoder = _DistanceEncoder(d_distance) if use_distance else None

        # Interaction classifier
        # Features: enhancer emb + promoter emb + element-wise product
        #           + optional distance features
        classifier_in = d_model * 3 + (d_distance if use_distance else 0)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _encode_sequence(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run encoder and return mean-pooled hidden states.

        Args:
            input_ids: ``(B, L)``
            attention_mask: ``(B, L)``

        Returns:
            ``(B, D)`` pooled embeddings.
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

        # Mean pool
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden.mean(dim=1)

        return pooled  # (B, D)

    def forward(
        self,
        enhancer_ids: Tensor,
        promoter_ids: Tensor,
        distance: Optional[Tensor] = None,
        enhancer_mask: Optional[Tensor] = None,
        promoter_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass returning interaction logits.

        Args:
            enhancer_ids: ``(B, L_e)``
            promoter_ids: ``(B, L_p)``
            distance: ``(B,)`` genomic distances (optional).
            enhancer_mask: ``(B, L_e)`` attention mask.
            promoter_mask: ``(B, L_p)`` attention mask.

        Returns:
            Logits tensor of shape ``(B, 1)``.
        """
        enh_emb = self._encode_sequence(enhancer_ids, enhancer_mask)
        enh_emb = self.enhancer_proj(enh_emb)  # (B, D)

        prom_emb = self._encode_sequence(promoter_ids, promoter_mask)
        prom_emb = self.promoter_proj(prom_emb)  # (B, D)

        # Combine features
        product = enh_emb * prom_emb  # element-wise product
        features = torch.cat([enh_emb, prom_emb, product], dim=-1)

        if self.use_distance and distance is not None and self.distance_encoder is not None:
            dist_feat = self.distance_encoder(distance)
            features = torch.cat([features, dist_feat], dim=-1)
        elif self.use_distance and self.distance_encoder is not None:
            # Use zero distance if not provided
            dummy_dist = torch.zeros(enh_emb.size(0), device=enh_emb.device)
            dist_feat = self.distance_encoder(dummy_dist)
            features = torch.cat([features, dist_feat], dim=-1)

        return self.classifier(features)  # (B, 1)

    @torch.no_grad()
    def predict_interaction(
        self,
        enhancer_ids: Tensor,
        promoter_ids: Tensor,
        distance: Optional[Union[float, Tensor]] = None,
        enhancer_mask: Optional[Tensor] = None,
        promoter_mask: Optional[Tensor] = None,
    ) -> EPInteractionResult:
        """Predict interaction for a single enhancer-promoter pair.

        Args:
            enhancer_ids: ``(1, L_e)`` or ``(L_e,)``.
            promoter_ids: ``(1, L_p)`` or ``(L_p,)``.
            distance: Genomic distance in base pairs (optional).
            enhancer_mask: Optional attention mask for enhancer.
            promoter_mask: Optional attention mask for promoter.

        Returns:
            :class:`EPInteractionResult` with the interaction score.
        """
        self.eval()
        if enhancer_ids.dim() == 1:
            enhancer_ids = enhancer_ids.unsqueeze(0)
        if promoter_ids.dim() == 1:
            promoter_ids = promoter_ids.unsqueeze(0)
        if enhancer_mask is not None and enhancer_mask.dim() == 1:
            enhancer_mask = enhancer_mask.unsqueeze(0)
        if promoter_mask is not None and promoter_mask.dim() == 1:
            promoter_mask = promoter_mask.unsqueeze(0)

        dist_tensor: Optional[Tensor] = None
        if distance is not None:
            if isinstance(distance, (int, float)):
                dist_tensor = torch.tensor(
                    [distance], dtype=torch.float32, device=enhancer_ids.device
                )
            else:
                dist_tensor = distance

        logit = self.forward(
            enhancer_ids, promoter_ids, dist_tensor, enhancer_mask, promoter_mask
        )
        score = float(torch.sigmoid(logit).item())
        label = "interacting" if score >= self.threshold else "non-interacting"

        # Get embeddings for downstream use
        enh_emb = self.enhancer_proj(
            self._encode_sequence(enhancer_ids, enhancer_mask)
        )
        prom_emb = self.promoter_proj(
            self._encode_sequence(promoter_ids, promoter_mask)
        )

        return EPInteractionResult(
            score=score,
            label=label,
            enhancer_embedding=enh_emb[0].cpu().numpy(),
            promoter_embedding=prom_emb[0].cpu().numpy(),
        )

    @torch.no_grad()
    def predict_batch(
        self,
        pairs: List[EPPair],
    ) -> List[EPInteractionResult]:
        """Predict interactions for a batch of enhancer-promoter pairs.

        Args:
            pairs: List of :class:`EPPair` objects.

        Returns:
            List of :class:`EPInteractionResult` objects.
        """
        self.eval()
        if not pairs:
            return []

        device = next(self.parameters()).device

        # Pad enhancer and promoter sequences separately
        max_enh = max(p.enhancer_ids.numel() for p in pairs)
        max_prom = max(p.promoter_ids.numel() for p in pairs)

        enh_batch = torch.zeros(len(pairs), max_enh, dtype=torch.long, device=device)
        prom_batch = torch.zeros(len(pairs), max_prom, dtype=torch.long, device=device)
        enh_masks = torch.zeros(len(pairs), max_enh, dtype=torch.long, device=device)
        prom_masks = torch.zeros(len(pairs), max_prom, dtype=torch.long, device=device)

        distances: List[float] = []
        for i, p in enumerate(pairs):
            enh = p.enhancer_ids.flatten()
            prom = p.promoter_ids.flatten()
            enh_batch[i, : enh.numel()] = enh
            prom_batch[i, : prom.numel()] = prom
            enh_masks[i, : enh.numel()] = 1
            prom_masks[i, : prom.numel()] = 1
            distances.append(p.distance if p.distance is not None else 0.0)

        dist_tensor = torch.tensor(distances, dtype=torch.float32, device=device)

        logits = self.forward(
            enh_batch, prom_batch, dist_tensor, enh_masks, prom_masks
        )
        scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

        results: List[EPInteractionResult] = []
        for i, s in enumerate(scores):
            score = float(s)
            label = "interacting" if score >= self.threshold else "non-interacting"
            results.append(EPInteractionResult(score=score, label=label))

        return results

    @torch.no_grad()
    def find_targets(
        self,
        enhancer_ids: Tensor,
        candidate_promoters: List[Tensor],
        distances: Optional[List[float]] = None,
        enhancer_mask: Optional[Tensor] = None,
        top_k: Optional[int] = None,
    ) -> List[EPInteractionResult]:
        """Find the most likely promoter targets for an enhancer.

        Args:
            enhancer_ids: ``(1, L)`` or ``(L,)`` enhancer token IDs.
            candidate_promoters: List of promoter token ID tensors.
            distances: Optional list of genomic distances.
            enhancer_mask: Optional attention mask for enhancer.
            top_k: Return only the top-k ranked targets.  If ``None``,
                return all candidates sorted by score.

        Returns:
            List of :class:`EPInteractionResult`, sorted by descending
            score.
        """
        self.eval()
        if not candidate_promoters:
            return []

        pairs: List[EPPair] = []
        for i, prom in enumerate(candidate_promoters):
            dist = distances[i] if distances is not None else None
            pairs.append(EPPair(
                enhancer_ids=enhancer_ids.flatten(),
                promoter_ids=prom.flatten(),
                distance=dist,
            ))

        results = self.predict_batch(pairs)
        # Sort by descending score
        results.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    def compute_loss(
        self,
        enhancer_ids: Tensor,
        promoter_ids: Tensor,
        labels: Tensor,
        distance: Optional[Tensor] = None,
        enhancer_mask: Optional[Tensor] = None,
        promoter_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute binary cross-entropy loss for interaction prediction.

        Args:
            enhancer_ids: ``(B, L_e)``
            promoter_ids: ``(B, L_p)``
            labels: Binary interaction labels ``(B,)`` or ``(B, 1)``.
            distance: Optional distances ``(B,)``.
            enhancer_mask: ``(B, L_e)``
            promoter_mask: ``(B, L_p)``

        Returns:
            Scalar loss tensor.
        """
        logits = self.forward(
            enhancer_ids, promoter_ids, distance, enhancer_mask, promoter_mask
        )
        labels = labels.float().view_as(logits)
        return F.binary_cross_entropy_with_logits(logits, labels)
