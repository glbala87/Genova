"""Multi-omics data alignment and attention-based fusion for Genova.

Aligns heterogeneous multi-omics data (DNA sequence, methylation, RNA-seq)
to shared genomic windows and fuses them using learned attention.  Handles
the inherent sparsity of multi-omics data where not all positions have
measurements from all modalities.

Example::

    aligner = MultiOmicsAligner(window_size=512)
    aligned = aligner.align(
        methylation_features=meth_dict,
        rnaseq_features=rna_dict,
    )

    fusion = OmicsDataFusion(d_model=256, num_modalities=3)
    fused = fusion(modality_tensors, modality_masks)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor


class MultiOmicsAligner:
    """Align multi-omics measurements to shared genomic windows.

    Takes heterogeneous data sources (methylation from bedMethyl, RNA-seq
    gene expression, etc.) and produces per-window feature tensors aligned
    to a common coordinate system.  Handles sparse data by producing
    validity masks alongside feature tensors.

    Args:
        window_size: Genomic window size in base pairs.
        max_methylation_sites: Maximum methylation sites per window.
        max_genes: Maximum gene expression features per window.
    """

    def __init__(
        self,
        window_size: int = 512,
        max_methylation_sites: int = 128,
        max_genes: int = 64,
    ) -> None:
        self.window_size = window_size
        self.max_methylation_sites = max_methylation_sites
        self.max_genes = max_genes

        logger.debug(
            "MultiOmicsAligner: window={}, max_meth={}, max_genes={}",
            window_size,
            max_methylation_sites,
            max_genes,
        )

    # ------------------------------------------------------------------
    # Methylation alignment
    # ------------------------------------------------------------------

    def align_methylation(
        self,
        chrom: str,
        start: int,
        end: int,
        methylation_data: pd.DataFrame,
    ) -> Dict[str, Tensor]:
        """Align methylation measurements to a genomic window.

        Expects a DataFrame with columns: ``pos`` (genomic position),
        ``beta`` (methylation level 0-1), and optionally ``coverage``.

        Args:
            chrom: Chromosome name.
            start: Window start (0-based, inclusive).
            end: Window end (exclusive).
            methylation_data: DataFrame of methylation measurements
                with at least ``pos`` and ``beta`` columns.

        Returns:
            Dictionary with:
                - ``values``: ``(max_sites, 2)`` [beta, relative_pos].
                - ``mask``: ``(max_sites,)`` bool.
                - ``num_sites``: integer tensor.
        """
        max_sites = self.max_methylation_sites
        window_len = end - start

        # Filter to window
        if methylation_data is not None and len(methylation_data) > 0:
            in_window = methylation_data[
                (methylation_data["pos"] >= start) & (methylation_data["pos"] < end)
            ]
            if "chrom" in methylation_data.columns:
                in_window = in_window[in_window["chrom"] == chrom]
        else:
            in_window = pd.DataFrame()

        values = torch.zeros(max_sites, 2, dtype=torch.float32)
        mask = torch.zeros(max_sites, dtype=torch.bool)

        num_sites = min(len(in_window), max_sites)
        if num_sites > 0:
            in_window = in_window.sort_values("pos").iloc[:num_sites]
            values[:num_sites, 0] = torch.tensor(
                in_window["beta"].values, dtype=torch.float32
            )
            values[:num_sites, 1] = torch.tensor(
                (in_window["pos"].values - start) / max(window_len, 1),
                dtype=torch.float32,
            )
            mask[:num_sites] = True

        return {
            "values": values,
            "mask": mask,
            "num_sites": torch.tensor(num_sites, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # RNA-seq alignment
    # ------------------------------------------------------------------

    def align_rnaseq(
        self,
        chrom: str,
        start: int,
        end: int,
        rnaseq_data: pd.DataFrame,
    ) -> Dict[str, Tensor]:
        """Align RNA-seq gene expression to a genomic window.

        Expects a DataFrame with columns: ``gene_start``, ``gene_end``,
        ``expression`` (e.g. TPM or log-TPM), and optionally ``chrom``,
        ``gene_id``.

        Genes overlapping the window are included.  Expression values
        are paired with the relative midpoint of overlap.

        Args:
            chrom: Chromosome name.
            start: Window start.
            end: Window end.
            rnaseq_data: DataFrame with gene expression data.

        Returns:
            Dictionary with:
                - ``values``: ``(max_genes, 2)`` [expression, relative_pos].
                - ``mask``: ``(max_genes,)`` bool.
                - ``num_genes``: integer tensor.
        """
        max_genes = self.max_genes
        window_len = end - start

        if rnaseq_data is not None and len(rnaseq_data) > 0:
            overlapping = rnaseq_data[
                (rnaseq_data["gene_end"] > start)
                & (rnaseq_data["gene_start"] < end)
            ]
            if "chrom" in rnaseq_data.columns:
                overlapping = overlapping[overlapping["chrom"] == chrom]
        else:
            overlapping = pd.DataFrame()

        values = torch.zeros(max_genes, 2, dtype=torch.float32)
        mask = torch.zeros(max_genes, dtype=torch.bool)

        num_genes = min(len(overlapping), max_genes)
        if num_genes > 0:
            overlapping = overlapping.iloc[:num_genes]
            expr = overlapping["expression"].values.astype(np.float32)
            # Midpoint of overlap region relative to window
            mid = (
                np.clip(overlapping["gene_start"].values, start, end)
                + np.clip(overlapping["gene_end"].values, start, end)
            ) / 2.0
            rel_pos = (mid - start) / max(window_len, 1)

            values[:num_genes, 0] = torch.from_numpy(expr)
            values[:num_genes, 1] = torch.from_numpy(rel_pos.astype(np.float32))
            mask[:num_genes] = True

        return {
            "values": values,
            "mask": mask,
            "num_genes": torch.tensor(num_genes, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Combined alignment
    # ------------------------------------------------------------------

    def align(
        self,
        chrom: str,
        start: int,
        end: Optional[int] = None,
        methylation_data: Optional[pd.DataFrame] = None,
        rnaseq_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Dict[str, Tensor]]:
        """Align all available modalities to a genomic window.

        Args:
            chrom: Chromosome name.
            start: Window start position.
            end: Window end position (default: ``start + window_size``).
            methylation_data: Optional methylation DataFrame.
            rnaseq_data: Optional RNA-seq DataFrame.

        Returns:
            Dictionary keyed by modality name (``"methylation"``,
            ``"rnaseq"``) with per-modality aligned feature dictionaries.
            Only modalities with provided data are included.
        """
        if end is None:
            end = start + self.window_size

        result: Dict[str, Dict[str, Tensor]] = {}

        if methylation_data is not None:
            result["methylation"] = self.align_methylation(
                chrom, start, end, methylation_data
            )

        if rnaseq_data is not None:
            result["rnaseq"] = self.align_rnaseq(
                chrom, start, end, rnaseq_data
            )

        return result


class OmicsDataFusion(nn.Module):
    """Attention-based fusion of multi-omics modalities.

    Fuses feature representations from multiple omics modalities
    (DNA sequence, methylation, RNA-seq) using multi-head cross-attention.
    Each modality is first projected to a common dimensionality, then
    fused via a learned attention mechanism that weights modality
    contributions based on context.

    Handles missing modalities gracefully through masking -- if a
    modality is absent for a sample, its contribution is zeroed out.

    Args:
        d_model: Common dimensionality for all modalities after projection.
        num_modalities: Number of omics modalities (including DNA sequence).
        n_heads: Number of attention heads for fusion.
        dropout: Dropout rate.
        num_fusion_layers: Number of cross-attention fusion layers.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_modalities: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        num_fusion_layers: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.n_heads = n_heads

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        # Cross-attention layers for fusion
        self.fusion_layers = nn.ModuleList([
            _FusionLayer(d_model, n_heads, dropout)
            for _ in range(num_fusion_layers)
        ])

        # Modality-type embedding (learned token to distinguish modalities)
        self.modality_embedding = nn.Embedding(num_modalities, d_model)

        # Final projection after fusion
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(dropout),
        )

        logger.debug(
            "OmicsDataFusion: d_model={}, modalities={}, heads={}, layers={}",
            d_model,
            num_modalities,
            n_heads,
            num_fusion_layers,
        )

    def forward(
        self,
        modality_features: List[Tensor],
        modality_masks: Optional[List[Tensor]] = None,
        query: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse multi-omics modality features.

        Args:
            modality_features: List of ``num_modalities`` tensors, each
                ``(B, L_i, D)`` where ``L_i`` is the modality-specific
                sequence length.  Use a zero tensor for missing modalities.
            modality_masks: Optional list of ``(B, L_i)`` bool masks
                (``True`` = valid position).  If ``None``, all positions
                are assumed valid.
            query: Optional query tensor ``(B, L_q, D)`` for cross-attention.
                If ``None``, the first modality (typically DNA sequence)
                is used as the query.

        Returns:
            Fused representation ``(B, L_q, D)``.
        """
        n_mod = len(modality_features)
        if n_mod == 0:
            raise ValueError("At least one modality must be provided")

        B = modality_features[0].shape[0]
        device = modality_features[0].device

        # Add modality-type embeddings
        enriched: List[Tensor] = []
        for i, feat in enumerate(modality_features):
            if i < self.num_modalities:
                mod_id = torch.full(
                    (B,), i, dtype=torch.long, device=device
                )
                mod_emb = self.modality_embedding(mod_id).unsqueeze(1)  # (B, 1, D)
                enriched.append(feat + mod_emb)
            else:
                enriched.append(feat)

        # Concatenate all modalities along sequence dimension for KV
        kv = torch.cat(enriched, dim=1)  # (B, sum(L_i), D)

        # Build concatenated mask
        if modality_masks is not None:
            combined_mask = torch.cat(modality_masks, dim=1)  # (B, sum(L_i))
        else:
            combined_mask = None

        # Query: first modality (DNA sequence) or provided
        if query is not None:
            q = query
        else:
            q = enriched[0]

        # Apply fusion layers
        fused = q
        for layer in self.fusion_layers:
            fused = layer(fused, kv, kv_mask=combined_mask)

        return self.output_projection(fused)


class _FusionLayer(nn.Module):
    """Single cross-attention fusion layer with residual and feed-forward.

    Implements: cross-attention -> add & norm -> FFN -> add & norm.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        kv: Tensor,
        kv_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Cross-attention with pre-norm residual.

        Args:
            query: ``(B, L_q, D)``
            kv: ``(B, L_kv, D)``
            kv_mask: ``(B, L_kv)`` bool mask for key-value.

        Returns:
            ``(B, L_q, D)``
        """
        B, L_q, D = query.shape
        L_kv = kv.shape[1]

        # Pre-norm
        q_norm = self.norm1(query)

        q = self.q_proj(q_norm).view(B, L_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, L_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, L_kv, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, L_q, L_kv)

        if kv_mask is not None:
            mask = kv_mask[:, None, None, :].to(dtype=attn.dtype)
            attn = attn + (1.0 - mask) * torch.finfo(attn.dtype).min

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, v)  # (B, H, L_q, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, L_q, D)

        # Residual connection
        x = query + self.out_dropout(self.out_proj(context))

        # FFN with pre-norm residual
        x = x + self.ffn(self.norm2(x))

        return x
