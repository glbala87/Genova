"""Multi-omics encoder and integration model for Genova.

Processes each omics modality (DNA sequence, methylation, RNA-seq) through
dedicated encoders before fusing them via cross-modal attention.  Missing
modalities are handled gracefully through learned masking so the model
can operate with any subset of available data.

Example::

    from genova.utils.config import ModelConfig
    cfg = ModelConfig(d_model=256, n_heads=4, n_layers=6, d_ff=1024,
                      vocab_size=4096)
    model = MultiOmicsGenovaModel(cfg)
    out = model(
        input_ids=seq_tokens,
        methylation_values=meth_vals,
        methylation_mask=meth_mask,
    )
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from genova.utils.config import ModelConfig
from genova.models.transformer import GenovaTransformer
from genova.multiomics.data_fusion import OmicsDataFusion


# Modality indices used for modality-type embeddings
MODALITY_DNA: int = 0
MODALITY_METHYLATION: int = 1
MODALITY_RNASEQ: int = 2


class ModalityProjection(nn.Module):
    """Project raw modality features into the model's hidden space.

    Each modality has different raw feature dimensions.  This module
    provides a modality-specific MLP projection with layer normalisation.

    Args:
        input_dim: Raw feature dimensionality for this modality.
        output_dim: Target dimensionality (typically ``d_model``).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim, eps=1e-12),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project features: ``(B, L, input_dim) -> (B, L, output_dim)``."""
        return self.projection(x)


class MethylationEncoder(nn.Module):
    """Encode methylation features (beta values + positions) per window.

    Takes sparse CpG-site-level methylation measurements and produces
    dense per-site representations.

    Args:
        d_model: Output dimensionality.
        max_sites: Maximum CpG sites per window.
        input_features: Number of input features per site (default 2:
            beta value and relative position).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        max_sites: int = 128,
        input_features: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_sites = max_sites

        self.site_encoder = nn.Sequential(
            nn.Linear(input_features, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(dropout),
        )

        # Self-attention to capture CpG-CpG interactions
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=max(1, d_model // 64),
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(
        self,
        methylation_values: Tensor,
        methylation_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode methylation features.

        Args:
            methylation_values: ``(B, max_sites, input_features)`` with
                features like [beta, relative_position].
            methylation_mask: ``(B, max_sites)`` bool mask (True = real site).

        Returns:
            ``(B, max_sites, d_model)`` encoded methylation features.
        """
        x = self.site_encoder(methylation_values)  # (B, S, D)

        # Mask invalid sites
        if methylation_mask is not None:
            x = x * methylation_mask.unsqueeze(-1).float()

        # Self-attention among CpG sites
        key_padding_mask = None
        if methylation_mask is not None:
            # MultiheadAttention expects True = ignore
            key_padding_mask = ~methylation_mask

        attn_out, _ = self.self_attn(
            x, x, x, key_padding_mask=key_padding_mask
        )
        x = self.attn_norm(x + attn_out)

        if methylation_mask is not None:
            x = x * methylation_mask.unsqueeze(-1).float()

        return x


class RNASeqEncoder(nn.Module):
    """Encode RNA-seq gene expression features per window.

    Takes per-gene expression values with positional information and
    produces dense representations.

    Args:
        d_model: Output dimensionality.
        max_genes: Maximum genes per window.
        input_features: Number of input features per gene (default 2:
            expression value and relative position).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        max_genes: int = 64,
        input_features: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_genes = max_genes

        self.gene_encoder = nn.Sequential(
            nn.Linear(input_features, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(dropout),
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=max(1, d_model // 64),
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(
        self,
        rnaseq_values: Tensor,
        rnaseq_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode RNA-seq features.

        Args:
            rnaseq_values: ``(B, max_genes, input_features)`` with
                features like [expression, relative_position].
            rnaseq_mask: ``(B, max_genes)`` bool mask.

        Returns:
            ``(B, max_genes, d_model)`` encoded RNA-seq features.
        """
        x = self.gene_encoder(rnaseq_values)

        if rnaseq_mask is not None:
            x = x * rnaseq_mask.unsqueeze(-1).float()

        key_padding_mask = None
        if rnaseq_mask is not None:
            key_padding_mask = ~rnaseq_mask

        attn_out, _ = self.self_attn(
            x, x, x, key_padding_mask=key_padding_mask
        )
        x = self.attn_norm(x + attn_out)

        if rnaseq_mask is not None:
            x = x * rnaseq_mask.unsqueeze(-1).float()

        return x


class MultiOmicsEncoder(nn.Module):
    """Process each omics modality separately then fuse via attention.

    Orchestrates modality-specific encoders (DNA backbone, methylation
    encoder, RNA-seq encoder) and fuses their outputs using
    :class:`OmicsDataFusion`.

    Args:
        config: Model configuration for the DNA backbone.
        max_methylation_sites: Max CpG sites per window.
        max_genes: Max genes per window.
        methylation_features: Input features per CpG site.
        rnaseq_features: Input features per gene.
        fusion_heads: Number of attention heads in the fusion layer.
        fusion_layers: Number of cross-attention fusion layers.
        embedding_type: Positional encoding for the DNA backbone.
    """

    def __init__(
        self,
        config: ModelConfig,
        max_methylation_sites: int = 128,
        max_genes: int = 64,
        methylation_features: int = 2,
        rnaseq_features: int = 2,
        fusion_heads: int = 4,
        fusion_layers: int = 2,
        embedding_type: str = "learned",
    ) -> None:
        super().__init__()
        self.config = config

        # DNA sequence backbone
        self.dna_backbone = GenovaTransformer(config, embedding_type=embedding_type)

        # Modality-specific encoders
        self.methylation_encoder = MethylationEncoder(
            d_model=config.d_model,
            max_sites=max_methylation_sites,
            input_features=methylation_features,
            dropout=config.dropout,
        )

        self.rnaseq_encoder = RNASeqEncoder(
            d_model=config.d_model,
            max_genes=max_genes,
            input_features=rnaseq_features,
            dropout=config.dropout,
        )

        # Cross-modal fusion
        self.fusion = OmicsDataFusion(
            d_model=config.d_model,
            num_modalities=3,  # DNA, methylation, RNA-seq
            n_heads=fusion_heads,
            dropout=config.dropout,
            num_fusion_layers=fusion_layers,
        )

        # Modality availability gate (learns to compensate for missing data)
        self.modality_gate = nn.Sequential(
            nn.Linear(3, config.d_model),  # 3 = number of modalities
            nn.Sigmoid(),
        )

        logger.info(
            "MultiOmicsEncoder: DNA backbone + methylation + RNA-seq, "
            "d_model={}, fusion_heads={}, fusion_layers={}",
            config.d_model,
            fusion_heads,
            fusion_layers,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        methylation_values: Optional[Tensor] = None,
        methylation_mask: Optional[Tensor] = None,
        rnaseq_values: Optional[Tensor] = None,
        rnaseq_mask: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward pass through all modality encoders and fusion.

        Args:
            input_ids: ``(B, L)`` DNA sequence token ids.
            attention_mask: ``(B, L)`` sequence attention mask.
            methylation_values: ``(B, S, F_m)`` methylation features.
            methylation_mask: ``(B, S)`` methylation site mask.
            rnaseq_values: ``(B, G, F_r)`` RNA-seq features.
            rnaseq_mask: ``(B, G)`` gene mask.
            segment_ids: Optional segment ids for DNA backbone.
            output_hidden_states: Return intermediate hidden states.

        Returns:
            Dict with:
                - ``last_hidden_state``: ``(B, L, D)`` fused representation.
                - ``dna_hidden``: ``(B, L, D)`` DNA-only representation.
                - ``modality_availability``: ``(B, 3)`` which modalities present.
                - ``hidden_states`` (optional): from DNA backbone.
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # 1. Encode DNA sequence
        dna_out = self.dna_backbone(
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            output_hidden_states=output_hidden_states,
        )
        dna_hidden = dna_out["last_hidden_state"]  # (B, L, D)

        # 2. Track modality availability
        modality_avail = torch.ones(B, 3, device=device, dtype=torch.float32)

        # 3. Prepare modality feature lists for fusion
        modality_features: List[Tensor] = [dna_hidden]
        modality_masks_list: List[Tensor] = []

        if attention_mask is not None:
            modality_masks_list.append(attention_mask.bool())
        else:
            modality_masks_list.append(
                torch.ones(B, dna_hidden.shape[1], device=device, dtype=torch.bool)
            )

        # 4. Encode methylation if available
        if methylation_values is not None:
            meth_encoded = self.methylation_encoder(
                methylation_values, methylation_mask
            )
            modality_features.append(meth_encoded)
            if methylation_mask is not None:
                modality_masks_list.append(methylation_mask.bool())
            else:
                modality_masks_list.append(
                    torch.ones(B, meth_encoded.shape[1], device=device, dtype=torch.bool)
                )
        else:
            modality_avail[:, MODALITY_METHYLATION] = 0.0
            # Add dummy zero features so fusion input count is consistent
            dummy_meth = torch.zeros(
                B, 1, self.config.d_model, device=device, dtype=torch.float32
            )
            modality_features.append(dummy_meth)
            modality_masks_list.append(
                torch.zeros(B, 1, device=device, dtype=torch.bool)
            )

        # 5. Encode RNA-seq if available
        if rnaseq_values is not None:
            rna_encoded = self.rnaseq_encoder(rnaseq_values, rnaseq_mask)
            modality_features.append(rna_encoded)
            if rnaseq_mask is not None:
                modality_masks_list.append(rnaseq_mask.bool())
            else:
                modality_masks_list.append(
                    torch.ones(B, rna_encoded.shape[1], device=device, dtype=torch.bool)
                )
        else:
            modality_avail[:, MODALITY_RNASEQ] = 0.0
            dummy_rna = torch.zeros(
                B, 1, self.config.d_model, device=device, dtype=torch.float32
            )
            modality_features.append(dummy_rna)
            modality_masks_list.append(
                torch.zeros(B, 1, device=device, dtype=torch.bool)
            )

        # 6. Apply modality availability gate
        mod_gate = self.modality_gate(modality_avail)  # (B, D)

        # 7. Cross-modal fusion
        fused = self.fusion(
            modality_features,
            modality_masks=modality_masks_list,
            query=dna_hidden,
        )  # (B, L, D)

        # Apply modality gate (scales output based on available modalities)
        fused = fused * mod_gate.unsqueeze(1)

        result: Dict[str, Tensor] = {
            "last_hidden_state": fused,
            "dna_hidden": dna_hidden,
            "modality_availability": modality_avail,
        }

        if output_hidden_states and "hidden_states" in dna_out:
            result["hidden_states"] = dna_out["hidden_states"]

        return result


class MultiOmicsGenovaModel(nn.Module):
    """Full multi-omics Genova model with task heads.

    End-to-end model that integrates DNA sequence, methylation, and
    RNA-seq data through modality-specific encoders and cross-modal
    attention fusion.  Supports downstream tasks via configurable
    prediction heads.

    Args:
        config: Model configuration.
        max_methylation_sites: Max CpG sites per window.
        max_genes: Max genes per window.
        methylation_features: Input features per CpG site.
        rnaseq_features: Input features per gene.
        fusion_heads: Attention heads in fusion layers.
        fusion_layers: Number of fusion layers.
        num_classes: Number of output classes (0 = regression mode).
        pool: Pooling strategy for classification/regression.
        embedding_type: Positional encoding for DNA backbone.
    """

    def __init__(
        self,
        config: ModelConfig,
        max_methylation_sites: int = 128,
        max_genes: int = 64,
        methylation_features: int = 2,
        rnaseq_features: int = 2,
        fusion_heads: int = 4,
        fusion_layers: int = 2,
        num_classes: int = 0,
        pool: str = "cls",
        embedding_type: str = "learned",
    ) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.pool = pool

        # Multi-omics encoder
        self.encoder = MultiOmicsEncoder(
            config=config,
            max_methylation_sites=max_methylation_sites,
            max_genes=max_genes,
            methylation_features=methylation_features,
            rnaseq_features=rnaseq_features,
            fusion_heads=fusion_heads,
            fusion_layers=fusion_layers,
            embedding_type=embedding_type,
        )

        # Optional classification/regression head
        if num_classes > 0:
            self.head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model, eps=1e-12),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, num_classes),
            )
        else:
            self.head = None

        logger.info(
            "MultiOmicsGenovaModel: d_model={}, num_classes={}, pool={}",
            config.d_model,
            num_classes,
            pool,
        )

    def _pool(
        self,
        hidden: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Pool sequence to a single vector."""
        if self.pool == "cls":
            return hidden[:, 0]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return hidden.mean(dim=1)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        methylation_values: Optional[Tensor] = None,
        methylation_mask: Optional[Tensor] = None,
        rnaseq_values: Optional[Tensor] = None,
        rnaseq_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward pass through the full multi-omics model.

        Args:
            input_ids: ``(B, L)`` DNA token ids.
            attention_mask: ``(B, L)`` attention mask.
            methylation_values: ``(B, S, F_m)`` methylation features.
            methylation_mask: ``(B, S)`` methylation mask.
            rnaseq_values: ``(B, G, F_r)`` RNA-seq features.
            rnaseq_mask: ``(B, G)`` gene mask.
            labels: ``(B,)`` for classification or ``(B, T)`` for regression.
            segment_ids: Optional segment ids.
            output_hidden_states: Return intermediate hidden states.

        Returns:
            Dict with:
                - ``last_hidden_state``: ``(B, L, D)`` fused.
                - ``dna_hidden``: ``(B, L, D)`` DNA-only.
                - ``modality_availability``: ``(B, 3)``.
                - ``logits``: ``(B, num_classes)`` if head exists.
                - ``loss``: scalar if labels provided.
                - ``hidden_states`` (optional).
        """
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            methylation_values=methylation_values,
            methylation_mask=methylation_mask,
            rnaseq_values=rnaseq_values,
            rnaseq_mask=rnaseq_mask,
            segment_ids=segment_ids,
            output_hidden_states=output_hidden_states,
        )

        result: Dict[str, Tensor] = {
            "last_hidden_state": enc_out["last_hidden_state"],
            "dna_hidden": enc_out["dna_hidden"],
            "modality_availability": enc_out["modality_availability"],
        }

        # Apply task head if configured
        if self.head is not None:
            hidden = enc_out["last_hidden_state"]
            pooled = self._pool(hidden, attention_mask)
            logits = self.head(pooled)
            result["logits"] = logits

            if labels is not None:
                if self.num_classes == 1:
                    # Regression
                    loss = F.mse_loss(logits.squeeze(-1), labels.float())
                else:
                    # Classification
                    loss = F.cross_entropy(logits, labels)
                result["loss"] = loss

        if output_hidden_states and "hidden_states" in enc_out:
            result["hidden_states"] = enc_out["hidden_states"]

        return result
