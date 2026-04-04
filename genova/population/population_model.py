"""Population-aware genomic models for Genova.

Provides learnable population embeddings, variant frequency integration,
and population-aware encoder / predictor architectures that combine DNA
sequence representations with population-specific allele frequency context.

Clinical impact: pathogenic variant frequencies differ substantially across
populations (e.g. sickle-cell variants in AFR, BRCA founder mutations in
EUR/Ashkenazi).  Population-aware models capture these differences to
improve variant classification and clinical interpretation.

Example::

    from genova.utils.config import ModelConfig
    cfg = ModelConfig(d_model=256, n_heads=4, n_layers=6, d_ff=1024,
                      vocab_size=4096)
    model = PopulationAwareVariantPredictor(
        config=cfg,
        num_populations=7,
        num_af_features=7,
        num_variant_classes=5,
    )
    out = model(input_ids, population_ids=pop_ids, af_features=af_feats)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from genova.utils.config import ModelConfig
from genova.models.transformer import GenovaTransformer
from genova.population.frequency_encoder import AlleleFrequencyEncoder


# Default population labels covering major superpopulations and Middle East
DEFAULT_POPULATION_LABELS: List[str] = [
    "EUR",  # European
    "AFR",  # African / African-American
    "EAS",  # East Asian
    "SAS",  # South Asian
    "AMR",  # Ad-mixed American / Latino
    "MEA",  # Middle Eastern / Arab
    "OCE",  # Oceanian
]


class PopulationEmbedding(nn.Module):
    """Learnable embedding layer for population labels.

    Maps discrete population identifiers to dense vectors that capture
    population-specific genetic structure.  Includes an ``UNKNOWN``
    population (index 0) for samples without ancestry information.

    Args:
        num_populations: Number of known population labels (excluding
            the UNKNOWN sentinel which is added automatically).
        embedding_dim: Dimensionality of population embeddings.
        dropout: Dropout rate applied after embedding lookup.
        population_labels: Optional human-readable labels for logging.
    """

    UNKNOWN_INDEX: int = 0

    def __init__(
        self,
        num_populations: int = 7,
        embedding_dim: int = 64,
        dropout: float = 0.1,
        population_labels: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        # +1 for the UNKNOWN sentinel at index 0
        self.num_populations = num_populations
        self.total_embeddings = num_populations + 1
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=self.total_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=None,
        )
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        # Label mapping for convenience
        self.labels = population_labels or DEFAULT_POPULATION_LABELS[:num_populations]
        self._label_to_index = {
            label: i + 1 for i, label in enumerate(self.labels)
        }
        self._label_to_index["UNKNOWN"] = self.UNKNOWN_INDEX

        self._init_weights()

        logger.debug(
            "PopulationEmbedding: {} populations + UNKNOWN, dim={}",
            num_populations,
            embedding_dim,
        )

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)

    def label_to_index(self, label: str) -> int:
        """Convert a population label string to its integer index.

        Returns ``UNKNOWN_INDEX`` (0) for unrecognised labels.
        """
        return self._label_to_index.get(label.upper(), self.UNKNOWN_INDEX)

    def labels_to_tensor(
        self,
        labels: List[str],
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Convert a batch of population label strings to an index tensor.

        Args:
            labels: List of population label strings.
            device: Target device for the tensor.

        Returns:
            ``(B,)`` long tensor of population indices.
        """
        indices = [self.label_to_index(l) for l in labels]
        return torch.tensor(indices, dtype=torch.long, device=device)

    def forward(self, population_ids: Tensor) -> Tensor:
        """Look up population embeddings.

        Args:
            population_ids: ``(B,)`` or ``(B, 1)`` integer tensor of
                population indices.

        Returns:
            ``(B, embedding_dim)`` float tensor.
        """
        if population_ids.dim() > 1:
            population_ids = population_ids.squeeze(-1)
        x = self.embedding(population_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class VariantFrequencyEncoder(nn.Module):
    """Project raw allele frequency features into the model's hidden space.

    Takes log-scaled allele frequency vectors (one per variant from
    :class:`AlleleFrequencyEncoder`) and projects them through a small
    MLP with layer normalisation.

    Args:
        num_af_features: Number of input AF features per variant
            (typically ``len(populations) + 1`` for global AF).
        hidden_dim: Intermediate MLP width.
        output_dim: Output dimensionality (should match ``d_model``).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_af_features: int = 7,
        hidden_dim: int = 64,
        output_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(num_af_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim, eps=1e-12),
        )

    def forward(
        self,
        af_features: Tensor,
        af_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Project allele frequency features.

        Args:
            af_features: ``(B, num_af_features)`` or
                ``(B, N_variants, num_af_features)`` log-scaled AFs.
            af_mask: Optional boolean mask ``(B, num_af_features)``
                indicating which AF values are observed.  When provided,
                missing values are zeroed before projection.

        Returns:
            Projected features with same batch dimensions plus
            ``output_dim`` as the last dimension.
        """
        if af_mask is not None:
            af_features = af_features * af_mask.float()
        return self.projection(af_features)


class PopulationAwareEncoder(nn.Module):
    """Combine DNA sequence embeddings with population and AF context.

    Wraps a :class:`GenovaTransformer` backbone and augments its output
    by fusing in population embeddings and variant frequency features
    via gated addition.  The gating mechanism allows the model to learn
    how much population context to inject at each position.

    Args:
        config: Model configuration for the transformer backbone.
        num_populations: Number of distinct population labels.
        num_af_features: Number of allele frequency features per variant.
        population_embedding_dim: Dimensionality of population embeddings.
        embedding_type: Positional encoding type for the backbone.
        population_labels: Optional list of population label strings.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_populations: int = 7,
        num_af_features: int = 7,
        population_embedding_dim: int = 64,
        embedding_type: str = "learned",
        population_labels: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.config = config

        # DNA sequence backbone
        self.backbone = GenovaTransformer(config, embedding_type=embedding_type)

        # Population embedding
        self.population_embedding = PopulationEmbedding(
            num_populations=num_populations,
            embedding_dim=population_embedding_dim,
            dropout=config.dropout,
            population_labels=population_labels,
        )

        # Project population embedding to model dimension
        self.pop_projection = nn.Linear(population_embedding_dim, config.d_model)

        # Variant frequency encoder
        self.variant_freq_encoder = VariantFrequencyEncoder(
            num_af_features=num_af_features,
            hidden_dim=max(64, config.d_model // 4),
            output_dim=config.d_model,
            dropout=config.dropout,
        )

        # Gating mechanism for population context injection
        # Learns per-position how much population info to incorporate
        self.pop_gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Sigmoid(),
        )

        # Gating mechanism for AF context injection
        self.af_gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Sigmoid(),
        )

        # Final layer norm after fusion
        self.fusion_norm = nn.LayerNorm(config.d_model, eps=1e-12)

        logger.info(
            "PopulationAwareEncoder: backbone_dim={}, pop_dim={}, "
            "num_pops={}, num_af={}",
            config.d_model,
            population_embedding_dim,
            num_populations,
            num_af_features,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        population_ids: Optional[Tensor] = None,
        af_features: Optional[Tensor] = None,
        af_mask: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward pass with population-aware context fusion.

        Args:
            input_ids: ``(B, L)`` token ids for DNA sequence.
            attention_mask: ``(B, L)`` mask (1 = real, 0 = pad).
            population_ids: ``(B,)`` integer population indices.
                If ``None``, no population context is injected.
            af_features: ``(B, num_af_features)`` log-scaled allele
                frequencies.  If ``None``, no AF context is injected.
            af_mask: ``(B, num_af_features)`` boolean mask for AFs.
            segment_ids: Optional segment ids for the backbone.
            output_hidden_states: Whether to return intermediate states.

        Returns:
            Dict with:
                - ``last_hidden_state``: ``(B, L, D)``
                - ``population_embedding``: ``(B, D)`` (if pop_ids given)
                - ``af_embedding``: ``(B, D)`` (if af_features given)
                - ``hidden_states`` (optional): list of ``(B, L, D)``
        """
        # 1. Run backbone
        enc_out = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            output_hidden_states=output_hidden_states,
        )
        hidden = enc_out["last_hidden_state"]  # (B, L, D)
        result: Dict[str, Tensor] = {}

        # 2. Inject population context via gated addition
        if population_ids is not None:
            pop_emb = self.population_embedding(population_ids)  # (B, pop_dim)
            pop_proj = self.pop_projection(pop_emb)  # (B, D)
            result["population_embedding"] = pop_proj

            # Broadcast to sequence length: (B, 1, D) -> (B, L, D)
            pop_expanded = pop_proj.unsqueeze(1).expand_as(hidden)

            # Compute gate: concat hidden and pop, then sigmoid
            gate_input = torch.cat([hidden, pop_expanded], dim=-1)  # (B, L, 2D)
            gate = self.pop_gate(gate_input)  # (B, L, D)

            hidden = hidden + gate * pop_expanded

        # 3. Inject AF context via gated addition
        if af_features is not None:
            af_emb = self.variant_freq_encoder(af_features, af_mask)  # (B, D)
            result["af_embedding"] = af_emb

            # If af_emb is per-variant (2D), broadcast to sequence
            if af_emb.dim() == 2:
                af_expanded = af_emb.unsqueeze(1).expand_as(hidden)
            else:
                af_expanded = af_emb

            gate_input = torch.cat([hidden, af_expanded], dim=-1)
            gate = self.af_gate(gate_input)

            hidden = hidden + gate * af_expanded

        hidden = self.fusion_norm(hidden)
        result["last_hidden_state"] = hidden

        if output_hidden_states and "hidden_states" in enc_out:
            result["hidden_states"] = enc_out["hidden_states"]

        return result


class PopulationAwareVariantPredictor(nn.Module):
    """Variant pathogenicity predictor with population context.

    Extends standard variant classification by incorporating population
    embeddings and allele frequency features.  This enables the model
    to learn that variant pathogenicity thresholds differ across
    populations (e.g. a variant common in one population may be benign
    while the same variant is ultra-rare and pathogenic in another).

    Output classes follow ACMG guidelines:
        0 = Benign, 1 = Likely Benign, 2 = VUS,
        3 = Likely Pathogenic, 4 = Pathogenic

    Args:
        config: Model configuration.
        num_populations: Number of population labels.
        num_af_features: Number of AF features per variant.
        num_variant_classes: Number of output classification classes.
        population_embedding_dim: Dimensionality of population embeddings.
        pool: Pooling strategy for classification (``"cls"`` or ``"mean"``).
        embedding_type: Positional encoding type for backbone.
        population_labels: Optional list of population label strings.
    """

    ACMG_CLASSES: List[str] = [
        "Benign",
        "Likely_Benign",
        "VUS",
        "Likely_Pathogenic",
        "Pathogenic",
    ]

    def __init__(
        self,
        config: ModelConfig,
        num_populations: int = 7,
        num_af_features: int = 7,
        num_variant_classes: int = 5,
        population_embedding_dim: int = 64,
        pool: str = "cls",
        embedding_type: str = "learned",
        population_labels: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_variant_classes = num_variant_classes
        self.pool = pool

        # Population-aware encoder
        self.encoder = PopulationAwareEncoder(
            config=config,
            num_populations=num_populations,
            num_af_features=num_af_features,
            population_embedding_dim=population_embedding_dim,
            embedding_type=embedding_type,
            population_labels=population_labels,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model, eps=1e-12),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, num_variant_classes),
        )

        logger.info(
            "PopulationAwareVariantPredictor: {} classes, pool={}",
            num_variant_classes,
            pool,
        )

    def _pool_hidden(
        self,
        hidden: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Pool sequence representations to a single vector.

        Args:
            hidden: ``(B, L, D)`` sequence representations.
            attention_mask: ``(B, L)`` mask.

        Returns:
            ``(B, D)`` pooled representation.
        """
        if self.pool == "cls":
            return hidden[:, 0]
        else:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return hidden.mean(dim=1)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        population_ids: Optional[Tensor] = None,
        af_features: Optional[Tensor] = None,
        af_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward pass for variant classification.

        Args:
            input_ids: ``(B, L)`` DNA sequence token ids.
            attention_mask: ``(B, L)`` attention mask.
            population_ids: ``(B,)`` population indices.
            af_features: ``(B, num_af_features)`` allele frequencies.
            af_mask: ``(B, num_af_features)`` AF validity mask.
            labels: ``(B,)`` integer class labels for training.
            segment_ids: Optional segment ids.
            output_hidden_states: Return intermediate states.

        Returns:
            Dict with ``logits`` ``(B, num_classes)``, optional ``loss``,
            optional ``probabilities``, and optional ``hidden_states``.
        """
        enc_out = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            population_ids=population_ids,
            af_features=af_features,
            af_mask=af_mask,
            segment_ids=segment_ids,
            output_hidden_states=output_hidden_states,
        )

        hidden = enc_out["last_hidden_state"]
        pooled = self._pool_hidden(hidden, attention_mask)

        logits = self.classifier(pooled)  # (B, num_classes)

        result: Dict[str, Tensor] = {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=-1),
        }

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss

        if "population_embedding" in enc_out:
            result["population_embedding"] = enc_out["population_embedding"]
        if "af_embedding" in enc_out:
            result["af_embedding"] = enc_out["af_embedding"]
        if output_hidden_states and "hidden_states" in enc_out:
            result["hidden_states"] = enc_out["hidden_states"]

        return result
