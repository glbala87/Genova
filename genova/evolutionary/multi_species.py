"""Multi-species training for cross-species genomic modelling.

Provides infrastructure for training a single Genova backbone on genomic
sequences from multiple species (human, mouse, zebrafish, primates, etc.)
with per-species embedding layers and a shared representation space.

Pre-computed alignments in MAF or pairwise format are accepted through the
:class:`HomologousAlignmentLoader` interface, enabling the model to learn
conserved cross-species features.

Example::

    from genova.utils.config import ModelConfig
    from genova.evolutionary.multi_species import (
        SpeciesConfig, MultiSpeciesEncoder,
    )

    species = [
        SpeciesConfig(name="human", genome_path="/data/hg38.fa", weight=1.0),
        SpeciesConfig(name="mouse", genome_path="/data/mm10.fa", weight=0.8),
    ]
    cfg = ModelConfig(d_model=256, n_heads=4, n_layers=6, d_ff=1024,
                      vocab_size=4096)
    encoder = MultiSpeciesEncoder(cfg, species_configs=species)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
# Species configuration
# ---------------------------------------------------------------------------


@dataclass
class SpeciesConfig:
    """Configuration for a single species in multi-species training.

    Attributes:
        name: Short identifier such as ``"human"``, ``"mouse"``.
        genome_path: Filesystem path to the reference FASTA.
        weight: Relative sampling weight during training.  Higher values
            cause the species to be sampled more frequently.
        taxonomy_id: Optional NCBI taxonomy identifier.
        metadata: Arbitrary extra metadata for the species.
    """

    name: str = "human"
    genome_path: str = ""
    weight: float = 1.0
    taxonomy_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Homologous alignment loader
# ---------------------------------------------------------------------------


class HomologousAlignmentLoader:
    """Interface for loading pre-computed homologous region alignments.

    Supports MAF (Multiple Alignment Format) and simple pairwise TSV files
    with columns: ``species_a, chrom_a, start_a, end_a, species_b, chrom_b,
    start_b, end_b, score``.

    Args:
        alignment_path: Path to MAF or pairwise alignment file.
        format: ``"maf"`` or ``"pairwise"``.  Auto-detected from extension
            when ``None``.
    """

    def __init__(
        self,
        alignment_path: Union[str, Path],
        format: Optional[str] = None,
    ) -> None:
        self.alignment_path = Path(alignment_path)
        if format is None:
            suffix = self.alignment_path.suffix.lower()
            self.format = "maf" if suffix == ".maf" else "pairwise"
        else:
            self.format = format
        self._alignments: List[Dict[str, Any]] = []
        logger.info(
            "HomologousAlignmentLoader initialised: path={}, format={}",
            self.alignment_path,
            self.format,
        )

    # ---- parsing -----------------------------------------------------------

    def load(self) -> List[Dict[str, Any]]:
        """Parse the alignment file and return a list of alignment records.

        Returns:
            List of dicts, each with keys ``species``, ``chrom``, ``start``,
            ``end``, ``strand``, ``sequence`` (for MAF) or ``species_a``,
            ``species_b``, etc. for pairwise.
        """
        if self.format == "maf":
            return self._parse_maf()
        return self._parse_pairwise()

    def _parse_maf(self) -> List[Dict[str, Any]]:
        """Parse a MAF alignment file into block records."""
        alignments: List[Dict[str, Any]] = []
        current_block: Dict[str, Any] = {}
        current_seqs: List[Dict[str, str]] = []

        with open(self.alignment_path, "r") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("#") or not line.strip():
                    continue
                if line.startswith("a"):
                    # start of a new alignment block
                    if current_seqs:
                        current_block["sequences"] = current_seqs
                        alignments.append(current_block)
                    current_block = self._parse_maf_header(line)
                    current_seqs = []
                elif line.startswith("s"):
                    current_seqs.append(self._parse_maf_s_line(line))

        # flush last block
        if current_seqs:
            current_block["sequences"] = current_seqs
            alignments.append(current_block)

        self._alignments = alignments
        logger.info("Parsed {} MAF alignment blocks", len(alignments))
        return alignments

    @staticmethod
    def _parse_maf_header(line: str) -> Dict[str, Any]:
        """Extract key-value pairs from a MAF ``a`` line."""
        parts = line.split()
        block: Dict[str, Any] = {}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                block[k] = v
        return block

    @staticmethod
    def _parse_maf_s_line(line: str) -> Dict[str, str]:
        """Parse a MAF ``s`` (sequence) line."""
        fields = line.split()
        return {
            "src": fields[1],
            "start": fields[2],
            "size": fields[3],
            "strand": fields[4],
            "src_size": fields[5],
            "text": fields[6] if len(fields) > 6 else "",
        }

    def _parse_pairwise(self) -> List[Dict[str, Any]]:
        """Parse a pairwise TSV alignment file."""
        alignments: List[Dict[str, Any]] = []
        with open(self.alignment_path, "r") as fh:
            header = None
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if header is None:
                    header = parts
                    continue
                record = dict(zip(header, parts))
                alignments.append(record)

        self._alignments = alignments
        logger.info("Parsed {} pairwise alignment records", len(alignments))
        return alignments

    def get_homologous_pairs(
        self,
        species_a: str,
        species_b: str,
    ) -> List[Dict[str, Any]]:
        """Filter loaded alignments to pairs between two species.

        Args:
            species_a: First species identifier.
            species_b: Second species identifier.

        Returns:
            Filtered list of alignment records involving both species.
        """
        if not self._alignments:
            self.load()

        if self.format == "maf":
            results: List[Dict[str, Any]] = []
            for block in self._alignments:
                seqs = block.get("sequences", [])
                species_in_block = {s["src"].split(".")[0] for s in seqs}
                if species_a in species_in_block and species_b in species_in_block:
                    results.append(block)
            return results

        # pairwise format
        return [
            r
            for r in self._alignments
            if (
                r.get("species_a") == species_a and r.get("species_b") == species_b
            )
            or (
                r.get("species_a") == species_b and r.get("species_b") == species_a
            )
        ]


# ---------------------------------------------------------------------------
# Species embedding layer
# ---------------------------------------------------------------------------


class SpeciesEmbedding(nn.Module):
    """Learnable per-species embedding added to token representations.

    Args:
        num_species: Number of species in the training set.
        d_model: Model hidden dimension.
    """

    def __init__(self, num_species: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_species, d_model)
        self.num_species = num_species
        self.d_model = d_model

    def forward(self, species_ids: Tensor) -> Tensor:
        """Return species embeddings.

        Args:
            species_ids: Integer tensor of shape ``(B,)`` or ``(B, 1)``
                identifying the species for each sample in the batch.

        Returns:
            ``(B, 1, d_model)`` tensor broadcastable over sequence length.
        """
        if species_ids.dim() == 1:
            species_ids = species_ids.unsqueeze(1)
        return self.embedding(species_ids)  # (B, 1, d_model)


# ---------------------------------------------------------------------------
# Multi-species encoder
# ---------------------------------------------------------------------------


class MultiSpeciesEncoder(nn.Module):
    """Wraps a Genova backbone with per-species embeddings to learn a shared
    representation space across multiple species.

    The forward pass accepts a ``species_ids`` tensor alongside the usual
    ``input_ids`` so that species-specific offsets are added to the token
    representations before they enter the backbone.

    Args:
        config: Model configuration determining the backbone architecture.
        species_configs: List of :class:`SpeciesConfig` for every species
            included in training.
        backbone: Optional pre-instantiated backbone module.  When ``None``,
            the backbone is constructed from *config.arch*.
        pooling: Sequence pooling strategy (``"mean"`` or ``"cls"``).
        shared_projection_dim: If set, a linear projection maps the pooled
            backbone output into a shared space of this dimensionality.
    """

    def __init__(
        self,
        config: ModelConfig,
        species_configs: Sequence[SpeciesConfig],
        backbone: Optional[nn.Module] = None,
        pooling: str = "mean",
        shared_projection_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.pooling = pooling

        # -- species registry -----------------------------------------------
        self.species_configs = list(species_configs)
        self.species_to_id: Dict[str, int] = {
            sc.name: idx for idx, sc in enumerate(self.species_configs)
        }
        num_species = len(self.species_configs)
        logger.info(
            "MultiSpeciesEncoder: {} species registered: {}",
            num_species,
            [sc.name for sc in self.species_configs],
        )

        # -- species embedding layer ----------------------------------------
        self.species_embedding = SpeciesEmbedding(num_species, config.d_model)

        # -- backbone -------------------------------------------------------
        if backbone is not None:
            self.backbone = backbone
        elif config.arch.lower() in ("transformer", "bert"):
            self.backbone = GenovaTransformer(config)
        elif config.arch.lower() in ("mamba", "ssm"):
            self.backbone = GenovaMamba(config)
        else:
            raise ValueError(
                f"Unsupported arch '{config.arch}'. Use 'transformer' or 'mamba'."
            )

        # -- shared projection (optional) -----------------------------------
        self.shared_projection: Optional[nn.Linear] = None
        if shared_projection_dim is not None:
            self.shared_projection = nn.Linear(
                config.d_model, shared_projection_dim
            )

        # -- sampling weights for dataloader --------------------------------
        total_w = sum(sc.weight for sc in self.species_configs)
        self._sampling_weights = [sc.weight / total_w for sc in self.species_configs]

    # -- helpers -------------------------------------------------------------

    def species_id_tensor(
        self,
        species_names: Sequence[str],
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Convert a batch of species name strings to an id tensor.

        Args:
            species_names: Sequence of species name strings of length ``B``.
            device: Target device for the returned tensor.

        Returns:
            ``(B,)`` integer tensor of species ids.
        """
        ids = [self.species_to_id[n] for n in species_names]
        return torch.tensor(ids, dtype=torch.long, device=device)

    @property
    def sampling_weights(self) -> List[float]:
        """Normalised sampling weights for each species."""
        return self._sampling_weights

    # -- pooling -------------------------------------------------------------

    def _pool(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Pool sequence-level representations into a single vector.

        Args:
            hidden_states: ``(B, L, D)`` encoder outputs.
            attention_mask: ``(B, L)`` mask (1 = real token, 0 = pad).

        Returns:
            ``(B, D)`` pooled representation.
        """
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        # mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return hidden_states.mean(dim=1)

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        species_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_pooled: bool = False,
    ) -> Dict[str, Tensor]:
        """Encode sequences with species-aware embeddings.

        Args:
            input_ids: ``(B, L)`` token ids.
            species_ids: ``(B,)`` integer species identifiers.
            attention_mask: ``(B, L)`` attention mask.
            return_pooled: If ``True``, include a ``"pooled"`` key with
                mean- or cls-pooled representations.

        Returns:
            Dict with at least ``"last_hidden_state"`` of shape ``(B, L, D)``.
            Optionally ``"pooled"`` of shape ``(B, D)`` or ``(B, proj_dim)``.
        """
        # Get backbone output
        backbone_out = self.backbone(input_ids, attention_mask=attention_mask)
        hidden_states = backbone_out["last_hidden_state"]  # (B, L, D)

        # Add species embedding (broadcast over sequence length)
        species_emb = self.species_embedding(species_ids)  # (B, 1, D)
        hidden_states = hidden_states + species_emb

        result: Dict[str, Tensor] = {"last_hidden_state": hidden_states}

        if return_pooled:
            pooled = self._pool(hidden_states, attention_mask)
            if self.shared_projection is not None:
                pooled = self.shared_projection(pooled)
            result["pooled"] = pooled

        return result

    def get_species_embeddings(self) -> Tensor:
        """Return the species embedding weight matrix.

        Returns:
            ``(num_species, d_model)`` tensor.
        """
        return self.species_embedding.embedding.weight.data

    def compute_species_similarity(self) -> Tensor:
        """Compute cosine similarity between all species embeddings.

        Returns:
            ``(num_species, num_species)`` similarity matrix.
        """
        emb = F.normalize(self.get_species_embeddings(), dim=-1)
        return emb @ emb.T

    def compute_alignment_loss(
        self,
        hidden_a: Tensor,
        hidden_b: Tensor,
        alignment_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute a contrastive alignment loss between homologous regions.

        Encourages embeddings of aligned (homologous) positions from
        different species to be similar in the shared space.

        Args:
            hidden_a: ``(B, L, D)`` hidden states for species A regions.
            hidden_b: ``(B, L, D)`` hidden states for species B regions.
            alignment_mask: ``(B, L)`` binary mask indicating aligned
                positions.  If ``None``, all positions are used.

        Returns:
            Scalar MSE loss between aligned representations.
        """
        if alignment_mask is not None:
            mask = alignment_mask.unsqueeze(-1).to(hidden_a.dtype)
            hidden_a = hidden_a * mask
            hidden_b = hidden_b * mask
            n_aligned = mask.sum().clamp(min=1.0)
            loss = ((hidden_a - hidden_b) ** 2).sum() / (n_aligned * hidden_a.size(-1))
        else:
            loss = F.mse_loss(hidden_a, hidden_b)
        return loss
