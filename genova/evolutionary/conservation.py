"""Conservation-aware predictions for evolutionary genomics.

Integrates PhyloP and PhastCons conservation scores with the Genova model,
providing conservation-weighted loss functions and evolutionary rate
estimation from multi-species embeddings.

Supports parsing conservation scores from bigWig files (via ``pyBigWig``
when available) or BED-format score files as a portable fallback.

Example::

    from genova.evolutionary.conservation import (
        ConservationScorer, ConservationWeightedLoss,
    )

    scorer = ConservationScorer(
        phylop_path="hg38.phyloP100way.bw",
        phastcons_path="hg38.phastCons100way.bw",
    )
    scores = scorer.score_region("chr1", 1000, 2000)

    loss_fn = ConservationWeightedLoss(base_loss="cross_entropy", alpha=0.5)
    loss = loss_fn(logits, targets, conservation_weights)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor


# ---------------------------------------------------------------------------
# Conservation score parser
# ---------------------------------------------------------------------------


class ConservationScorer:
    """Load and query PhyloP / PhastCons conservation scores.

    Tries to use ``pyBigWig`` for native bigWig access.  Falls back to a
    simple BED-format parser when the library is unavailable or the file
    extension is ``.bed`` / ``.bedgraph``.

    Args:
        phylop_path: Path to PhyloP score file (bigWig or BED).
        phastcons_path: Path to PhastCons score file (bigWig or BED).
        default_score: Score value used when a region has no data.
    """

    def __init__(
        self,
        phylop_path: Optional[Union[str, Path]] = None,
        phastcons_path: Optional[Union[str, Path]] = None,
        default_score: float = 0.0,
    ) -> None:
        self.phylop_path = Path(phylop_path) if phylop_path else None
        self.phastcons_path = Path(phastcons_path) if phastcons_path else None
        self.default_score = default_score

        self._phylop_handle: Any = None
        self._phastcons_handle: Any = None
        self._phylop_bed: Optional[Dict[str, np.ndarray]] = None
        self._phastcons_bed: Optional[Dict[str, np.ndarray]] = None

        self._bigwig_available = self._check_bigwig()

        if self.phylop_path is not None:
            self._open_score_file("phylop", self.phylop_path)
        if self.phastcons_path is not None:
            self._open_score_file("phastcons", self.phastcons_path)

    @staticmethod
    def _check_bigwig() -> bool:
        """Check whether pyBigWig is installed."""
        try:
            import pyBigWig  # noqa: F401

            return True
        except ImportError:
            logger.debug("pyBigWig not available; falling back to BED parsing")
            return False

    def _open_score_file(self, score_type: str, path: Path) -> None:
        """Open a score file (bigWig or BED)."""
        suffix = path.suffix.lower()
        if suffix in (".bw", ".bigwig") and self._bigwig_available:
            import pyBigWig

            handle = pyBigWig.open(str(path))
            if score_type == "phylop":
                self._phylop_handle = handle
            else:
                self._phastcons_handle = handle
            logger.info("Opened {} bigWig: {}", score_type, path)
        elif suffix in (".bed", ".bedgraph", ".tsv", ".txt"):
            bed_data = self._parse_bed(path)
            if score_type == "phylop":
                self._phylop_bed = bed_data
            else:
                self._phastcons_bed = bed_data
            logger.info(
                "Parsed {} BED file: {} ({} chromosomes)",
                score_type,
                path,
                len(bed_data),
            )
        else:
            logger.warning(
                "Unrecognised extension {} for {}; attempting BED parse",
                suffix,
                score_type,
            )
            bed_data = self._parse_bed(path)
            if score_type == "phylop":
                self._phylop_bed = bed_data
            else:
                self._phastcons_bed = bed_data

    @staticmethod
    def _parse_bed(path: Path) -> Dict[str, np.ndarray]:
        """Parse a BED/bedGraph file into per-chromosome score arrays.

        Expects tab-separated lines: ``chrom  start  end  score``.
        Builds a dense array per chromosome up to the maximum end coordinate.
        """
        from collections import defaultdict

        raw: Dict[str, List[Tuple[int, int, float]]] = defaultdict(list)
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("track"):
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                chrom = parts[0]
                start, end = int(parts[1]), int(parts[2])
                score = float(parts[3])
                raw[chrom].append((start, end, score))

        result: Dict[str, np.ndarray] = {}
        for chrom, intervals in raw.items():
            max_end = max(e for _, e, _ in intervals)
            arr = np.zeros(max_end, dtype=np.float32)
            for s, e, v in intervals:
                arr[s:e] = v
            result[chrom] = arr

        return result

    # -- querying -----------------------------------------------------------

    def score_region(
        self,
        chrom: str,
        start: int,
        end: int,
        score_type: str = "phylop",
    ) -> np.ndarray:
        """Retrieve per-base conservation scores for a genomic region.

        Args:
            chrom: Chromosome name (e.g. ``"chr1"``).
            start: 0-based start coordinate.
            end: 0-based end coordinate (exclusive).
            score_type: ``"phylop"`` or ``"phastcons"``.

        Returns:
            1-D numpy array of shape ``(end - start,)`` with per-base scores.
        """
        length = end - start

        if score_type == "phylop":
            return self._query("phylop", chrom, start, end, length)
        elif score_type == "phastcons":
            return self._query("phastcons", chrom, start, end, length)
        else:
            raise ValueError(f"Unknown score_type '{score_type}'; use 'phylop' or 'phastcons'")

    def _query(
        self,
        score_type: str,
        chrom: str,
        start: int,
        end: int,
        length: int,
    ) -> np.ndarray:
        """Internal query dispatcher."""
        handle = self._phylop_handle if score_type == "phylop" else self._phastcons_handle
        bed = self._phylop_bed if score_type == "phylop" else self._phastcons_bed

        if handle is not None:
            vals = handle.values(chrom, start, end)
            arr = np.array(vals, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=self.default_score)
            return arr
        elif bed is not None and chrom in bed:
            chrom_arr = bed[chrom]
            safe_start = min(start, len(chrom_arr))
            safe_end = min(end, len(chrom_arr))
            result = np.full(length, self.default_score, dtype=np.float32)
            valid_len = safe_end - safe_start
            if valid_len > 0:
                result[:valid_len] = chrom_arr[safe_start:safe_end]
            return result
        else:
            logger.debug(
                "No {} data for {}:{}-{}; returning default",
                score_type,
                chrom,
                start,
                end,
            )
            return np.full(length, self.default_score, dtype=np.float32)

    def score_batch(
        self,
        regions: List[Tuple[str, int, int]],
        score_type: str = "phylop",
        pad_value: float = 0.0,
    ) -> Tensor:
        """Score a batch of regions, padding to uniform length.

        Args:
            regions: List of ``(chrom, start, end)`` tuples.
            score_type: ``"phylop"`` or ``"phastcons"``.
            pad_value: Value used for padding shorter regions.

        Returns:
            Float tensor of shape ``(B, max_len)`` with conservation scores.
        """
        arrays = [self.score_region(c, s, e, score_type) for c, s, e in regions]
        max_len = max(len(a) for a in arrays)
        padded = np.full((len(arrays), max_len), pad_value, dtype=np.float32)
        for i, a in enumerate(arrays):
            padded[i, : len(a)] = a
        return torch.from_numpy(padded)

    def close(self) -> None:
        """Close any open bigWig file handles."""
        for handle in (self._phylop_handle, self._phastcons_handle):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
        self._phylop_handle = None
        self._phastcons_handle = None


# ---------------------------------------------------------------------------
# Conservation-weighted loss
# ---------------------------------------------------------------------------


class ConservationWeightedLoss(nn.Module):
    """Loss function that up-weights predictions at conserved positions.

    Combines a base loss with per-position conservation weights so that
    errors at evolutionarily conserved sites incur a larger penalty.

    The effective weight for position *i* is::

        w_i = 1.0 + alpha * conservation_i

    Args:
        base_loss: ``"cross_entropy"`` or ``"mse"``.
        alpha: Scaling factor controlling how strongly conservation
            modulates the loss.  ``alpha=0`` recovers the uniform loss.
        min_weight: Floor for per-position weights to prevent zero weighting.
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(
        self,
        base_loss: str = "cross_entropy",
        alpha: float = 0.5,
        min_weight: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if base_loss not in ("cross_entropy", "mse"):
            raise ValueError(f"base_loss must be 'cross_entropy' or 'mse', got '{base_loss}'")
        self.base_loss = base_loss
        self.alpha = alpha
        self.min_weight = min_weight
        self.reduction = reduction

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        conservation_weights: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute conservation-weighted loss.

        Args:
            predictions: Model output.
                For cross-entropy: ``(B, L, V)`` logits.
                For MSE: ``(B, L)`` or ``(B, L, D)`` predictions.
            targets: Ground-truth labels.
                For cross-entropy: ``(B, L)`` integer class ids.
                For MSE: same shape as *predictions*.
            conservation_weights: ``(B, L)`` per-position conservation scores
                (e.g. PhyloP values, typically in ``[0, 1]``).
            attention_mask: ``(B, L)`` binary mask (1 = real, 0 = pad).

        Returns:
            Scalar loss tensor.
        """
        # Compute effective weights
        weights = (1.0 + self.alpha * conservation_weights).clamp(min=self.min_weight)

        if attention_mask is not None:
            weights = weights * attention_mask.to(weights.dtype)

        if self.base_loss == "cross_entropy":
            # predictions: (B, L, V) -> (B*L, V);  targets: (B, L) -> (B*L)
            B, L, V = predictions.shape
            logits_flat = predictions.reshape(-1, V)
            targets_flat = targets.reshape(-1)
            weights_flat = weights.reshape(-1)

            per_token_loss = F.cross_entropy(
                logits_flat, targets_flat, reduction="none"
            )
            weighted_loss = per_token_loss * weights_flat

        else:  # mse
            per_pos_loss = F.mse_loss(predictions, targets, reduction="none")
            if per_pos_loss.dim() > 2:
                per_pos_loss = per_pos_loss.mean(dim=-1)  # reduce feature dim
            weighted_loss = per_pos_loss * weights

        if self.reduction == "mean":
            denom = weights.sum().clamp(min=1.0) if attention_mask is not None else weighted_loss.numel()
            return weighted_loss.sum() / denom
        return weighted_loss.sum()

    def extra_repr(self) -> str:
        return (
            f"base_loss={self.base_loss}, alpha={self.alpha}, "
            f"min_weight={self.min_weight}, reduction={self.reduction}"
        )


# ---------------------------------------------------------------------------
# Evolutionary rate estimation
# ---------------------------------------------------------------------------


class EvolutionaryRateEstimator:
    """Estimate per-position evolutionary rates from multi-species embeddings.

    Given embeddings for homologous regions from multiple species, estimates
    the rate of sequence evolution at each position by measuring embedding
    divergence across species.

    Higher divergence indicates faster evolution (less constraint); lower
    divergence indicates stronger purifying selection.
    """

    def __init__(self, metric: str = "cosine") -> None:
        """
        Args:
            metric: Distance metric between embeddings.
                ``"cosine"`` (1 - cosine similarity) or ``"euclidean"``.
        """
        if metric not in ("cosine", "euclidean"):
            raise ValueError(f"metric must be 'cosine' or 'euclidean', got '{metric}'")
        self.metric = metric

    def estimate_rates(
        self,
        embeddings: Dict[str, Tensor],
        reference_species: str = "human",
    ) -> Dict[str, Tensor]:
        """Estimate evolutionary rates from multi-species embeddings.

        Args:
            embeddings: Mapping from species name to tensor of shape
                ``(L, D)`` representing aligned position embeddings.
            reference_species: Species to use as the reference for pairwise
                distance computation.

        Returns:
            Dict with keys:
                - ``"pairwise_rates"``: dict mapping species pairs to
                  ``(L,)`` per-position distance tensors.
                - ``"mean_rate"``: ``(L,)`` mean rate across all pairs.
                - ``"rate_variance"``: ``(L,)`` variance of rates.
        """
        ref_emb = embeddings[reference_species]  # (L, D)
        other_species = [s for s in embeddings if s != reference_species]

        pairwise: Dict[str, Tensor] = {}
        all_rates: List[Tensor] = []

        for species in other_species:
            sp_emb = embeddings[species]  # (L, D)
            dist = self._pairwise_distance(ref_emb, sp_emb)  # (L,)
            pair_key = f"{reference_species}_vs_{species}"
            pairwise[pair_key] = dist
            all_rates.append(dist)

        if all_rates:
            stacked = torch.stack(all_rates, dim=0)  # (S, L)
            mean_rate = stacked.mean(dim=0)
            rate_var = stacked.var(dim=0) if len(all_rates) > 1 else torch.zeros_like(mean_rate)
        else:
            L = ref_emb.size(0)
            mean_rate = torch.zeros(L, device=ref_emb.device)
            rate_var = torch.zeros(L, device=ref_emb.device)

        return {
            "pairwise_rates": pairwise,
            "mean_rate": mean_rate,
            "rate_variance": rate_var,
        }

    def _pairwise_distance(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute per-position distance between two aligned embedding sets.

        Args:
            a: ``(L, D)`` reference embeddings.
            b: ``(L, D)`` comparison embeddings.

        Returns:
            ``(L,)`` per-position distance.
        """
        if self.metric == "cosine":
            a_norm = F.normalize(a, dim=-1)
            b_norm = F.normalize(b, dim=-1)
            similarity = (a_norm * b_norm).sum(dim=-1)
            return 1.0 - similarity
        else:  # euclidean
            return torch.norm(a - b, dim=-1)

    def identify_conserved_regions(
        self,
        mean_rate: Tensor,
        threshold: float = 0.1,
        min_length: int = 10,
    ) -> List[Tuple[int, int]]:
        """Identify contiguous conserved regions (low evolutionary rate).

        Args:
            mean_rate: ``(L,)`` mean evolutionary rate per position.
            threshold: Positions with rate below this are considered conserved.
            min_length: Minimum length of a conserved region to report.

        Returns:
            List of ``(start, end)`` tuples for conserved regions.
        """
        conserved_mask = (mean_rate < threshold).cpu().numpy()
        regions: List[Tuple[int, int]] = []
        start: Optional[int] = None

        for i, is_conserved in enumerate(conserved_mask):
            if is_conserved and start is None:
                start = i
            elif not is_conserved and start is not None:
                if i - start >= min_length:
                    regions.append((start, i))
                start = None

        # handle region extending to the end
        if start is not None and len(conserved_mask) - start >= min_length:
            regions.append((start, len(conserved_mask)))

        return regions
