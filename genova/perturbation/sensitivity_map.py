"""Sensitivity analysis for identifying putative functional genomic elements.

Builds per-position sensitivity maps from in-silico perturbation results,
identifies contiguous sensitive regions, and performs statistical
significance testing against a random mutation background.

Example::

    mapper = SensitivityMapper(model, tokenizer, device="cuda")
    smap = mapper.build_map("ACGT" * 500)
    regions = mapper.find_sensitive_regions(smap, min_length=10)
    pvalues = mapper.significance_test(smap, n_permutations=1000)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from scipy import stats as sp_stats

from genova.data.tokenizer import GenomicTokenizer
from genova.perturbation.variant_simulator import VariantSimulator

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SensitiveRegion:
    """A contiguous genomic region with elevated sensitivity.

    Attributes:
        start: 0-based start position (inclusive).
        end: End position (exclusive).
        mean_score: Average sensitivity score across the region.
        max_score: Peak sensitivity score.
        peak_position: Position of the peak score.
        p_value: Statistical significance (if computed).
    """

    start: int
    end: int
    mean_score: float = 0.0
    max_score: float = 0.0
    peak_position: int = 0
    p_value: float = 1.0

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class SensitivityMap:
    """Per-position sensitivity scores for a sequence.

    Attributes:
        sequence: The analysed DNA sequence.
        scores: ``(L,)`` array of sensitivity scores.
        mean_score: Global mean sensitivity.
        std_score: Global standard deviation.
        threshold: Threshold used for region detection.
    """

    sequence: str
    scores: np.ndarray
    mean_score: float = 0.0
    std_score: float = 0.0
    threshold: float = 0.0


# ---------------------------------------------------------------------------
# SensitivityMapper
# ---------------------------------------------------------------------------


class SensitivityMapper:
    """Build sensitivity maps and identify functional elements.

    Uses :class:`VariantSimulator` internally to perform saturation
    mutagenesis, then aggregates per-position effects into a sensitivity
    map.

    Args:
        model: Trained Genova model.
        tokenizer: Tokenizer with built vocabulary.
        device: Inference device.
        batch_size: Batch size for variant inference.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: GenomicTokenizer,
        *,
        device: Union[str, torch.device] = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.batch_size = batch_size

        self.simulator = VariantSimulator(
            model, tokenizer, device=device, batch_size=batch_size
        )

        logger.info(
            "SensitivityMapper initialised (device={}, batch_size={}).",
            self.device,
            batch_size,
        )

    # ------------------------------------------------------------------
    # Map construction
    # ------------------------------------------------------------------

    def build_map(
        self,
        sequence: str,
        *,
        max_length: Optional[int] = None,
        region: Optional[Tuple[int, int]] = None,
        metric: str = "l2",
        aggregation: str = "max",
    ) -> SensitivityMap:
        """Build a per-position sensitivity map via saturation mutagenesis.

        Args:
            sequence: DNA sequence to analyse.
            max_length: Maximum token length for model.
            region: Optional ``(start, end)`` to restrict analysis.
            metric: Effect metric: ``"l2"``, ``"cosine"``, or ``"effect_size"``.
            aggregation: How to aggregate across alternative alleles at
                each position: ``"max"``, ``"mean"``, or ``"sum"``.

        Returns:
            A :class:`SensitivityMap` with per-position scores.
        """
        effects = self.simulator.saturate_snps(
            sequence, max_length=max_length, region=region
        )

        seq_len = len(sequence)
        position_values: Dict[int, List[float]] = {}

        for eff in effects:
            if metric == "cosine":
                val = 1.0 - eff.cosine_similarity
            elif metric == "effect_size":
                val = eff.effect_size
            else:
                val = eff.l2_distance

            position_values.setdefault(eff.position, []).append(val)

        scores = np.zeros(seq_len, dtype=np.float64)
        for pos, vals in position_values.items():
            if 0 <= pos < seq_len:
                arr = np.array(vals)
                if aggregation == "mean":
                    scores[pos] = float(arr.mean())
                elif aggregation == "sum":
                    scores[pos] = float(arr.sum())
                else:  # max
                    scores[pos] = float(arr.max())

        mean_score = float(scores.mean())
        std_score = float(scores.std())
        threshold = mean_score + 2.0 * std_score

        smap = SensitivityMap(
            sequence=sequence,
            scores=scores,
            mean_score=mean_score,
            std_score=std_score,
            threshold=threshold,
        )

        logger.info(
            "Built sensitivity map: mean={:.4f}, std={:.4f}, threshold={:.4f}.",
            mean_score,
            std_score,
            threshold,
        )
        return smap

    # ------------------------------------------------------------------
    # Region detection
    # ------------------------------------------------------------------

    def find_sensitive_regions(
        self,
        sensitivity_map: SensitivityMap,
        *,
        threshold: Optional[float] = None,
        min_length: int = 5,
        merge_distance: int = 3,
    ) -> List[SensitiveRegion]:
        """Identify contiguous regions with elevated sensitivity.

        Scans the sensitivity map for runs of positions above the
        threshold, merges nearby regions, and filters by minimum length.

        Args:
            sensitivity_map: Output from :meth:`build_map`.
            threshold: Score threshold.  Defaults to the map's own
                ``mean + 2 * std`` threshold.
            min_length: Minimum region length to report.
            merge_distance: Maximum gap between regions to merge.

        Returns:
            List of :class:`SensitiveRegion` objects sorted by
            descending mean score.
        """
        scores = sensitivity_map.scores
        if threshold is None:
            threshold = sensitivity_map.threshold

        # Find positions above threshold
        above = np.where(scores >= threshold)[0]
        if len(above) == 0:
            logger.info("No positions exceed threshold {:.4f}.", threshold)
            return []

        # Group into contiguous runs (allowing merge_distance gaps)
        raw_regions: List[Tuple[int, int]] = []
        region_start = int(above[0])
        region_end = int(above[0]) + 1

        for pos in above[1:]:
            pos_int = int(pos)
            if pos_int <= region_end + merge_distance:
                region_end = pos_int + 1
            else:
                raw_regions.append((region_start, region_end))
                region_start = pos_int
                region_end = pos_int + 1
        raw_regions.append((region_start, region_end))

        # Filter by min_length and compute stats
        regions: List[SensitiveRegion] = []
        for start, end in raw_regions:
            if end - start < min_length:
                continue

            region_scores = scores[start:end]
            peak_idx = int(np.argmax(region_scores))

            regions.append(
                SensitiveRegion(
                    start=start,
                    end=end,
                    mean_score=float(region_scores.mean()),
                    max_score=float(region_scores.max()),
                    peak_position=start + peak_idx,
                )
            )

        regions.sort(key=lambda r: r.mean_score, reverse=True)

        logger.info(
            "Found {} sensitive regions (threshold={:.4f}, min_length={}).",
            len(regions),
            threshold,
            min_length,
        )
        return regions

    # ------------------------------------------------------------------
    # Statistical significance
    # ------------------------------------------------------------------

    def significance_test(
        self,
        sensitivity_map: SensitivityMap,
        *,
        n_permutations: int = 1000,
        method: str = "permutation",
        regions: Optional[List[SensitiveRegion]] = None,
    ) -> Dict[str, Any]:
        """Test statistical significance of sensitivity scores.

        Compares observed sensitivity scores against a null distribution
        generated by random permutation of position labels.

        Args:
            sensitivity_map: Output from :meth:`build_map`.
            n_permutations: Number of permutations for the null.
            method: ``"permutation"`` (position-label shuffling) or
                ``"zscore"`` (parametric z-test against global distribution).
            regions: If provided, compute p-values for these specific
                regions; otherwise compute per-position p-values.

        Returns:
            Dict with:
                - ``p_values``: Per-position ``(L,)`` array or per-region
                  list of p-values.
                - ``significant_positions``: Positions with
                  Bonferroni-corrected p < 0.05.
                - ``null_mean``: Mean of the null distribution.
                - ``null_std``: Std of the null distribution.
                - ``method``: Method used.
        """
        scores = sensitivity_map.scores
        n = len(scores)

        if method == "zscore":
            return self._zscore_test(scores, regions)

        # Permutation test
        rng = np.random.default_rng(42)

        if regions is not None:
            return self._region_permutation_test(
                scores, regions, n_permutations, rng
            )

        # Per-position permutation test
        null_maxima = np.zeros(n_permutations, dtype=np.float64)
        for perm_idx in range(n_permutations):
            shuffled = rng.permutation(scores)
            null_maxima[perm_idx] = shuffled.max()

        # p-value: fraction of permutations where max >= observed
        p_values = np.zeros(n, dtype=np.float64)
        for pos in range(n):
            p_values[pos] = float(np.mean(null_maxima >= scores[pos]))

        # Bonferroni correction
        corrected = np.minimum(p_values * n, 1.0)
        significant = np.where(corrected < 0.05)[0].tolist()

        null_mean = float(null_maxima.mean())
        null_std = float(null_maxima.std())

        logger.info(
            "Permutation test ({} permutations): {} significant positions "
            "(Bonferroni p < 0.05).",
            n_permutations,
            len(significant),
        )

        return {
            "p_values": p_values,
            "significant_positions": significant,
            "null_mean": null_mean,
            "null_std": null_std,
            "method": "permutation",
        }

    def _zscore_test(
        self,
        scores: np.ndarray,
        regions: Optional[List[SensitiveRegion]],
    ) -> Dict[str, Any]:
        """Parametric z-test assuming normal distribution of scores."""
        mean = float(scores.mean())
        std = float(scores.std())

        if std < 1e-12:
            p_values = np.ones(len(scores), dtype=np.float64)
        else:
            z_scores = (scores - mean) / std
            # One-sided p-value (right tail)
            p_values = 1.0 - sp_stats.norm.cdf(z_scores)

        if regions is not None:
            region_pvals: List[float] = []
            for reg in regions:
                region_scores = scores[reg.start : reg.end]
                if std < 1e-12:
                    region_pvals.append(1.0)
                else:
                    z = (float(region_scores.mean()) - mean) / (
                        std / np.sqrt(max(1, reg.length))
                    )
                    region_pvals.append(float(1.0 - sp_stats.norm.cdf(z)))
                    reg.p_value = region_pvals[-1]

            return {
                "p_values": region_pvals,
                "significant_positions": [],
                "null_mean": mean,
                "null_std": std,
                "method": "zscore",
            }

        # Bonferroni correction
        n = len(scores)
        corrected = np.minimum(p_values * n, 1.0)
        significant = np.where(corrected < 0.05)[0].tolist()

        return {
            "p_values": p_values,
            "significant_positions": significant,
            "null_mean": mean,
            "null_std": std,
            "method": "zscore",
        }

    def _region_permutation_test(
        self,
        scores: np.ndarray,
        regions: List[SensitiveRegion],
        n_permutations: int,
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """Permutation test for region-level significance."""
        region_pvals: List[float] = []

        for reg in regions:
            observed_mean = float(scores[reg.start : reg.end].mean())
            count_above = 0

            for _ in range(n_permutations):
                # Random region of same length
                max_start = len(scores) - reg.length
                if max_start <= 0:
                    rand_mean = float(scores.mean())
                else:
                    rand_start = rng.integers(0, max_start + 1)
                    rand_mean = float(
                        scores[rand_start : rand_start + reg.length].mean()
                    )
                if rand_mean >= observed_mean:
                    count_above += 1

            pval = (count_above + 1) / (n_permutations + 1)
            reg.p_value = pval
            region_pvals.append(pval)

        null_mean = float(scores.mean())
        null_std = float(scores.std())

        significant_regions = [
            i for i, p in enumerate(region_pvals) if p < 0.05
        ]

        logger.info(
            "Region permutation test: {} / {} regions significant (p < 0.05).",
            len(significant_regions),
            len(regions),
        )

        return {
            "p_values": region_pvals,
            "significant_positions": significant_regions,
            "null_mean": null_mean,
            "null_std": null_std,
            "method": "permutation",
        }
