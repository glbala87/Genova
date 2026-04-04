"""ONT bedMethyl file processing for Genova multi-omics integration.

Parses Oxford Nanopore Technology (ONT) bedMethyl output files, extracts
methylation signals per genomic window, and aggregates at CpG site level.

bedMethyl format (modkit output):
    chrom, start, end, name, score, strand, start_thick, end_thick,
    colour, coverage, percent_modified

Example::

    processor = ONTMethylationProcessor(window_size=512)
    features = processor.process_file("sample.bed")
    window_feats = processor.get_window_features("chr1", 10000, 10512)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import Tensor


# bedMethyl column names following modkit conventions
_BEDMETHYL_COLUMNS: List[str] = [
    "chrom",
    "start",
    "end",
    "name",
    "score",
    "strand",
    "start_thick",
    "end_thick",
    "colour",
    "coverage",
    "percent_modified",
]

# Minimum coverage threshold to consider a CpG site reliable
_DEFAULT_MIN_COVERAGE: int = 5


class ONTMethylationProcessor:
    """Process ONT bedMethyl files for multi-omics integration.

    Reads bedMethyl output (typically from ``modkit``), filters by coverage
    and modification type, and provides methods to extract methylation
    features for arbitrary genomic windows.

    Internally maintains a position-indexed data structure for efficient
    window-based queries.

    Args:
        window_size: Default genomic window size in base pairs for
            feature extraction.
        min_coverage: Minimum read coverage to include a CpG site.
        modification_name: Name of the modification to filter for in
            the ``name`` column (e.g. ``"m"`` for 5mC, ``"h"`` for 5hmC).
            If ``None``, all modifications are included.
        aggregate_strands: If ``True``, merge forward and reverse strand
            measurements at each CpG position.
    """

    def __init__(
        self,
        window_size: int = 512,
        min_coverage: int = _DEFAULT_MIN_COVERAGE,
        modification_name: Optional[str] = "m",
        aggregate_strands: bool = True,
    ) -> None:
        self.window_size = window_size
        self.min_coverage = min_coverage
        self.modification_name = modification_name
        self.aggregate_strands = aggregate_strands

        # Indexed data: chrom -> sorted array of (position, beta_value, coverage)
        self._data: Dict[str, np.ndarray] = {}
        self._loaded: bool = False

        logger.debug(
            "ONTMethylationProcessor: window={}, min_cov={}, mod={}",
            window_size,
            min_coverage,
            modification_name,
        )

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def process_file(
        self,
        path: Union[str, Path],
        sep: str = "\t",
        has_header: bool = False,
    ) -> Dict[str, int]:
        """Load and index a bedMethyl file.

        Args:
            path: Path to the bedMethyl file.
            sep: Field separator (default tab).
            has_header: Whether the file has a header row.

        Returns:
            Dictionary with summary statistics:
                - ``total_sites``: total CpG sites loaded.
                - ``filtered_sites``: sites passing coverage filter.
                - ``chromosomes``: number of distinct chromosomes.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"bedMethyl file not found: {path}")

        logger.info("Loading bedMethyl from {}", path)

        header = 0 if has_header else None
        df = pd.read_csv(
            path,
            sep=sep,
            header=header,
            names=None if has_header else _BEDMETHYL_COLUMNS[:11],
            comment="#",
            low_memory=False,
        )

        # If file has more columns than expected, trim or adapt
        if len(df.columns) > 11 and not has_header:
            df = df.iloc[:, :11]
            df.columns = _BEDMETHYL_COLUMNS[:11]

        # Ensure required columns
        required = {"chrom", "start", "coverage", "percent_modified"}
        available = set(df.columns)
        missing = required - available
        if missing:
            raise ValueError(
                f"bedMethyl file missing required columns: {missing}. "
                f"Available: {sorted(available)}"
            )

        # Filter by modification type
        if self.modification_name is not None and "name" in df.columns:
            df = df[df["name"].str.contains(self.modification_name, na=False)]

        total_sites = len(df)

        # Filter by coverage
        df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce").fillna(0)
        df = df[df["coverage"] >= self.min_coverage].copy()

        # Compute beta values (0-1 scale)
        df["percent_modified"] = pd.to_numeric(
            df["percent_modified"], errors="coerce"
        ).fillna(0.0)
        df["beta"] = df["percent_modified"] / 100.0
        df["beta"] = df["beta"].clip(0.0, 1.0)

        # Aggregate strands if requested
        if self.aggregate_strands and "strand" in df.columns:
            df = (
                df.groupby(["chrom", "start"])
                .agg(
                    beta=("beta", "mean"),
                    coverage=("coverage", "sum"),
                )
                .reset_index()
            )

        # Build indexed data structure
        self._data.clear()
        for chrom, group in df.groupby("chrom"):
            positions = group["start"].values.astype(np.int64)
            betas = group["beta"].values.astype(np.float32)
            coverages = group["coverage"].values.astype(np.float32)

            # Sort by position for efficient windowed queries
            sort_idx = np.argsort(positions)
            self._data[str(chrom)] = np.column_stack([
                positions[sort_idx],
                betas[sort_idx],
                coverages[sort_idx],
            ])

        self._loaded = True
        filtered_sites = sum(len(v) for v in self._data.values())

        stats = {
            "total_sites": total_sites,
            "filtered_sites": filtered_sites,
            "chromosomes": len(self._data),
        }

        logger.info(
            "Loaded {} CpG sites ({} after filtering) across {} chromosomes",
            total_sites,
            filtered_sites,
            len(self._data),
        )

        return stats

    # ------------------------------------------------------------------
    # Window-based feature extraction
    # ------------------------------------------------------------------

    def get_window_features(
        self,
        chrom: str,
        start: int,
        end: Optional[int] = None,
        max_sites: int = 128,
    ) -> Dict[str, Tensor]:
        """Extract methylation features for a genomic window.

        Returns padded tensors of CpG beta values, relative positions
        within the window, and coverage values.

        Args:
            chrom: Chromosome name (e.g. ``"chr1"``).
            start: Window start position (0-based).
            end: Window end position.  If ``None``, uses
                ``start + window_size``.
            max_sites: Maximum number of CpG sites to return per window.
                Extra sites are dropped; fewer are zero-padded.

        Returns:
            Dictionary with:
                - ``beta_values``: ``(max_sites,)`` methylation levels [0, 1].
                - ``positions``: ``(max_sites,)`` relative positions in [0, 1].
                - ``coverage``: ``(max_sites,)`` read coverage values.
                - ``mask``: ``(max_sites,)`` bool tensor (True = real site).
                - ``num_sites``: scalar integer tensor.
        """
        if not self._loaded:
            logger.warning("No bedMethyl data loaded; returning empty features")
            return self._empty_window_features(max_sites)

        if end is None:
            end = start + self.window_size

        window_length = end - start

        chrom_data = self._data.get(chrom)
        if chrom_data is None:
            return self._empty_window_features(max_sites)

        # Binary search for positions within the window
        positions = chrom_data[:, 0]
        left = np.searchsorted(positions, start, side="left")
        right = np.searchsorted(positions, end, side="right")

        sites = chrom_data[left:right]
        num_sites = min(len(sites), max_sites)

        # Allocate output tensors
        beta_values = torch.zeros(max_sites, dtype=torch.float32)
        rel_positions = torch.zeros(max_sites, dtype=torch.float32)
        coverage = torch.zeros(max_sites, dtype=torch.float32)
        mask = torch.zeros(max_sites, dtype=torch.bool)

        if num_sites > 0:
            selected = sites[:num_sites]
            beta_values[:num_sites] = torch.from_numpy(selected[:, 1])
            # Normalise positions to [0, 1] relative to window
            rel_positions[:num_sites] = torch.from_numpy(
                (selected[:, 0] - start).astype(np.float32) / max(window_length, 1)
            )
            coverage[:num_sites] = torch.from_numpy(selected[:, 2])
            mask[:num_sites] = True

        return {
            "beta_values": beta_values,
            "positions": rel_positions,
            "coverage": coverage,
            "mask": mask,
            "num_sites": torch.tensor(num_sites, dtype=torch.long),
        }

    def get_batch_features(
        self,
        regions: List[Tuple[str, int, int]],
        max_sites: int = 128,
    ) -> Dict[str, Tensor]:
        """Extract methylation features for a batch of genomic windows.

        Args:
            regions: List of ``(chrom, start, end)`` tuples.
            max_sites: Maximum CpG sites per window.

        Returns:
            Dictionary with batched tensors (first dim = batch):
                - ``beta_values``: ``(B, max_sites)``
                - ``positions``: ``(B, max_sites)``
                - ``coverage``: ``(B, max_sites)``
                - ``mask``: ``(B, max_sites)``
                - ``num_sites``: ``(B,)``
        """
        all_features = [
            self.get_window_features(c, s, e, max_sites=max_sites)
            for c, s, e in regions
        ]

        if not all_features:
            return {
                "beta_values": torch.empty(0, max_sites),
                "positions": torch.empty(0, max_sites),
                "coverage": torch.empty(0, max_sites),
                "mask": torch.empty(0, max_sites, dtype=torch.bool),
                "num_sites": torch.empty(0, dtype=torch.long),
            }

        return {
            key: torch.stack([f[key] for f in all_features])
            for key in all_features[0]
        }

    def get_cpg_density(
        self,
        chrom: str,
        start: int,
        end: Optional[int] = None,
    ) -> float:
        """Compute CpG site density for a genomic window.

        Args:
            chrom: Chromosome name.
            start: Window start.
            end: Window end (default: ``start + window_size``).

        Returns:
            Number of CpG sites per kilobase in the window.
        """
        if end is None:
            end = start + self.window_size

        chrom_data = self._data.get(chrom)
        if chrom_data is None:
            return 0.0

        positions = chrom_data[:, 0]
        left = np.searchsorted(positions, start, side="left")
        right = np.searchsorted(positions, end, side="right")

        n_sites = right - left
        window_kb = (end - start) / 1000.0
        return n_sites / max(window_kb, 1e-9)

    # ------------------------------------------------------------------
    # Properties and helpers
    # ------------------------------------------------------------------

    @property
    def chromosomes(self) -> List[str]:
        """List of chromosomes with loaded data."""
        return sorted(self._data.keys())

    @property
    def total_sites(self) -> int:
        """Total number of CpG sites across all chromosomes."""
        return sum(len(v) for v in self._data.values())

    def _empty_window_features(self, max_sites: int) -> Dict[str, Tensor]:
        """Return empty feature tensors for a window with no data."""
        return {
            "beta_values": torch.zeros(max_sites, dtype=torch.float32),
            "positions": torch.zeros(max_sites, dtype=torch.float32),
            "coverage": torch.zeros(max_sites, dtype=torch.float32),
            "mask": torch.zeros(max_sites, dtype=torch.bool),
            "num_sites": torch.tensor(0, dtype=torch.long),
        }

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "empty"
        return (
            f"ONTMethylationProcessor("
            f"window_size={self.window_size}, "
            f"min_coverage={self.min_coverage}, "
            f"status={status}, "
            f"total_sites={self.total_sites})"
        )
