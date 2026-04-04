"""Allele frequency encoding for population-aware genomic models.

Parses allele frequency data from gnomAD/QGP-style TSV or VCF formats and
encodes population-specific frequencies as log-scaled continuous features
suitable for neural network consumption.

Handles missing frequency data gracefully by imputing with configurable
defaults and producing validity masks for downstream masking.

Example::

    encoder = AlleleFrequencyEncoder(
        populations=["EUR", "AFR", "EAS", "SAS", "AMR", "MEA"],
    )
    features, mask = encoder.encode_from_tsv("variants.tsv")
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import Tensor


# Default superpopulation labels used across gnomAD / QGP
DEFAULT_POPULATIONS: List[str] = ["EUR", "AFR", "EAS", "SAS", "AMR", "MEA"]

# Minimum allele frequency floor to avoid log(0)
_AF_FLOOR: float = 1e-8

# Sentinel used to mark missing AF values before imputation
_MISSING_SENTINEL: float = -1.0


class AlleleFrequencyEncoder:
    """Encode allele frequencies from population databases as model features.

    Reads TSV or VCF-derived tables with per-population allele frequency
    columns and produces log-scaled tensors.  Missing values are imputed
    with a configurable default (typically a very small frequency) and a
    boolean validity mask is returned alongside the features.

    Args:
        populations: Ordered list of population labels. Column names in
            input files are expected to follow the pattern
            ``AF_{pop}`` (e.g. ``AF_EUR``).
        default_af: Default allele frequency used when a value is missing
            for a given population.  Should be a small positive number.
        log_base: Base of the logarithm used for scaling.  Common choices
            are ``10`` (log10) or ``math.e`` (natural log).
        include_global_af: If ``True``, also encode the ``AF`` (global)
            column when present.
    """

    def __init__(
        self,
        populations: Optional[List[str]] = None,
        default_af: float = 1e-6,
        log_base: float = 10.0,
        include_global_af: bool = True,
    ) -> None:
        self.populations = populations or DEFAULT_POPULATIONS
        self.default_af = default_af
        self.log_base = log_base
        self.include_global_af = include_global_af
        self._log_scale = math.log(log_base)

        # Expected column names
        self._pop_columns = [f"AF_{p}" for p in self.populations]

        logger.debug(
            "AlleleFrequencyEncoder initialised with {} populations, "
            "default_af={}, log_base={}",
            len(self.populations),
            default_af,
            log_base,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_features(self) -> int:
        """Number of output features per variant."""
        n = len(self.populations)
        if self.include_global_af:
            n += 1
        return n

    def encode_frequencies(
        self,
        af_values: Dict[str, Optional[float]],
    ) -> Tuple[Tensor, Tensor]:
        """Encode a single variant's allele frequencies.

        Args:
            af_values: Mapping from population label (e.g. ``"EUR"``) to
                allele frequency.  ``None`` indicates a missing value.

        Returns:
            Tuple of:
                - ``features``: ``(num_features,)`` float tensor of
                  log-scaled frequencies.
                - ``mask``: ``(num_features,)`` bool tensor where ``True``
                  indicates the value was observed (not imputed).
        """
        features: List[float] = []
        mask: List[bool] = []

        for pop in self.populations:
            af = af_values.get(pop)
            if af is None or (isinstance(af, float) and math.isnan(af)):
                features.append(self._log_scale_af(self.default_af))
                mask.append(False)
            else:
                features.append(self._log_scale_af(max(float(af), _AF_FLOOR)))
                mask.append(True)

        if self.include_global_af:
            global_af = af_values.get("GLOBAL") or af_values.get("AF")
            if global_af is None or (isinstance(global_af, float) and math.isnan(global_af)):
                features.append(self._log_scale_af(self.default_af))
                mask.append(False)
            else:
                features.append(self._log_scale_af(max(float(global_af), _AF_FLOOR)))
                mask.append(True)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
        )

    def encode_batch(
        self,
        af_records: Sequence[Dict[str, Optional[float]]],
    ) -> Tuple[Tensor, Tensor]:
        """Encode a batch of variant allele frequencies.

        Args:
            af_records: Sequence of per-variant AF dictionaries.

        Returns:
            Tuple of:
                - ``features``: ``(N, num_features)`` float tensor.
                - ``mask``: ``(N, num_features)`` bool tensor.
        """
        if not af_records:
            return (
                torch.empty(0, self.num_features, dtype=torch.float32),
                torch.empty(0, self.num_features, dtype=torch.bool),
            )

        feat_list, mask_list = [], []
        for record in af_records:
            f, m = self.encode_frequencies(record)
            feat_list.append(f)
            mask_list.append(m)

        return torch.stack(feat_list), torch.stack(mask_list)

    def encode_from_tsv(
        self,
        path: Union[str, Path],
        chrom_col: str = "CHROM",
        pos_col: str = "POS",
        ref_col: str = "REF",
        alt_col: str = "ALT",
        sep: str = "\t",
    ) -> Tuple[Tensor, Tensor, pd.DataFrame]:
        """Parse a TSV file and encode allele frequencies.

        Expects columns for chromosome, position, ref, alt, and
        per-population AF columns named ``AF_{POP}`` (e.g. ``AF_EUR``).
        Optionally a global ``AF`` column.

        Args:
            path: Path to the TSV file.
            chrom_col: Column name for chromosome.
            pos_col: Column name for genomic position.
            ref_col: Column name for reference allele.
            alt_col: Column name for alternate allele.
            sep: Field separator.

        Returns:
            Tuple of:
                - ``features``: ``(N, num_features)`` float tensor.
                - ``mask``: ``(N, num_features)`` bool tensor.
                - ``variants``: DataFrame with variant identifiers
                  (chrom, pos, ref, alt).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"AF file not found: {path}")

        logger.info("Loading allele frequencies from {}", path)
        df = pd.read_csv(path, sep=sep, low_memory=False)

        # Identify available AF columns
        available_pop_cols = [c for c in self._pop_columns if c in df.columns]
        missing_pop_cols = [c for c in self._pop_columns if c not in df.columns]

        if missing_pop_cols:
            logger.warning(
                "Missing AF columns for populations: {}",
                [c.replace("AF_", "") for c in missing_pop_cols],
            )

        # Build records
        records: List[Dict[str, Optional[float]]] = []
        for _, row in df.iterrows():
            rec: Dict[str, Optional[float]] = {}
            for pop, col in zip(self.populations, self._pop_columns):
                if col in df.columns:
                    val = row[col]
                    rec[pop] = None if pd.isna(val) else float(val)
                else:
                    rec[pop] = None
            # Global AF
            if "AF" in df.columns:
                val = row["AF"]
                rec["GLOBAL"] = None if pd.isna(val) else float(val)
            records.append(rec)

        features, mask = self.encode_batch(records)

        # Build variant identifier DataFrame
        id_cols = [c for c in [chrom_col, pos_col, ref_col, alt_col] if c in df.columns]
        variants = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)

        logger.info(
            "Encoded {} variants with {} features each",
            len(features),
            self.num_features,
        )

        return features, mask, variants

    def encode_from_vcf_af_fields(
        self,
        info_fields: Dict[str, str],
    ) -> Tuple[Tensor, Tensor]:
        """Encode AF from a VCF INFO field dictionary.

        Parses VCF-style INFO key-value pairs where AF fields follow
        gnomAD conventions (e.g. ``AF_eur``, ``AF_afr``, etc.).

        Args:
            info_fields: Dictionary of INFO field keys to their string
                values.

        Returns:
            Tuple of ``(features, mask)`` tensors for a single variant.
        """
        af_values: Dict[str, Optional[float]] = {}

        for pop in self.populations:
            # Try common VCF naming conventions
            for key_pattern in [f"AF_{pop}", f"AF_{pop.lower()}", f"gnomAD_AF_{pop}"]:
                if key_pattern in info_fields:
                    try:
                        af_values[pop] = float(info_fields[key_pattern])
                    except (ValueError, TypeError):
                        af_values[pop] = None
                    break
            else:
                af_values[pop] = None

        # Global AF
        for key in ["AF", "gnomAD_AF"]:
            if key in info_fields:
                try:
                    af_values["GLOBAL"] = float(info_fields[key])
                except (ValueError, TypeError):
                    af_values["GLOBAL"] = None
                break

        return self.encode_frequencies(af_values)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_scale_af(self, af: float) -> float:
        """Apply log scaling to an allele frequency value.

        Transforms AF to ``log(AF) / log(base)`` which maps the typical
        range [1e-8, 1.0] to a more neural-network-friendly range.

        For base=10: AF=1e-6 -> -6.0, AF=0.01 -> -2.0, AF=0.5 -> ~-0.3
        """
        return math.log(max(af, _AF_FLOOR)) / self._log_scale

    def __repr__(self) -> str:
        return (
            f"AlleleFrequencyEncoder("
            f"populations={self.populations}, "
            f"default_af={self.default_af}, "
            f"log_base={self.log_base}, "
            f"num_features={self.num_features})"
        )
