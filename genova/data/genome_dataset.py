"""PyTorch Dataset for genomic sequences with MLM support.

Reads FASTA files via pyfaidx for memory-mapped access, supports sliding
window extraction, BED filtering, reverse-complement augmentation,
GC-content filtering, and Masked Language Modeling sample generation.

Example::

    from genova.data.tokenizer import GenomicTokenizer
    from genova.data.genome_dataset import GenomeDataset

    tokenizer = GenomicTokenizer(mode="kmer", k=6)
    tokenizer.build_vocab()
    ds = GenomeDataset(
        fasta_path="hg38.fa",
        tokenizer=tokenizer,
        window_size=2048,
        stride=512,
    )
    sample = ds[0]  # {"input_ids": ..., "attention_mask": ..., "labels": ...}
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger

from genova.data.tokenizer import GenomicTokenizer, reverse_complement

# ---------------------------------------------------------------------------
# BED parsing
# ---------------------------------------------------------------------------

_COMPLEMENT_MAP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _parse_bed(bed_path: Union[str, Path]) -> Dict[str, List[Tuple[int, int]]]:
    """Parse a BED file into a dict mapping chrom -> sorted list of (start, end).

    Handles empty files and skips comment / track lines.
    """
    regions: Dict[str, List[Tuple[int, int]]] = {}
    bed_path = Path(bed_path)
    if not bed_path.exists():
        raise FileNotFoundError(f"BED file not found: {bed_path}")

    with open(bed_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            try:
                start, end = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            regions.setdefault(chrom, []).append((start, end))

    for chrom in regions:
        regions[chrom].sort()
    return regions


def _overlaps_any(
    start: int,
    end: int,
    intervals: List[Tuple[int, int]],
) -> bool:
    """Return True if [start, end) overlaps any interval in a sorted list."""
    import bisect

    # Binary search for the first interval whose end > start
    idx = bisect.bisect_left(intervals, (start,))
    # Check the interval at idx and the one before it
    for i in range(max(0, idx - 1), min(len(intervals), idx + 2)):
        istart, iend = intervals[i]
        if istart < end and iend > start:
            return True
    return False


# ---------------------------------------------------------------------------
# GC content
# ---------------------------------------------------------------------------

def _gc_content(seq: str) -> float:
    """Compute GC fraction of a DNA sequence."""
    if not seq:
        return 0.0
    seq_upper = seq.upper()
    gc = seq_upper.count("G") + seq_upper.count("C")
    total = len(seq_upper) - seq_upper.count("N")
    return gc / max(total, 1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class GenomeDataset(Dataset):
    """Memory-efficient PyTorch Dataset over a FASTA genome.

    Windows are lazily fetched from the FASTA index so that genomes
    larger than available RAM can be processed.

    Args:
        fasta_path: Path to an indexed FASTA file (``.fai`` sidecar
            expected; pyfaidx will create one if absent).
        tokenizer: A :class:`GenomicTokenizer` with a built vocabulary.
        window_size: Extraction window size in base pairs.
        stride: Step between consecutive windows in base pairs.
        bed_path: Optional BED file; when provided, only windows
            overlapping at least one BED region are included.
        reverse_complement_prob: Probability of returning the reverse
            complement of a window (data augmentation).
        gc_min: Minimum GC content to keep a window (``None`` = no filter).
        gc_max: Maximum GC content to keep a window (``None`` = no filter).
        mask_prob: Fraction of tokens masked for MLM training.
        max_n_fraction: Maximum fraction of ``N`` bases allowed in a
            window.  Windows exceeding this threshold are skipped.
        max_length: Maximum token-sequence length (for truncation).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        fasta_path: Union[str, Path],
        tokenizer: GenomicTokenizer,
        window_size: int = 2048,
        stride: int = 512,
        bed_path: Optional[Union[str, Path]] = None,
        reverse_complement_prob: float = 0.0,
        gc_min: Optional[float] = None,
        gc_max: Optional[float] = None,
        mask_prob: float = 0.15,
        max_n_fraction: float = 0.1,
        max_length: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if not (1_000 <= window_size <= 10_000):
            logger.warning(
                "window_size={} is outside the recommended 1kb-10kb range",
                window_size,
            )
        if not (0.0 <= mask_prob <= 1.0):
            raise ValueError(f"mask_prob must be in [0, 1], got {mask_prob}")

        self.fasta_path = Path(fasta_path)
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride
        self.reverse_complement_prob = reverse_complement_prob
        self.gc_min = gc_min
        self.gc_max = gc_max
        self.mask_prob = mask_prob
        self.max_n_fraction = max_n_fraction
        self.max_length = max_length
        self.seed = seed

        # Lazy-import pyfaidx to avoid hard dependency at import time
        try:
            import pyfaidx
        except ImportError as exc:
            raise ImportError(
                "pyfaidx is required for GenomeDataset. "
                "Install it with: pip install pyfaidx"
            ) from exc

        if not self.fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.fasta_path}")

        self._fasta = pyfaidx.Fasta(str(self.fasta_path), read_ahead=10_000)

        # Parse BED regions if provided
        self._bed_regions: Optional[Dict[str, List[Tuple[int, int]]]] = None
        if bed_path is not None:
            self._bed_regions = _parse_bed(bed_path)
            if not self._bed_regions:
                logger.warning("BED file {} is empty; dataset will have 0 windows", bed_path)

        # Build window index: list of (chrom, start) tuples
        self._windows: List[Tuple[str, int]] = []
        self._build_window_index()

        logger.info(
            "GenomeDataset: {} windows from {} (window={}bp, stride={}bp)",
            len(self._windows),
            self.fasta_path.name,
            self.window_size,
            self.stride,
        )

    # ---------------------------------------------------------- index building
    def _build_window_index(self) -> None:
        """Pre-compute the list of valid (chrom, start) pairs."""
        rng = random.Random(self.seed)

        for chrom in sorted(self._fasta.keys()):
            chrom_len = len(self._fasta[chrom])
            if chrom_len < self.window_size:
                continue

            # If BED regions are provided but this chrom has none, skip
            if self._bed_regions is not None and chrom not in self._bed_regions:
                continue

            bed_intervals = (
                self._bed_regions[chrom] if self._bed_regions is not None else None
            )

            for start in range(0, chrom_len - self.window_size + 1, self.stride):
                end = start + self.window_size

                # BED overlap filter
                if bed_intervals is not None and not _overlaps_any(start, end, bed_intervals):
                    continue

                self._windows.append((chrom, start))

        # Deterministic shuffle so epoch ordering is reproducible
        rng.shuffle(self._windows)

    # ---------------------------------------------------------- extraction
    def _extract_sequence(self, chrom: str, start: int) -> str:
        """Fetch a window from the FASTA (zero-based, half-open)."""
        return str(self._fasta[chrom][start : start + self.window_size])

    # ---------------------------------------------------------- filters
    def _passes_filters(self, seq: str) -> bool:
        """Return True if the sequence passes N-fraction and GC filters."""
        seq_upper = seq.upper()

        # N-fraction filter
        n_frac = seq_upper.count("N") / max(len(seq_upper), 1)
        if n_frac > self.max_n_fraction:
            return False

        # GC-content filter
        if self.gc_min is not None or self.gc_max is not None:
            gc = _gc_content(seq_upper)
            if self.gc_min is not None and gc < self.gc_min:
                return False
            if self.gc_max is not None and gc > self.gc_max:
                return False

        return True

    # ---------------------------------------------------------- MLM masking
    def _apply_mlm_masking(
        self,
        input_ids: List[int],
        rng: random.Random,
    ) -> Tuple[List[int], List[int]]:
        """Apply MLM masking strategy.

        Strategy (following BERT):
          - 80 %  replace with [MASK]
          - 10 %  replace with a random token
          - 10 %  keep unchanged

        Args:
            input_ids: Original token ids (may include [CLS]/[SEP]).
            rng: Seeded random generator.

        Returns:
            Tuple of (masked_input_ids, labels).  Labels are -100 for
            positions that should be ignored in the loss.
        """
        masked_ids = list(input_ids)
        labels = [-100] * len(input_ids)

        special_ids = {
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
        }

        # Collect maskable positions
        candidates = [
            i for i, tid in enumerate(input_ids) if tid not in special_ids
        ]

        n_mask = max(1, int(len(candidates) * self.mask_prob))
        masked_positions = rng.sample(candidates, min(n_mask, len(candidates)))

        # Determine random token pool (non-special)
        non_special_ids = [
            tid for tid in self.tokenizer.id_to_token
            if tid not in special_ids
        ]

        for pos in masked_positions:
            labels[pos] = input_ids[pos]
            r = rng.random()
            if r < 0.8:
                masked_ids[pos] = self.tokenizer.mask_token_id
            elif r < 0.9:
                masked_ids[pos] = rng.choice(non_special_ids) if non_special_ids else input_ids[pos]
            # else: keep unchanged (10 %)

        return masked_ids, labels

    # ---------------------------------------------------------- __len__ / __getitem__
    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single MLM training sample.

        Returns:
            Dict with keys ``input_ids``, ``attention_mask``, ``labels``.
        """
        chrom, start = self._windows[idx]
        seq = self._extract_sequence(chrom, start)

        # Per-sample deterministic RNG
        rng = random.Random(self.seed + idx)

        # Filter: if this window fails, walk forward to the next valid one.
        # In practice the index was pre-built so most will pass, but GC/N
        # filtering after extraction is a safety net.
        if not self._passes_filters(seq):
            # Replace with an empty / padded sample
            pad_len = self.max_length or 128
            return {
                "input_ids": torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long),
                "attention_mask": torch.zeros(pad_len, dtype=torch.long),
                "labels": torch.full((pad_len,), -100, dtype=torch.long),
            }

        # Reverse complement augmentation
        if self.reverse_complement_prob > 0 and rng.random() < self.reverse_complement_prob:
            seq = reverse_complement(seq)

        # Tokenize
        input_ids = self.tokenizer.encode(
            seq,
            add_special_tokens=True,
            max_length=self.max_length,
        )

        # MLM masking
        if self.mask_prob > 0:
            masked_ids, labels = self._apply_mlm_masking(input_ids, rng)
        else:
            masked_ids = input_ids
            labels = [-100] * len(input_ids)

        attention_mask = [1] * len(masked_ids)

        return {
            "input_ids": torch.tensor(masked_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # ---------------------------------------------------------- cleanup
    def close(self) -> None:
        """Close the underlying FASTA handle."""
        if hasattr(self, "_fasta") and self._fasta is not None:
            self._fasta.close()
            self._fasta = None

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"GenomeDataset(fasta={self.fasta_path.name!r}, "
            f"windows={len(self._windows)}, "
            f"window_size={self.window_size}, stride={self.stride})"
        )
