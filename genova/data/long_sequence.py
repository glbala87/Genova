"""Long-sequence dataset for Genova (Phase 2 -- sequences up to 100kb+).

Extends the base :class:`GenomeDataset` to handle very long genomic
sequences via chunking with configurable overlap and random cropping
augmentation, while keeping memory usage bounded.

Example::

    from genova.data.tokenizer import GenomicTokenizer
    from genova.data.long_sequence import LongSequenceDataset

    tokenizer = GenomicTokenizer(mode="kmer", k=6)
    tokenizer.build_vocab()
    ds = LongSequenceDataset(
        fasta_path="hg38.fa",
        tokenizer=tokenizer,
        window_size=100_000,  # 100 kb windows
        stride=50_000,
        chunk_length=4096,
        chunk_overlap=256,
    )
    sample = ds[0]
    # sample["input_ids"]  shape: (num_chunks, chunk_length)
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger

from genova.data.genome_dataset import GenomeDataset
from genova.data.tokenizer import GenomicTokenizer, reverse_complement


class LongSequenceDataset(GenomeDataset):
    """Dataset for long genomic sequences (up to 100 kb+).

    Instead of returning a single flat token sequence per window, this
    dataset chunks the encoded tokens into fixed-length segments with
    configurable overlap.  This allows downstream models (e.g., hierarchical
    transformers) to process very long contexts without exceeding GPU memory.

    Additional augmentations:

    * **Random cropping**: Instead of always starting at the window
      boundary, randomly shift the start position within
      ``[0, crop_max_shift)`` base pairs.

    Args:
        fasta_path: Path to an indexed FASTA file.
        tokenizer: A :class:`GenomicTokenizer` with a built vocabulary.
        window_size: Window size in base pairs (can be up to 100 kb+).
        stride: Step between consecutive windows (bp).
        chunk_length: Maximum number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between consecutive
            chunks.  Helps the model attend across chunk boundaries.
        random_crop: Whether to apply random cropping augmentation.
        crop_max_shift: Maximum random shift in base pairs for cropping.
        bed_path: Optional BED file for region filtering.
        reverse_complement_prob: Probability of reverse-complement augment.
        gc_min: Minimum GC content.
        gc_max: Maximum GC content.
        mask_prob: MLM mask probability.
        max_n_fraction: Maximum fraction of N bases.
        seed: Random seed.
    """

    def __init__(
        self,
        fasta_path: Union[str, Path],
        tokenizer: GenomicTokenizer,
        window_size: int = 100_000,
        stride: int = 50_000,
        chunk_length: int = 4096,
        chunk_overlap: int = 256,
        random_crop: bool = True,
        crop_max_shift: int = 1000,
        bed_path: Optional[Union[str, Path]] = None,
        reverse_complement_prob: float = 0.0,
        gc_min: Optional[float] = None,
        gc_max: Optional[float] = None,
        mask_prob: float = 0.15,
        max_n_fraction: float = 0.1,
        seed: int = 42,
    ) -> None:
        if chunk_overlap >= chunk_length:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_length ({chunk_length})"
            )
        if chunk_length < 1:
            raise ValueError(f"chunk_length must be >= 1, got {chunk_length}")

        self.chunk_length = chunk_length
        self.chunk_overlap = chunk_overlap
        self.random_crop = random_crop
        self.crop_max_shift = crop_max_shift

        # Bypass the 1kb-10kb warning in the parent by calling __init__
        # with the given window_size (long sequences are expected here).
        super().__init__(
            fasta_path=fasta_path,
            tokenizer=tokenizer,
            window_size=window_size,
            stride=stride,
            bed_path=bed_path,
            reverse_complement_prob=reverse_complement_prob,
            gc_min=gc_min,
            gc_max=gc_max,
            mask_prob=mask_prob,
            max_n_fraction=max_n_fraction,
            max_length=None,  # We handle length via chunking
            seed=seed,
        )

        logger.info(
            "LongSequenceDataset: chunk_length={}, overlap={}, random_crop={}",
            self.chunk_length,
            self.chunk_overlap,
            self.random_crop,
        )

    # ---------------------------------------------------------- chunking
    def _chunk_ids(
        self,
        token_ids: List[int],
    ) -> List[List[int]]:
        """Split a token-id list into overlapping chunks.

        The last chunk is right-padded with ``[PAD]`` if shorter than
        *chunk_length*.

        Returns:
            List of token-id lists, each of length *chunk_length*.
        """
        step = self.chunk_length - self.chunk_overlap
        if step < 1:
            step = 1

        chunks: List[List[int]] = []
        for start in range(0, len(token_ids), step):
            chunk = token_ids[start : start + self.chunk_length]
            if len(chunk) < self.chunk_length:
                pad_len = self.chunk_length - len(chunk)
                chunk = chunk + [self.tokenizer.pad_token_id] * pad_len
            chunks.append(chunk)

        # Safety: always return at least one chunk
        if not chunks:
            chunks = [[self.tokenizer.pad_token_id] * self.chunk_length]

        return chunks

    # ---------------------------------------------------------- __getitem__
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a long-sequence sample split into chunks.

        Returns:
            Dict with keys:
              - ``input_ids``:      ``(num_chunks, chunk_length)``
              - ``attention_mask``:  ``(num_chunks, chunk_length)``
              - ``labels``:         ``(num_chunks, chunk_length)``
              - ``num_chunks``:     scalar int tensor
        """
        chrom, start = self._windows[idx]
        rng = random.Random(self.seed + idx)

        # Random cropping: shift the start position
        actual_start = start
        if self.random_crop and self.crop_max_shift > 0:
            import pyfaidx

            chrom_len = len(self._fasta[chrom])
            max_shift = min(self.crop_max_shift, chrom_len - start - self.window_size)
            max_shift = max(0, max_shift)
            if max_shift > 0:
                actual_start = start + rng.randint(0, max_shift)

        seq = self._extract_sequence(chrom, actual_start)

        # Filter check
        if not self._passes_filters(seq):
            empty_chunk = [self.tokenizer.pad_token_id] * self.chunk_length
            return {
                "input_ids": torch.tensor([empty_chunk], dtype=torch.long),
                "attention_mask": torch.zeros(1, self.chunk_length, dtype=torch.long),
                "labels": torch.full((1, self.chunk_length), -100, dtype=torch.long),
                "num_chunks": torch.tensor(1, dtype=torch.long),
            }

        # Reverse complement augmentation
        if self.reverse_complement_prob > 0 and rng.random() < self.reverse_complement_prob:
            seq = reverse_complement(seq)

        # Tokenize (no truncation -- we chunk instead)
        token_ids = self.tokenizer.encode(seq, add_special_tokens=True)

        # MLM masking on the flat sequence
        if self.mask_prob > 0:
            masked_ids, labels = self._apply_mlm_masking(token_ids, rng)
        else:
            masked_ids = token_ids
            labels = [-100] * len(token_ids)

        # Chunk
        id_chunks = self._chunk_ids(masked_ids)
        label_chunks = self._chunk_ids(labels)

        # Ensure label padding uses -100
        for lc in label_chunks:
            for i in range(len(lc)):
                if lc[i] == self.tokenizer.pad_token_id:
                    lc[i] = -100

        # Build attention masks (1 where not PAD)
        attn_chunks: List[List[int]] = []
        for ic in id_chunks:
            attn_chunks.append([0 if t == self.tokenizer.pad_token_id else 1 for t in ic])

        return {
            "input_ids": torch.tensor(id_chunks, dtype=torch.long),
            "attention_mask": torch.tensor(attn_chunks, dtype=torch.long),
            "labels": torch.tensor(label_chunks, dtype=torch.long),
            "num_chunks": torch.tensor(len(id_chunks), dtype=torch.long),
        }

    def __repr__(self) -> str:
        return (
            f"LongSequenceDataset(fasta={self.fasta_path.name!r}, "
            f"windows={len(self._windows)}, "
            f"window_size={self.window_size}, "
            f"chunk_length={self.chunk_length}, "
            f"chunk_overlap={self.chunk_overlap})"
        )


# ---------------------------------------------------------------------------
# Long-sequence collate
# ---------------------------------------------------------------------------


def long_sequence_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Collate long-sequence samples, padding to the maximum number of chunks.

    All samples are padded to have the same number of chunks.

    Returns:
        Dict with:
          - ``input_ids``:      ``(B, max_chunks, chunk_length)``
          - ``attention_mask``:  ``(B, max_chunks, chunk_length)``
          - ``labels``:         ``(B, max_chunks, chunk_length)``
          - ``num_chunks``:     ``(B,)``
    """
    max_chunks = max(sample["input_ids"].size(0) for sample in batch)
    chunk_length = batch[0]["input_ids"].size(1)

    padded_ids: List[torch.Tensor] = []
    padded_attn: List[torch.Tensor] = []
    padded_labels: List[torch.Tensor] = []
    num_chunks_list: List[torch.Tensor] = []

    for sample in batch:
        nc = sample["input_ids"].size(0)
        pad_chunks = max_chunks - nc

        if pad_chunks > 0:
            pad_ids = torch.full((pad_chunks, chunk_length), pad_token_id, dtype=torch.long)
            pad_attn = torch.zeros(pad_chunks, chunk_length, dtype=torch.long)
            pad_lbl = torch.full((pad_chunks, chunk_length), -100, dtype=torch.long)

            padded_ids.append(torch.cat([sample["input_ids"], pad_ids], dim=0))
            padded_attn.append(torch.cat([sample["attention_mask"], pad_attn], dim=0))
            padded_labels.append(torch.cat([sample["labels"], pad_lbl], dim=0))
        else:
            padded_ids.append(sample["input_ids"])
            padded_attn.append(sample["attention_mask"])
            padded_labels.append(sample["labels"])

        num_chunks_list.append(sample["num_chunks"])

    return {
        "input_ids": torch.stack(padded_ids),
        "attention_mask": torch.stack(padded_attn),
        "labels": torch.stack(padded_labels),
        "num_chunks": torch.stack(num_chunks_list),
    }
