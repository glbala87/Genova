"""Preprocessing pipeline: FASTA to tokenized Arrow/Parquet format.

Processes a genome FASTA file into chunked, tokenized datasets stored as
Parquet files for fast downstream loading.  Supports multiprocessing and
tracks progress with tqdm.

Example::

    from genova.utils.config import DataConfig
    from genova.data.preprocessing import preprocess_genome

    config = DataConfig(genome_fasta="hg38.fa", kmer_size=6, seq_length=2048)
    preprocess_genome("hg38.fa", "data/processed", config)
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from genova.data.tokenizer import GenomicTokenizer


def _gc_content(seq: str) -> float:
    """Compute GC fraction of a DNA sequence."""
    if not seq:
        return 0.0
    gc = seq.count("G") + seq.count("C")
    total = len(seq) - seq.count("N")
    return gc / max(total, 1)


# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------

# Module-level tokenizer used by worker processes (set via initializer).
_worker_tokenizer: Optional[GenomicTokenizer] = None


def _init_worker(tokenizer_path: str) -> None:
    """Per-worker initializer: load the tokenizer once."""
    global _worker_tokenizer
    _worker_tokenizer = GenomicTokenizer.load(tokenizer_path)


def _tokenize_chunk(
    chunk: List[Tuple[str, str, int, int]],
    *,
    max_length: int,
    gc_min: Optional[float],
    gc_max: Optional[float],
    max_n_fraction: float,
) -> List[Dict[str, Any]]:
    """Tokenize a list of (chrom, sequence, start, end) tuples.

    Returns a list of dicts ready to be written to Arrow.
    """
    assert _worker_tokenizer is not None, "Worker tokenizer not initialised"
    results: List[Dict[str, Any]] = []

    for chrom, seq, start, end in chunk:
        seq_upper = seq.upper()

        # N-fraction filter
        n_frac = seq_upper.count("N") / max(len(seq_upper), 1)
        if n_frac > max_n_fraction:
            continue

        # GC filter
        gc = _gc_content(seq_upper)
        if gc_min is not None and gc < gc_min:
            continue
        if gc_max is not None and gc > gc_max:
            continue

        token_ids = _worker_tokenizer.encode(
            seq_upper,
            add_special_tokens=True,
            max_length=max_length,
        )

        results.append(
            {
                "chrom": chrom,
                "start": start,
                "end": end,
                "input_ids": token_ids,
                "length": len(token_ids),
                "gc_content": round(gc, 4),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Sequence extraction
# ---------------------------------------------------------------------------


def _extract_windows(
    fasta_path: Union[str, Path],
    window_size: int,
    stride: int,
    bed_path: Optional[Union[str, Path]] = None,
) -> List[Tuple[str, str, int, int]]:
    """Extract all sliding windows from a FASTA file.

    Each element is ``(chrom, sequence, start, end)``.  Sequences are read
    lazily via pyfaidx to avoid loading the whole genome into memory.
    """
    try:
        import pyfaidx
    except ImportError as exc:
        raise ImportError("pyfaidx is required. Install with: pip install pyfaidx") from exc

    fasta = pyfaidx.Fasta(str(fasta_path), read_ahead=10_000)
    windows: List[Tuple[str, str, int, int]] = []

    # Optional BED filtering
    bed_regions: Optional[Dict[str, List[Tuple[int, int]]]] = None
    if bed_path is not None:
        from genova.data.genome_dataset import _parse_bed, _overlaps_any

        bed_regions = _parse_bed(bed_path)

    for chrom in sorted(fasta.keys()):
        chrom_len = len(fasta[chrom])
        if chrom_len < window_size:
            continue

        if bed_regions is not None and chrom not in bed_regions:
            continue

        intervals = bed_regions[chrom] if bed_regions is not None else None

        for start in range(0, chrom_len - window_size + 1, stride):
            end = start + window_size
            if intervals is not None:
                from genova.data.genome_dataset import _overlaps_any

                if not _overlaps_any(start, end, intervals):
                    continue
            seq = str(fasta[chrom][start:end])
            windows.append((chrom, seq, start, end))

    fasta.close()
    return windows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def preprocess_genome(
    fasta_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: Any,
    *,
    bed_path: Optional[Union[str, Path]] = None,
    gc_min: Optional[float] = None,
    gc_max: Optional[float] = None,
    max_n_fraction: float = 0.1,
    chunk_size: int = 5_000,
    num_workers: Optional[int] = None,
) -> Path:
    """Process a FASTA genome into tokenized Parquet files.

    Args:
        fasta_path: Path to the genome FASTA file.
        output_dir: Directory where Parquet shards and metadata will be
            written.
        config: A :class:`~genova.utils.config.DataConfig` instance (or any
            object with the attributes ``tokenizer``, ``kmer_size``,
            ``stride``, ``seq_length``, ``max_tokens``, ``num_workers``).
        bed_path: Optional BED file for region filtering.
        gc_min: Minimum GC content threshold.
        gc_max: Maximum GC content threshold.
        max_n_fraction: Maximum fraction of N bases per window.
        chunk_size: Number of windows per processing chunk (controls memory).
        num_workers: Override for ``config.num_workers``.  ``0`` or ``1``
            disables multiprocessing.

    Returns:
        Path to the output directory.
    """
    fasta_path = Path(fasta_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    n_workers = num_workers if num_workers is not None else getattr(config, "num_workers", 4)
    window_size = getattr(config, "seq_length", 2048)
    stride_bp = getattr(config, "stride", 1)
    # For preprocessing, stride in bp (not token stride)
    # Use a reasonable default if stride=1 (token-level default)
    if stride_bp <= 1:
        stride_bp = window_size // 2
    max_length = getattr(config, "max_tokens", 1024)
    kmer_size = getattr(config, "kmer_size", 6)
    tok_mode = getattr(config, "tokenizer", "kmer")

    logger.info(
        "Preprocessing genome: {} -> {} (window={}bp, stride={}bp, workers={})",
        fasta_path.name,
        output_dir,
        window_size,
        stride_bp,
        n_workers,
    )

    # --- Build / save tokenizer -------------------------------------------
    tokenizer = GenomicTokenizer(mode=tok_mode, k=kmer_size)
    tokenizer.build_vocab()
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    # --- Extract windows --------------------------------------------------
    logger.info("Extracting sliding windows from FASTA...")
    windows = _extract_windows(fasta_path, window_size, stride_bp, bed_path)
    total_windows = len(windows)
    logger.info("Total windows extracted: {}", total_windows)

    if total_windows == 0:
        logger.warning("No windows extracted. Check FASTA content and filters.")
        _save_metadata(output_dir, tokenizer, config, total_windows=0, total_tokens=0)
        return output_dir

    # --- Chunk and process ------------------------------------------------
    chunks = [
        windows[i : i + chunk_size] for i in range(0, total_windows, chunk_size)
    ]

    worker_fn = partial(
        _tokenize_chunk,
        max_length=max_length,
        gc_min=gc_min,
        gc_max=gc_max,
        max_n_fraction=max_n_fraction,
    )

    all_records: List[Dict[str, Any]] = []
    shard_idx = 0
    records_per_shard = 50_000
    total_tokens = 0

    if n_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(str(tokenizer_path),),
        ) as pool:
            for result in tqdm(
                pool.imap(worker_fn, chunks),
                total=len(chunks),
                desc="Tokenizing",
                unit="chunk",
            ):
                all_records.extend(result)

                # Flush to Parquet shards periodically
                while len(all_records) >= records_per_shard:
                    shard_records = all_records[:records_per_shard]
                    all_records = all_records[records_per_shard:]
                    total_tokens += _write_shard(output_dir, shard_idx, shard_records)
                    shard_idx += 1
    else:
        # Single-process path
        _init_worker(str(tokenizer_path))
        for chunk in tqdm(chunks, desc="Tokenizing", unit="chunk"):
            result = worker_fn(chunk)
            all_records.extend(result)

            while len(all_records) >= records_per_shard:
                shard_records = all_records[:records_per_shard]
                all_records = all_records[records_per_shard:]
                total_tokens += _write_shard(output_dir, shard_idx, shard_records)
                shard_idx += 1

    # Flush remaining records
    if all_records:
        total_tokens += _write_shard(output_dir, shard_idx, all_records)
        shard_idx += 1

    # --- Metadata ---------------------------------------------------------
    _save_metadata(
        output_dir,
        tokenizer,
        config,
        total_windows=total_windows,
        total_tokens=total_tokens,
        num_shards=shard_idx,
    )

    logger.info(
        "Preprocessing complete: {} shards, {} windows, {} tokens",
        shard_idx,
        total_windows,
        total_tokens,
    )
    return output_dir


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_shard(
    output_dir: Path,
    shard_idx: int,
    records: List[Dict[str, Any]],
) -> int:
    """Write a list of records to a Parquet shard.  Returns total token count."""
    if not records:
        return 0

    table = pa.table(
        {
            "chrom": pa.array([r["chrom"] for r in records], type=pa.string()),
            "start": pa.array([r["start"] for r in records], type=pa.int64()),
            "end": pa.array([r["end"] for r in records], type=pa.int64()),
            "input_ids": pa.array(
                [r["input_ids"] for r in records],
                type=pa.list_(pa.int32()),
            ),
            "length": pa.array([r["length"] for r in records], type=pa.int32()),
            "gc_content": pa.array(
                [r["gc_content"] for r in records], type=pa.float32()
            ),
        }
    )

    shard_path = output_dir / f"shard_{shard_idx:05d}.parquet"
    pq.write_table(table, shard_path, compression="zstd")
    logger.debug("Wrote shard {} ({} records)", shard_path.name, len(records))

    return sum(r["length"] for r in records)


def _save_metadata(
    output_dir: Path,
    tokenizer: GenomicTokenizer,
    config: Any,
    **stats: Any,
) -> None:
    """Write a JSON metadata file alongside the shards."""
    meta: Dict[str, Any] = {
        "tokenizer": {
            "mode": tokenizer.mode,
            "k": tokenizer.k,
            "stride": tokenizer.stride,
            "vocab_size": tokenizer.vocab_size,
        },
    }

    # Include DataConfig fields if available
    try:
        meta["config"] = asdict(config)
    except TypeError:
        meta["config"] = str(config)

    meta["stats"] = stats

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    logger.info("Metadata saved to {}", meta_path)
