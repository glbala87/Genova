#!/usr/bin/env python3
"""Prepare training data from downloaded reference genome.

Processes a FASTA reference genome (e.g. GRCh38) into tokenised training
samples for the Genova foundation model. Chromosomes are split into
train/val/test sets, windowed, tokenised, MLM-masked, and saved as
Parquet shards for efficient data loading during training.

Usage:
    python scripts/prepare_training_data.py \\
        --fasta data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa \\
        --output-dir data/prepared \\
        --window-size 2048 \\
        --stride 512 \\
        --tokenizer-mode kmer \\
        --kmer-size 6 \\
        --num-workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prepare_data")

# ---------------------------------------------------------------------------
# Chromosome split definitions
# ---------------------------------------------------------------------------

# Standard genomic ML split (Enformer / Basenji convention)
#   Train: chr1-chr17
#   Val:   chr18-chr20
#   Test:  chr21-chr22
TRAIN_CHROMS = [f"chr{i}" for i in range(1, 18)] + [str(i) for i in range(1, 18)]
VAL_CHROMS = [f"chr{i}" for i in range(18, 21)] + [str(i) for i in range(18, 21)]
TEST_CHROMS = [f"chr{i}" for i in range(21, 23)] + [str(i) for i in range(21, 23)]


def classify_chromosome(chrom: str) -> Optional[str]:
    """Classify a chromosome name into train/val/test split.

    Args:
        chrom: Chromosome name as it appears in the FASTA.

    Returns:
        Split name ('train', 'val', 'test') or None if not a standard
        autosome (e.g. chrX, chrY, chrM, unplaced scaffolds).
    """
    if chrom in TRAIN_CHROMS:
        return "train"
    if chrom in VAL_CHROMS:
        return "val"
    if chrom in TEST_CHROMS:
        return "test"
    return None


# ---------------------------------------------------------------------------
# Sequence windowing
# ---------------------------------------------------------------------------

def extract_windows(
    sequence: str,
    chrom: str,
    window_size: int,
    stride: int,
    max_n_fraction: float = 0.1,
) -> List[Dict[str, Any]]:
    """Extract sliding windows from a chromosome sequence.

    Args:
        sequence: Full chromosome DNA string (uppercase).
        chrom: Chromosome name for metadata.
        window_size: Length of each window in base pairs.
        stride: Step size between consecutive windows.
        max_n_fraction: Skip windows with more than this fraction of N bases.

    Returns:
        List of dicts with 'sequence', 'chrom', 'start', 'end' keys.
    """
    windows = []
    seq_len = len(sequence)

    if seq_len < window_size:
        logger.warning(
            "Chromosome %s is shorter than window_size (%d < %d), skipping.",
            chrom, seq_len, window_size,
        )
        return windows

    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        subseq = sequence[start:end]

        # Filter windows with too many N bases
        n_count = subseq.count("N")
        if n_count / window_size > max_n_fraction:
            continue

        windows.append({
            "chrom": chrom,
            "start": start,
            "end": end,
            "sequence": subseq,
        })

    return windows


# ---------------------------------------------------------------------------
# Tokenisation and MLM sample generation
# ---------------------------------------------------------------------------

def tokenise_windows(
    windows: List[Dict[str, Any]],
    tokenizer_mode: str,
    kmer_size: int,
    mask_prob: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """Tokenise and apply MLM masking to a list of windows.

    This function imports Genova's tokenizer to ensure consistency with
    the training pipeline.

    Args:
        windows: List of window dicts with 'sequence' key.
        tokenizer_mode: 'nucleotide' or 'kmer'.
        kmer_size: k-mer size (only used if tokenizer_mode='kmer').
        mask_prob: Fraction of tokens to mask for MLM.
        max_tokens: Maximum token sequence length.

    Returns:
        List of dicts with 'input_ids', 'attention_mask', 'labels',
        and metadata fields.
    """
    import random
    from genova.data.tokenizer import GenomicTokenizer, SPECIAL_TOKENS

    # Build tokenizer and vocabulary
    tokenizer = GenomicTokenizer(
        mode=tokenizer_mode,
        k=kmer_size,
        stride=1,
        add_special_tokens=True,
    )
    tokenizer.build_vocab()

    special_ids = set(SPECIAL_TOKENS.values())
    rng = random.Random(42)
    non_special_ids = [
        tid for tid in tokenizer.id_to_token if tid not in special_ids
    ]

    samples = []
    for win in windows:
        seq = win["sequence"]
        input_ids = tokenizer.encode(seq, max_length=max_tokens)

        # Apply MLM masking (BERT-style: 80% MASK, 10% random, 10% keep)
        masked_ids = list(input_ids)
        labels = [-100] * len(input_ids)

        candidates = [
            i for i, tid in enumerate(input_ids) if tid not in special_ids
        ]
        n_mask = max(1, int(len(candidates) * mask_prob))
        chosen = rng.sample(candidates, min(n_mask, len(candidates)))

        for pos in chosen:
            labels[pos] = input_ids[pos]
            r = rng.random()
            if r < 0.8:
                masked_ids[pos] = tokenizer.mask_token_id
            elif r < 0.9:
                masked_ids[pos] = rng.choice(non_special_ids)
            # else: keep original (10%)

        attention_mask = [1] * len(masked_ids)

        samples.append({
            "input_ids": masked_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "chrom": win["chrom"],
            "start": win["start"],
            "end": win["end"],
            "seq_length": len(masked_ids),
        })

    return samples


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------

def process_chromosome(
    fasta_path: str,
    chrom: str,
    window_size: int,
    stride: int,
    tokenizer_mode: str,
    kmer_size: int,
    mask_prob: float,
    max_tokens: int,
    max_n_fraction: float,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Process a single chromosome: extract windows and tokenise.

    Designed to be called in a subprocess via ProcessPoolExecutor.

    Args:
        fasta_path: Path to the FASTA file.
        chrom: Name of the chromosome to process.
        window_size: Window size in base pairs.
        stride: Stride in base pairs.
        tokenizer_mode: Tokenizer mode.
        kmer_size: k-mer size.
        mask_prob: MLM mask probability.
        max_tokens: Max token length.
        max_n_fraction: Max N-base fraction per window.

    Returns:
        Tuple of (chrom_name, list_of_sample_dicts).
    """
    import pyfaidx

    fasta = pyfaidx.Fasta(fasta_path, read_ahead=10_000)

    if chrom not in fasta:
        return chrom, []

    sequence = str(fasta[chrom][:]).upper()
    fasta.close()

    # Extract windows
    windows = extract_windows(
        sequence, chrom, window_size, stride, max_n_fraction
    )

    if not windows:
        return chrom, []

    # Tokenise
    samples = tokenise_windows(
        windows, tokenizer_mode, kmer_size, mask_prob, max_tokens
    )

    return chrom, samples


# ---------------------------------------------------------------------------
# Parquet shard writer
# ---------------------------------------------------------------------------

def save_parquet_shards(
    samples: List[Dict[str, Any]],
    output_dir: Path,
    split: str,
    shard_size: int = 50_000,
) -> int:
    """Save tokenised samples as Parquet shards.

    Args:
        samples: List of sample dicts.
        output_dir: Root output directory.
        split: Split name ('train', 'val', 'test').
        shard_size: Maximum number of samples per shard.

    Returns:
        Number of shards written.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.error(
            "pyarrow is required for Parquet output. "
            "Install with: pip install pyarrow"
        )
        # Fall back to JSON lines
        return _save_jsonl_fallback(samples, output_dir, split, shard_size)

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    n_shards = 0
    for start in range(0, len(samples), shard_size):
        chunk = samples[start : start + shard_size]
        shard_path = split_dir / f"shard_{n_shards:05d}.parquet"

        # Build columnar arrays
        table = pa.table({
            "input_ids": [s["input_ids"] for s in chunk],
            "attention_mask": [s["attention_mask"] for s in chunk],
            "labels": [s["labels"] for s in chunk],
            "chrom": [s["chrom"] for s in chunk],
            "start": [s["start"] for s in chunk],
            "end": [s["end"] for s in chunk],
            "seq_length": [s["seq_length"] for s in chunk],
        })

        pq.write_table(table, shard_path, compression="snappy")
        n_shards += 1
        logger.info(
            "  Wrote shard %s (%d samples)", shard_path.name, len(chunk)
        )

    return n_shards


def _save_jsonl_fallback(
    samples: List[Dict[str, Any]],
    output_dir: Path,
    split: str,
    shard_size: int,
) -> int:
    """Fallback: save as JSON lines when pyarrow is not available."""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    n_shards = 0
    for start in range(0, len(samples), shard_size):
        chunk = samples[start : start + shard_size]
        shard_path = split_dir / f"shard_{n_shards:05d}.jsonl"

        with open(shard_path, "w") as fh:
            for s in chunk:
                fh.write(json.dumps(s) + "\n")

        n_shards += 1
        logger.info(
            "  Wrote shard %s (%d samples, JSONL fallback)",
            shard_path.name, len(chunk),
        )

    return n_shards


# ---------------------------------------------------------------------------
# BED file generation
# ---------------------------------------------------------------------------

def write_bed_file(
    samples: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Write a BED file listing all regions in the sample set.

    This BED file can be used by GenomeDataset during training.

    Args:
        samples: List of sample dicts with chrom/start/end keys.
        output_path: Path for the BED file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with open(output_path, "w") as fh:
        for s in samples:
            key = (s["chrom"], s["start"], s["end"])
            if key not in seen:
                fh.write(f"{s['chrom']}\t{s['start']}\t{s['end']}\n")
                seen.add(key)
    logger.info("BED file written: %s (%d regions)", output_path, len(seen))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare training data from a reference genome for Genova.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python scripts/prepare_training_data.py --fasta data/reference/hg38.fa

  # Nucleotide-level tokenisation with larger windows
  python scripts/prepare_training_data.py \\
      --fasta data/reference/hg38.fa \\
      --tokenizer-mode nucleotide \\
      --window-size 4000 \\
      --stride 1000

  # k-mer tokenisation for Mamba long-context model
  python scripts/prepare_training_data.py \\
      --fasta data/reference/hg38.fa \\
      --tokenizer-mode kmer --kmer-size 6 \\
      --window-size 10000 --stride 2500
        """,
    )

    parser.add_argument(
        "--fasta", type=str, required=True,
        help="Path to the reference genome FASTA file (indexed with .fai).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/prepared",
        help="Output directory for prepared data (default: data/prepared).",
    )
    parser.add_argument(
        "--window-size", type=int, default=2048,
        help="Sliding window size in base pairs (default: 2048).",
    )
    parser.add_argument(
        "--stride", type=int, default=512,
        help="Stride between windows in base pairs (default: 512).",
    )
    parser.add_argument(
        "--tokenizer-mode", type=str, default="nucleotide",
        choices=["nucleotide", "kmer"],
        help="Tokenizer mode (default: nucleotide).",
    )
    parser.add_argument(
        "--kmer-size", type=int, default=6,
        help="k-mer size for kmer tokenizer mode (default: 6).",
    )
    parser.add_argument(
        "--mask-prob", type=float, default=0.15,
        help="MLM mask probability (default: 0.15).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="Maximum token sequence length (default: 1024).",
    )
    parser.add_argument(
        "--max-n-fraction", type=float, default=0.1,
        help="Max fraction of N bases allowed per window (default: 0.1).",
    )
    parser.add_argument(
        "--shard-size", type=int, default=50000,
        help="Max samples per Parquet shard (default: 50000).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of parallel workers for chromosome processing (default: 4).",
    )
    parser.add_argument(
        "--chromosomes", type=str, nargs="*", default=None,
        help="Specific chromosomes to process (default: all autosomes).",
    )

    args = parser.parse_args()

    # Validate inputs
    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        logger.error("FASTA file not found: %s", fasta_path)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Genova Training Data Preparation")
    logger.info("=" * 70)
    logger.info("FASTA:          %s", fasta_path)
    logger.info("Output:         %s", output_dir)
    logger.info("Window size:    %d bp", args.window_size)
    logger.info("Stride:         %d bp", args.stride)
    logger.info("Tokenizer:      %s (k=%d)", args.tokenizer_mode, args.kmer_size)
    logger.info("Mask prob:      %.2f", args.mask_prob)
    logger.info("Max tokens:     %d", args.max_tokens)
    logger.info("Num workers:    %d", args.num_workers)
    logger.info("")

    # Discover chromosomes in the FASTA
    try:
        import pyfaidx
        fasta = pyfaidx.Fasta(str(fasta_path))
        all_chroms = list(fasta.keys())
        fasta.close()
    except ImportError:
        logger.error(
            "pyfaidx is required. Install with: pip install pyfaidx"
        )
        sys.exit(1)

    # Filter to requested chromosomes
    if args.chromosomes:
        chroms_to_process = [c for c in args.chromosomes if c in all_chroms]
    else:
        # Process all autosomes (chr1-22 or 1-22)
        chroms_to_process = [c for c in all_chroms if classify_chromosome(c) is not None]

    if not chroms_to_process:
        logger.error("No valid chromosomes found in FASTA.")
        sys.exit(1)

    logger.info("Chromosomes to process: %s", chroms_to_process)
    logger.info("")

    # Process chromosomes in parallel
    split_samples: Dict[str, List[Dict[str, Any]]] = {
        "train": [], "val": [], "test": [],
    }

    start_time = time.time()
    n_workers = min(args.num_workers, len(chroms_to_process))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for chrom in chroms_to_process:
            future = executor.submit(
                process_chromosome,
                str(fasta_path),
                chrom,
                args.window_size,
                args.stride,
                args.tokenizer_mode,
                args.kmer_size,
                args.mask_prob,
                args.max_tokens,
                args.max_n_fraction,
            )
            futures[future] = chrom

        for future in as_completed(futures):
            chrom = futures[future]
            try:
                chrom_name, samples = future.result()
                split = classify_chromosome(chrom_name)
                if split is not None and samples:
                    split_samples[split].extend(samples)
                    logger.info(
                        "Processed %s: %d windows -> %s split",
                        chrom_name, len(samples), split,
                    )
                elif not samples:
                    logger.warning("Chromosome %s yielded no windows.", chrom_name)
            except Exception as exc:
                logger.error("Error processing %s: %s", chrom, exc)

    elapsed = time.time() - start_time
    logger.info("")
    logger.info("Window extraction completed in %.1f seconds.", elapsed)

    # Save Parquet shards and BED files
    logger.info("")
    logger.info("--- Saving Parquet shards ---")

    total_shards = 0
    for split_name, samples in split_samples.items():
        if not samples:
            logger.warning("No samples for %s split.", split_name)
            continue

        logger.info("Saving %s split: %d samples", split_name, len(samples))
        n_shards = save_parquet_shards(
            samples, output_dir, split_name, shard_size=args.shard_size,
        )
        total_shards += n_shards

        # Write BED file for this split
        bed_path = output_dir / f"{split_name}_regions.bed"
        write_bed_file(samples, bed_path)

    # Save tokenizer configuration
    logger.info("")
    logger.info("--- Saving tokenizer ---")
    from genova.data.tokenizer import GenomicTokenizer

    tokenizer = GenomicTokenizer(
        mode=args.tokenizer_mode,
        k=args.kmer_size,
        stride=1,
        add_special_tokens=True,
    )
    tokenizer.build_vocab()
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    logger.info("Tokenizer saved to %s (vocab_size=%d)", tokenizer_path, tokenizer.vocab_size)

    # Save processing metadata
    stats = {
        "fasta": str(fasta_path),
        "window_size": args.window_size,
        "stride": args.stride,
        "tokenizer_mode": args.tokenizer_mode,
        "kmer_size": args.kmer_size,
        "mask_prob": args.mask_prob,
        "max_tokens": args.max_tokens,
        "max_n_fraction": args.max_n_fraction,
        "vocab_size": tokenizer.vocab_size,
        "splits": {},
        "total_shards": total_shards,
        "processing_time_s": round(elapsed, 1),
    }

    total_windows = 0
    total_tokens = 0
    for split_name, samples in split_samples.items():
        n_samples = len(samples)
        n_tokens = sum(s["seq_length"] for s in samples)
        stats["splits"][split_name] = {
            "n_samples": n_samples,
            "n_tokens": n_tokens,
        }
        total_windows += n_samples
        total_tokens += n_tokens

    stats["total_windows"] = total_windows
    stats["total_tokens"] = total_tokens

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)

    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("Preparation Complete")
    logger.info("=" * 70)
    logger.info("Total windows:   %d", total_windows)
    logger.info("Total tokens:    %d", total_tokens)
    logger.info("Parquet shards:  %d", total_shards)
    logger.info("Vocab size:      %d", tokenizer.vocab_size)
    logger.info("")
    for split_name in ("train", "val", "test"):
        info = stats["splits"].get(split_name, {})
        logger.info(
            "  %-6s  %8d windows  %12d tokens",
            split_name,
            info.get("n_samples", 0),
            info.get("n_tokens", 0),
        )
    logger.info("")
    logger.info("Output directory: %s", output_dir)
    logger.info("Stats file:       %s", stats_path)
    logger.info("Processing time:  %.1f s", elapsed)


if __name__ == "__main__":
    main()
