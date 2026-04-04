"""Data quality reporting for genomic FASTA files and tokenised datasets.

Analyses sequence composition, detects potential contamination via k-mer
profiling, evaluates repeat content, and computes per-chromosome coverage
statistics.

Usage::

    from genova.data.quality_report import DataQualityReporter

    reporter = DataQualityReporter()
    report = reporter.analyze_fasta("genome.fa")
    reporter.generate_report("quality_report.md", format="markdown")
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# QualityReport
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Container for data quality analysis results.

    Attributes
    ----------
    file_path : str
        Analysed file path.
    num_sequences : int
        Total number of sequences / contigs.
    total_bases : int
        Total number of bases.
    sequence_lengths : dict
        Min, max, mean, median, N50 of sequence lengths.
    base_composition : dict
        Per-base counts and fractions.
    gc_content : dict
        Overall and per-sequence GC statistics.
    n_content : dict
        N-base statistics (total, fraction, longest run).
    kmer_analysis : dict
        K-mer frequency analysis for contamination detection.
    repeat_content : dict
        Simple repeat and low-complexity analysis.
    per_chromosome : dict
        Per-chromosome / per-contig breakdown.
    warnings : list of str
        Potential quality issues detected.
    """

    file_path: str = ""
    num_sequences: int = 0
    total_bases: int = 0
    sequence_lengths: Dict[str, Any] = field(default_factory=dict)
    base_composition: Dict[str, Any] = field(default_factory=dict)
    gc_content: Dict[str, Any] = field(default_factory=dict)
    n_content: Dict[str, Any] = field(default_factory=dict)
    kmer_analysis: Dict[str, Any] = field(default_factory=dict)
    repeat_content: Dict[str, Any] = field(default_factory=dict)
    per_chromosome: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the report as Markdown."""
        lines: List[str] = []
        lines.append("# Data Quality Report")
        lines.append("")
        lines.append(f"**File:** `{self.file_path}`  ")
        lines.append(f"**Sequences:** {self.num_sequences:,}  ")
        lines.append(f"**Total Bases:** {self.total_bases:,}")
        lines.append("")

        # Warnings
        if self.warnings:
            lines.append("## Warnings")
            lines.append("")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        # Sequence lengths
        if self.sequence_lengths:
            lines.append("## Sequence Length Distribution")
            lines.append("")
            lines.append("| Statistic | Value |")
            lines.append("|-----------|-------|")
            for k, v in self.sequence_lengths.items():
                display = f"{v:,.0f}" if isinstance(v, (int, float)) else str(v)
                lines.append(f"| {k} | {display} |")
            lines.append("")

        # Base composition
        if self.base_composition:
            lines.append("## Base Composition")
            lines.append("")
            lines.append("| Base | Count | Fraction |")
            lines.append("|------|-------|----------|")
            for base in "ACGTN":
                count = self.base_composition.get(f"{base}_count", 0)
                frac = self.base_composition.get(f"{base}_fraction", 0)
                lines.append(f"| {base} | {count:,} | {frac:.4f} |")
            lines.append("")

        # GC content
        if self.gc_content:
            lines.append("## GC Content")
            lines.append("")
            lines.append(f"- **Overall GC:** {self.gc_content.get('overall_gc', 0):.4f}")
            lines.append(f"- **Mean per-sequence GC:** {self.gc_content.get('mean_gc', 0):.4f}")
            lines.append(f"- **Std per-sequence GC:** {self.gc_content.get('std_gc', 0):.4f}")
            lines.append("")

        # N content
        if self.n_content:
            lines.append("## N Content")
            lines.append("")
            lines.append(f"- **Total Ns:** {self.n_content.get('total_n', 0):,}")
            lines.append(f"- **N Fraction:** {self.n_content.get('n_fraction', 0):.6f}")
            lines.append(f"- **Longest N run:** {self.n_content.get('longest_n_run', 0):,}")
            lines.append("")

        # K-mer analysis
        if self.kmer_analysis:
            lines.append("## K-mer Analysis")
            lines.append("")
            lines.append(f"- **K:** {self.kmer_analysis.get('k', 0)}")
            lines.append(f"- **Unique k-mers:** {self.kmer_analysis.get('unique_kmers', 0):,}")
            lines.append(f"- **Expected (uniform):** {self.kmer_analysis.get('expected_uniform', 0):,}")
            top = self.kmer_analysis.get("top_kmers", [])
            if top:
                lines.append("")
                lines.append("**Top 10 k-mers:**")
                lines.append("")
                lines.append("| K-mer | Count | Frequency |")
                lines.append("|-------|-------|-----------|")
                for kmer, count, freq in top[:10]:
                    lines.append(f"| {kmer} | {count:,} | {freq:.6f} |")
            lines.append("")

            if self.kmer_analysis.get("contamination_flag"):
                lines.append(
                    "> **Warning:** Unexpected k-mer distribution detected. "
                    "Possible contamination."
                )
                lines.append("")

        # Repeat content
        if self.repeat_content:
            lines.append("## Repeat Content")
            lines.append("")
            lines.append(f"- **Dinucleotide repeats:** {self.repeat_content.get('dinuc_repeat_fraction', 0):.4f}")
            lines.append(f"- **Trinucleotide repeats:** {self.repeat_content.get('trinuc_repeat_fraction', 0):.4f}")
            lines.append(f"- **Low complexity fraction:** {self.repeat_content.get('low_complexity_fraction', 0):.4f}")
            lines.append("")

        # Per chromosome
        if self.per_chromosome:
            lines.append("## Per-Chromosome Summary")
            lines.append("")
            lines.append("| Chromosome | Length | GC | N Fraction |")
            lines.append("|------------|--------|-----|------------|")
            for chrom, data in sorted(self.per_chromosome.items()):
                length = data.get("length", 0)
                gc = data.get("gc", 0)
                nf = data.get("n_fraction", 0)
                lines.append(f"| {chrom} | {length:,} | {gc:.4f} | {nf:.6f} |")
            lines.append("")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Render the report as simple HTML."""
        try:
            import markdown

            return markdown.markdown(self.to_markdown(), extensions=["tables"])
        except ImportError:
            # Fallback: wrap Markdown in <pre>
            return f"<html><body><pre>{self.to_markdown()}</pre></body></html>"


# ---------------------------------------------------------------------------
# DataQualityReporter
# ---------------------------------------------------------------------------

class DataQualityReporter:
    """Analyse genomic data quality and generate reports.

    Parameters
    ----------
    kmer_k : int
        K-mer size for contamination analysis (default 4).
    max_sequences : int
        Maximum sequences to analyse (for very large files). -1 means all.
    """

    def __init__(self, kmer_k: int = 4, max_sequences: int = -1) -> None:
        self.kmer_k = kmer_k
        self.max_sequences = max_sequences
        self._report: Optional[QualityReport] = None

    # -- FASTA analysis ------------------------------------------------------

    def analyze_fasta(self, fasta_path: Union[str, Path]) -> QualityReport:
        """Analyse a FASTA file for quality metrics.

        Parameters
        ----------
        fasta_path : str or Path
            Path to the FASTA file.

        Returns
        -------
        QualityReport
        """
        fasta_path = Path(fasta_path)
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

        logger.info("Analysing FASTA: {}", fasta_path)
        sequences: Dict[str, str] = {}
        current_name: Optional[str] = None
        current_seq: List[str] = []

        with open(fasta_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    if current_name is not None:
                        sequences[current_name] = "".join(current_seq)
                    current_name = line[1:].split()[0]
                    current_seq = []
                    if 0 < self.max_sequences <= len(sequences):
                        break
                else:
                    current_seq.append(line.upper())

        if current_name is not None and (self.max_sequences < 0 or len(sequences) < self.max_sequences):
            sequences[current_name] = "".join(current_seq)

        return self._analyze_sequences(sequences, str(fasta_path))

    def _analyze_sequences(
        self, sequences: Dict[str, str], file_path: str
    ) -> QualityReport:
        """Core analysis on a dict of name->sequence."""
        report = QualityReport(file_path=file_path)
        report.num_sequences = len(sequences)

        if not sequences:
            report.warnings.append("No sequences found in file.")
            self._report = report
            return report

        all_seq = list(sequences.values())
        lengths = np.array([len(s) for s in all_seq])
        report.total_bases = int(np.sum(lengths))

        # Sequence lengths
        sorted_lengths = np.sort(lengths)[::-1]
        cumsum = np.cumsum(sorted_lengths)
        n50_idx = np.searchsorted(cumsum, report.total_bases / 2)
        n50 = int(sorted_lengths[min(n50_idx, len(sorted_lengths) - 1)])

        report.sequence_lengths = {
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "N50": n50,
            "total": report.total_bases,
        }

        # Base composition
        concat = "".join(all_seq)
        base_counts = Counter(concat)
        total = len(concat)
        for base in "ACGTN":
            count = base_counts.get(base, 0)
            report.base_composition[f"{base}_count"] = count
            report.base_composition[f"{base}_fraction"] = count / total if total > 0 else 0

        # GC content
        gc_total = base_counts.get("G", 0) + base_counts.get("C", 0)
        acgt_total = sum(base_counts.get(b, 0) for b in "ACGT")
        overall_gc = gc_total / acgt_total if acgt_total > 0 else 0

        per_seq_gc: List[float] = []
        for s in all_seq:
            acgt = sum(1 for c in s if c in "ACGT")
            gc = sum(1 for c in s if c in "GC")
            per_seq_gc.append(gc / acgt if acgt > 0 else 0)

        report.gc_content = {
            "overall_gc": overall_gc,
            "mean_gc": float(np.mean(per_seq_gc)),
            "std_gc": float(np.std(per_seq_gc)),
            "min_gc": float(np.min(per_seq_gc)),
            "max_gc": float(np.max(per_seq_gc)),
        }

        # N content
        n_count = base_counts.get("N", 0)
        longest_n = self._longest_run(concat, "N")
        report.n_content = {
            "total_n": n_count,
            "n_fraction": n_count / total if total > 0 else 0,
            "longest_n_run": longest_n,
        }

        if n_count / total > 0.05 if total > 0 else False:
            report.warnings.append(
                f"High N content: {n_count / total:.2%} of bases are N."
            )

        # K-mer analysis
        report.kmer_analysis = self._kmer_analysis(concat)

        # Repeat content
        report.repeat_content = self._repeat_analysis(concat)

        # Per-chromosome
        for name, seq in sequences.items():
            seq_len = len(seq)
            acgt = sum(1 for c in seq if c in "ACGT")
            gc = sum(1 for c in seq if c in "GC")
            n_count_chr = sum(1 for c in seq if c == "N")
            report.per_chromosome[name] = {
                "length": seq_len,
                "gc": gc / acgt if acgt > 0 else 0,
                "n_fraction": n_count_chr / seq_len if seq_len > 0 else 0,
            }

        # Warnings
        if overall_gc < 0.30:
            report.warnings.append(f"Low overall GC content: {overall_gc:.2%}")
        elif overall_gc > 0.65:
            report.warnings.append(f"High overall GC content: {overall_gc:.2%}")

        self._report = report
        return report

    # -- Tokenized data analysis ---------------------------------------------

    def analyze_tokenized_data(self, data_dir: Union[str, Path]) -> QualityReport:
        """Analyse a directory of tokenised data files.

        Looks for ``.npy``, ``.pt``, or ``.bin`` files and computes token
        distribution statistics.

        Parameters
        ----------
        data_dir : str or Path
            Directory containing tokenised data files.

        Returns
        -------
        QualityReport
        """
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {data_dir}")

        report = QualityReport(file_path=str(data_dir))
        token_counts: Counter = Counter()
        total_tokens = 0
        file_lengths: List[int] = []
        num_files = 0

        # Scan for numpy files
        for fp in sorted(data_dir.glob("*.npy")):
            try:
                arr = np.load(fp)
                flat = arr.flatten()
                token_counts.update(flat.tolist())
                total_tokens += len(flat)
                file_lengths.append(len(flat))
                num_files += 1
            except Exception as e:
                report.warnings.append(f"Failed to load {fp.name}: {e}")

        # Scan for .pt files (PyTorch)
        for fp in sorted(data_dir.glob("*.pt")):
            try:
                import torch

                data = torch.load(fp, map_location="cpu")
                if isinstance(data, torch.Tensor):
                    flat = data.flatten().numpy()
                    token_counts.update(flat.tolist())
                    total_tokens += len(flat)
                    file_lengths.append(len(flat))
                    num_files += 1
            except Exception as e:
                report.warnings.append(f"Failed to load {fp.name}: {e}")

        report.num_sequences = num_files
        report.total_bases = total_tokens

        if file_lengths:
            arr = np.array(file_lengths)
            report.sequence_lengths = {
                "min": int(np.min(arr)),
                "max": int(np.max(arr)),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
            }

        if token_counts:
            n_unique = len(token_counts)
            top_tokens = token_counts.most_common(20)
            report.kmer_analysis = {
                "unique_tokens": n_unique,
                "total_tokens": total_tokens,
                "top_tokens": [
                    (str(t), c, c / total_tokens) for t, c in top_tokens
                ],
            }

        self._report = report
        return report

    # -- Report generation ---------------------------------------------------

    def generate_report(
        self,
        output_path: Union[str, Path],
        format: str = "markdown",
    ) -> None:
        """Write the last analysis result to a file.

        Parameters
        ----------
        output_path : str or Path
            Output file path.
        format : str
            ``"markdown"`` or ``"html"``.
        """
        if self._report is None:
            raise RuntimeError("No analysis has been run yet. Call analyze_fasta() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "html":
            content = self._report.to_html()
        else:
            content = self._report.to_markdown()

        output_path.write_text(content, encoding="utf-8")
        logger.info("Quality report saved to {}", output_path)

    # -- Internal helpers ----------------------------------------------------

    @staticmethod
    def _longest_run(seq: str, char: str) -> int:
        """Find the longest consecutive run of *char* in *seq*."""
        max_run = 0
        current = 0
        for c in seq:
            if c == char:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        return max_run

    def _kmer_analysis(self, seq: str) -> Dict[str, Any]:
        """Count k-mers and flag potential contamination."""
        k = self.kmer_k
        if len(seq) < k:
            return {}

        kmer_counts: Counter = Counter()
        for i in range(len(seq) - k + 1):
            kmer = seq[i: i + k]
            if "N" not in kmer:
                kmer_counts[kmer] += 1

        total_kmers = sum(kmer_counts.values())
        expected_uniform = 4 ** k
        unique_kmers = len(kmer_counts)

        top_kmers = [
            (kmer, count, count / total_kmers if total_kmers > 0 else 0)
            for kmer, count in kmer_counts.most_common(20)
        ]

        # Contamination heuristic: if one k-mer dominates > 5% that's unusual
        contamination_flag = False
        if top_kmers and top_kmers[0][2] > 0.05:
            contamination_flag = True

        return {
            "k": k,
            "unique_kmers": unique_kmers,
            "expected_uniform": expected_uniform,
            "total_kmers": total_kmers,
            "top_kmers": top_kmers,
            "contamination_flag": contamination_flag,
        }

    def _repeat_analysis(self, seq: str) -> Dict[str, Any]:
        """Analyse simple repeats and low-complexity regions."""
        if not seq:
            return {}

        total = len(seq)

        # Dinucleotide repeats (e.g., ATATAT...)
        dinuc_repeat = 0
        for i in range(0, total - 3):
            if seq[i] == seq[i + 2] and seq[i + 1] == seq[i + 3]:
                dinuc_repeat += 1

        # Trinucleotide repeats
        trinuc_repeat = 0
        for i in range(0, total - 5):
            if seq[i: i + 3] == seq[i + 3: i + 6]:
                trinuc_repeat += 1

        # Low complexity: runs of single base >= 10
        low_complexity = 0
        current_base = ""
        current_run = 0
        for c in seq:
            if c == current_base:
                current_run += 1
            else:
                if current_run >= 10:
                    low_complexity += current_run
                current_base = c
                current_run = 1
        if current_run >= 10:
            low_complexity += current_run

        return {
            "dinuc_repeat_fraction": dinuc_repeat / total if total > 0 else 0,
            "trinuc_repeat_fraction": trinuc_repeat / total if total > 0 else 0,
            "low_complexity_fraction": low_complexity / total if total > 0 else 0,
            "low_complexity_bases": low_complexity,
        }
