"""Generation quality metrics for synthetic DNA sequences.

Evaluates the quality and realism of generated DNA sequences by comparing
them against real genomic data across multiple criteria: nucleotide
composition, GC content distribution, k-mer frequency spectra, motif
enrichment, and trivial repeat detection.

Example::

    from genova.generative.evaluation import GenerationEvaluator

    evaluator = GenerationEvaluator(reference_sequences=real_seqs)
    metrics = evaluator.evaluate(generated_sequences)
    all_metrics = evaluator.compute_all_metrics(generated_sequences)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _count_nucleotides(sequence: str) -> Dict[str, int]:
    """Count occurrences of each nucleotide in a sequence."""
    seq_upper = sequence.upper()
    return {nuc: seq_upper.count(nuc) for nuc in "ACGTN"}


def _gc_content(sequence: str) -> float:
    """Compute GC content fraction for a single sequence."""
    seq_upper = sequence.upper()
    gc = seq_upper.count("G") + seq_upper.count("C")
    total = len(seq_upper)
    return gc / max(total, 1)


def _kmer_frequencies(sequence: str, k: int) -> Dict[str, float]:
    """Compute normalised k-mer frequency distribution."""
    seq_upper = sequence.upper()
    counts: Counter = Counter()
    total = max(len(seq_upper) - k + 1, 1)
    for i in range(len(seq_upper) - k + 1):
        kmer = seq_upper[i : i + k]
        if "N" not in kmer:
            counts[kmer] += 1
    return {kmer: count / total for kmer, count in counts.items()}


def _jensen_shannon_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    all_keys = set(p.keys()) | set(q.keys())
    p_arr = np.array([p.get(k, 0.0) for k in all_keys])
    q_arr = np.array([q.get(k, 0.0) for k in all_keys])

    # Normalise
    p_sum = p_arr.sum()
    q_sum = q_arr.sum()
    if p_sum > 0:
        p_arr = p_arr / p_sum
    if q_sum > 0:
        q_arr = q_arr / q_sum

    m = 0.5 * (p_arr + q_arr)

    # KL divergence with epsilon for numerical stability
    eps = 1e-12
    kl_pm = np.sum(p_arr * np.log((p_arr + eps) / (m + eps)))
    kl_qm = np.sum(q_arr * np.log((q_arr + eps) / (m + eps)))

    return float(0.5 * (kl_pm + kl_qm))


def _detect_trivial_repeats(
    sequence: str,
    min_repeat_len: int = 1,
    max_repeat_len: int = 6,
    min_fraction: float = 0.5,
) -> Dict[str, Any]:
    """Detect if a sequence is dominated by trivial tandem repeats.

    Args:
        sequence: DNA sequence string.
        min_repeat_len: Minimum repeat unit length to check.
        max_repeat_len: Maximum repeat unit length to check.
        min_fraction: If the best repeat covers more than this fraction
            of the sequence, it is flagged as trivial.

    Returns:
        Dict with ``"is_trivial"`` bool, ``"repeat_unit"``, ``"fraction"``.
    """
    seq_upper = sequence.upper()
    best_unit = ""
    best_fraction = 0.0

    for rlen in range(min_repeat_len, min_repeat_len + max_repeat_len):
        if rlen > len(seq_upper):
            break
        unit = seq_upper[:rlen]
        # Count how many consecutive repeats from the start
        repeated = unit * (len(seq_upper) // rlen + 1)
        match_count = sum(
            1 for a, b in zip(seq_upper, repeated) if a == b
        )
        fraction = match_count / max(len(seq_upper), 1)
        if fraction > best_fraction:
            best_fraction = fraction
            best_unit = unit

    return {
        "is_trivial": best_fraction >= min_fraction,
        "repeat_unit": best_unit,
        "fraction": best_fraction,
    }


# ---------------------------------------------------------------------------
# Generation evaluator
# ---------------------------------------------------------------------------


class GenerationEvaluator:
    """Comprehensive evaluation of generated DNA sequence quality.

    Compares generated sequences against optional reference (real) sequences
    across multiple metrics:

    - **Nucleotide composition**: Distribution of A, C, G, T, N.
    - **GC content**: Mean and standard deviation of GC fraction.
    - **K-mer frequencies**: Jensen-Shannon divergence of k-mer spectra.
    - **Motif enrichment**: Presence of known biological motifs.
    - **Trivial repeat detection**: Fraction of sequences with trivial repeats.

    Args:
        reference_sequences: Optional list of real DNA sequences for
            comparison.
        kmer_sizes: List of k-mer sizes to evaluate.
        motifs: Dict mapping motif name to regex pattern.
    """

    # Common biological motifs (regex patterns)
    DEFAULT_MOTIFS: Dict[str, str] = {
        "TATA_box": r"TATA[AT]A[AT]",
        "CpG": r"CG",
        "CAAT_box": r"GG[CT]CAATCT",
        "GC_box": r"GGGCGG",
        "E_box": r"CA[ACGT]{2}TG",
        "AP1": r"TGA[CG]TCA",
        "SP1": r"GG[GA]GG[GA]",
        "KOZAK": r"[AG]CCATGG",
    }

    def __init__(
        self,
        reference_sequences: Optional[List[str]] = None,
        kmer_sizes: Optional[List[int]] = None,
        motifs: Optional[Dict[str, str]] = None,
    ) -> None:
        self.reference_sequences = reference_sequences
        self.kmer_sizes = kmer_sizes or [3, 4, 6]
        self.motifs = motifs if motifs is not None else dict(self.DEFAULT_MOTIFS)

        # Pre-compute reference statistics
        self._ref_gc_stats: Optional[Dict[str, float]] = None
        self._ref_nuc_fracs: Optional[Dict[str, float]] = None
        self._ref_kmer_freqs: Optional[Dict[int, Dict[str, float]]] = None

        if self.reference_sequences:
            self._precompute_reference_stats()

    def _precompute_reference_stats(self) -> None:
        """Pre-compute statistics for the reference sequences."""
        ref = self.reference_sequences
        assert ref is not None

        # GC content
        gc_values = [_gc_content(s) for s in ref]
        self._ref_gc_stats = {
            "mean": float(np.mean(gc_values)),
            "std": float(np.std(gc_values)),
            "median": float(np.median(gc_values)),
        }

        # Nucleotide fractions
        total_counts: Dict[str, int] = {n: 0 for n in "ACGTN"}
        for s in ref:
            for nuc, cnt in _count_nucleotides(s).items():
                total_counts[nuc] += cnt
        total = sum(total_counts.values())
        self._ref_nuc_fracs = {n: c / max(total, 1) for n, c in total_counts.items()}

        # K-mer frequencies (pooled across all reference sequences)
        self._ref_kmer_freqs = {}
        for k in self.kmer_sizes:
            pooled: Counter = Counter()
            total_kmers = 0
            for s in ref:
                for kmer, freq in _kmer_frequencies(s, k).items():
                    pooled[kmer] += freq
                    total_kmers += 1
            total_sum = sum(pooled.values())
            self._ref_kmer_freqs[k] = {
                km: c / max(total_sum, 1e-12) for km, c in pooled.items()
            }

        logger.info(
            "Reference stats computed: {} sequences, mean GC={:.3f}",
            len(ref),
            self._ref_gc_stats["mean"],
        )

    # -- individual metrics --------------------------------------------------

    def nucleotide_composition(
        self,
        sequences: List[str],
    ) -> Dict[str, Any]:
        """Compute nucleotide composition statistics.

        Args:
            sequences: Generated DNA sequences.

        Returns:
            Dict with ``"fractions"`` and ``"valid"`` (no unknown nucleotides
            beyond N).
        """
        total_counts: Dict[str, int] = {n: 0 for n in "ACGTN"}
        invalid_count = 0

        for seq in sequences:
            counts = _count_nucleotides(seq)
            for nuc, cnt in counts.items():
                total_counts[nuc] += cnt
            # Check for invalid characters
            valid_chars = set("ACGTN")
            if set(seq.upper()) - valid_chars:
                invalid_count += 1

        total = sum(total_counts.values())
        fractions = {n: c / max(total, 1) for n, c in total_counts.items()}

        result: Dict[str, Any] = {
            "fractions": fractions,
            "n_fraction": fractions.get("N", 0.0),
            "valid_sequences": len(sequences) - invalid_count,
            "invalid_sequences": invalid_count,
            "validity_rate": (len(sequences) - invalid_count) / max(len(sequences), 1),
        }

        if self._ref_nuc_fracs is not None:
            # L1 distance from reference
            l1 = sum(
                abs(fractions.get(n, 0) - self._ref_nuc_fracs.get(n, 0))
                for n in "ACGT"
            )
            result["composition_l1_distance"] = l1

        return result

    def gc_content_analysis(
        self,
        sequences: List[str],
    ) -> Dict[str, Any]:
        """Analyse GC content distribution.

        Args:
            sequences: Generated DNA sequences.

        Returns:
            Dict with ``"mean"``, ``"std"``, ``"median"``, and optional
            comparison to reference.
        """
        gc_values = np.array([_gc_content(s) for s in sequences])

        result: Dict[str, Any] = {
            "mean": float(gc_values.mean()),
            "std": float(gc_values.std()),
            "median": float(np.median(gc_values)),
            "min": float(gc_values.min()),
            "max": float(gc_values.max()),
            "values": gc_values,
        }

        if self._ref_gc_stats is not None:
            result["ref_mean"] = self._ref_gc_stats["mean"]
            result["ref_std"] = self._ref_gc_stats["std"]
            result["mean_difference"] = abs(
                result["mean"] - self._ref_gc_stats["mean"]
            )

        return result

    def kmer_analysis(
        self,
        sequences: List[str],
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compare k-mer frequency distributions.

        Args:
            sequences: Generated DNA sequences.
            k: K-mer size.  If ``None``, uses all sizes in ``self.kmer_sizes``.

        Returns:
            Dict mapping k-mer size to JSD and top enriched/depleted k-mers.
        """
        sizes = [k] if k is not None else self.kmer_sizes
        result: Dict[str, Any] = {}

        for ksize in sizes:
            # Pool generated k-mer frequencies
            gen_pooled: Counter = Counter()
            for s in sequences:
                for kmer, freq in _kmer_frequencies(s, ksize).items():
                    gen_pooled[kmer] += freq
            total = sum(gen_pooled.values())
            gen_freqs = {km: c / max(total, 1e-12) for km, c in gen_pooled.items()}

            entry: Dict[str, Any] = {
                "num_unique_kmers": len(gen_freqs),
                "top_kmers": dict(
                    sorted(gen_freqs.items(), key=lambda x: -x[1])[:10]
                ),
            }

            if self._ref_kmer_freqs and ksize in self._ref_kmer_freqs:
                jsd = _jensen_shannon_divergence(gen_freqs, self._ref_kmer_freqs[ksize])
                entry["jsd"] = jsd

                # Most enriched and depleted k-mers
                diffs = {}
                all_kmers = set(gen_freqs.keys()) | set(self._ref_kmer_freqs[ksize].keys())
                for km in all_kmers:
                    diff = gen_freqs.get(km, 0) - self._ref_kmer_freqs[ksize].get(km, 0)
                    diffs[km] = diff
                sorted_diffs = sorted(diffs.items(), key=lambda x: -abs(x[1]))
                entry["most_enriched"] = dict(sorted_diffs[:5])
                entry["most_depleted"] = dict(sorted_diffs[-5:])

            result[f"{ksize}mer"] = entry

        return result

    def motif_enrichment(
        self,
        sequences: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Analyse enrichment of known biological motifs.

        Args:
            sequences: Generated DNA sequences.

        Returns:
            Dict mapping motif name to occurrence stats.
        """
        results: Dict[str, Dict[str, Any]] = {}

        for motif_name, pattern in self.motifs.items():
            total_hits = 0
            seqs_with_hit = 0

            for seq in sequences:
                hits = len(re.findall(pattern, seq.upper()))
                total_hits += hits
                if hits > 0:
                    seqs_with_hit += 1

            n_seqs = max(len(sequences), 1)
            mean_length = np.mean([len(s) for s in sequences]) if sequences else 0

            results[motif_name] = {
                "total_hits": total_hits,
                "sequences_with_motif": seqs_with_hit,
                "motif_rate": seqs_with_hit / n_seqs,
                "hits_per_kb": total_hits / (n_seqs * mean_length / 1000) if mean_length > 0 else 0,
                "pattern": pattern,
            }

        return results

    def trivial_repeat_analysis(
        self,
        sequences: List[str],
        min_fraction: float = 0.5,
    ) -> Dict[str, Any]:
        """Detect sequences dominated by trivial tandem repeats.

        Args:
            sequences: Generated DNA sequences.
            min_fraction: Threshold fraction to flag a sequence as trivial.

        Returns:
            Dict with ``"trivial_count"``, ``"trivial_rate"``, and per-sequence
            details.
        """
        trivial_count = 0
        details: List[Dict[str, Any]] = []

        for i, seq in enumerate(sequences):
            result = _detect_trivial_repeats(seq, min_fraction=min_fraction)
            if result["is_trivial"]:
                trivial_count += 1
            details.append({"index": i, **result})

        n = max(len(sequences), 1)
        return {
            "trivial_count": trivial_count,
            "trivial_rate": trivial_count / n,
            "non_trivial_rate": (n - trivial_count) / n,
            "details": details,
        }

    # -- aggregate methods ---------------------------------------------------

    def evaluate(
        self,
        sequences: List[str],
    ) -> Dict[str, Any]:
        """Run a quick evaluation of generated sequences.

        Computes nucleotide composition, GC content, and trivial repeat
        rate.  For a comprehensive evaluation, use :meth:`compute_all_metrics`.

        Args:
            sequences: Generated DNA sequences.

        Returns:
            Summary dict with key metrics.
        """
        nuc = self.nucleotide_composition(sequences)
        gc = self.gc_content_analysis(sequences)
        repeats = self.trivial_repeat_analysis(sequences)

        summary: Dict[str, Any] = {
            "num_sequences": len(sequences),
            "mean_length": float(np.mean([len(s) for s in sequences])) if sequences else 0,
            "validity_rate": nuc["validity_rate"],
            "gc_content_mean": gc["mean"],
            "gc_content_std": gc["std"],
            "trivial_repeat_rate": repeats["trivial_rate"],
            "n_fraction": nuc["n_fraction"],
        }

        if "composition_l1_distance" in nuc:
            summary["composition_l1_distance"] = nuc["composition_l1_distance"]
        if "mean_difference" in gc:
            summary["gc_mean_difference"] = gc["mean_difference"]

        logger.info(
            "Quick evaluation: {} seqs, validity={:.3f}, GC={:.3f}, "
            "trivial_rate={:.3f}",
            len(sequences),
            summary["validity_rate"],
            summary["gc_content_mean"],
            summary["trivial_repeat_rate"],
        )

        return summary

    def compute_all_metrics(
        self,
        sequences: List[str],
    ) -> Dict[str, Any]:
        """Comprehensive evaluation across all metrics.

        Args:
            sequences: Generated DNA sequences.

        Returns:
            Dict with sub-dicts for each metric category.
        """
        result: Dict[str, Any] = {
            "summary": self.evaluate(sequences),
            "nucleotide_composition": self.nucleotide_composition(sequences),
            "gc_content": self.gc_content_analysis(sequences),
            "kmer_analysis": self.kmer_analysis(sequences),
            "motif_enrichment": self.motif_enrichment(sequences),
            "trivial_repeats": self.trivial_repeat_analysis(sequences),
        }

        # Remove numpy arrays from gc_content for serialisability
        if "values" in result["gc_content"]:
            result["gc_content"]["values"] = result["gc_content"]["values"].tolist()

        logger.info("Full evaluation completed for {} sequences", len(sequences))
        return result
