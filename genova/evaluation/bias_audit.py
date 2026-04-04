"""Bias audit framework for genomic models.

Evaluates model performance disparities across populations, GC-content bins,
repeat regions, and individual chromosomes.  Generates structured reports
with optional plots.

Usage::

    from genova.evaluation.bias_audit import BiasAuditor

    auditor = BiasAuditor(model, tokenizer)
    pop_report = auditor.audit_population_bias(model, data_by_population)
    gc_report  = auditor.audit_gc_bias(model, data, gc_bins=[0.3, 0.4, 0.5, 0.6, 0.7])
    chr_report = auditor.audit_chromosome_bias(model, data_by_chrom)
    full       = auditor.generate_report()
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class BiasReport:
    """Container for bias audit results.

    Attributes
    ----------
    population_bias : dict
        Per-population metrics and disparities.
    gc_bias : dict
        Per-GC-bin metrics.
    chromosome_bias : dict
        Per-chromosome metrics.
    repeat_bias : dict
        Repetitive vs unique region metrics.
    summary : str
        Human-readable summary.
    """

    population_bias: Dict[str, Any] = field(default_factory=dict)
    gc_bias: Dict[str, Any] = field(default_factory=dict)
    chromosome_bias: Dict[str, Any] = field(default_factory=dict)
    repeat_bias: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_markdown(self) -> str:
        """Render the full report as Markdown."""
        lines: List[str] = []
        lines.append("# Bias Audit Report")
        lines.append("")

        if self.summary:
            lines.append("## Executive Summary")
            lines.append("")
            lines.append(self.summary)
            lines.append("")

        # Population bias
        if self.population_bias:
            lines.append("## Population Bias")
            lines.append("")
            metrics_keys = set()
            for pop_data in self.population_bias.values():
                if isinstance(pop_data, dict):
                    metrics_keys.update(pop_data.keys())
            metrics_keys.discard("n_samples")
            metrics_keys = sorted(metrics_keys)

            header = "| Population | N |"
            sep = "|------------|---|"
            for mk in metrics_keys:
                header += f" {mk} |"
                sep += "------|"
            lines.append(header)
            lines.append(sep)

            for pop, data in sorted(self.population_bias.items()):
                if not isinstance(data, dict):
                    continue
                n = data.get("n_samples", "?")
                row = f"| {pop} | {n} |"
                for mk in metrics_keys:
                    val = data.get(mk, "")
                    if isinstance(val, float):
                        row += f" {val:.4f} |"
                    else:
                        row += f" {val} |"
                lines.append(row)
            lines.append("")

        # GC bias
        if self.gc_bias:
            lines.append("## GC Content Bias")
            lines.append("")
            lines.append("| GC Bin | N | Mean Score | Std |")
            lines.append("|--------|---|-----------|-----|")
            for gc_bin, data in sorted(self.gc_bias.items()):
                if not isinstance(data, dict):
                    continue
                n = data.get("n_samples", "?")
                mean = data.get("mean_score", 0)
                std = data.get("std_score", 0)
                lines.append(f"| {gc_bin} | {n} | {mean:.4f} | {std:.4f} |")
            lines.append("")

        # Chromosome bias
        if self.chromosome_bias:
            lines.append("## Chromosome Bias")
            lines.append("")
            metrics_keys = set()
            for chrom_data in self.chromosome_bias.values():
                if isinstance(chrom_data, dict):
                    metrics_keys.update(chrom_data.keys())
            metrics_keys.discard("n_samples")
            metrics_keys = sorted(metrics_keys)

            header = "| Chromosome | N |"
            sep = "|------------|---|"
            for mk in metrics_keys:
                header += f" {mk} |"
                sep += "------|"
            lines.append(header)
            lines.append(sep)

            for chrom, data in sorted(self.chromosome_bias.items(),
                                       key=lambda x: _chrom_sort_key(x[0])):
                if not isinstance(data, dict):
                    continue
                n = data.get("n_samples", "?")
                row = f"| {chrom} | {n} |"
                for mk in metrics_keys:
                    val = data.get(mk, "")
                    if isinstance(val, float):
                        row += f" {val:.4f} |"
                    else:
                        row += f" {val} |"
                lines.append(row)
            lines.append("")

        # Repeat bias
        if self.repeat_bias:
            lines.append("## Repeat Region Bias")
            lines.append("")
            lines.append("| Region Type | N | Mean Score | Std |")
            lines.append("|-------------|---|-----------|-----|")
            for rtype, data in self.repeat_bias.items():
                if not isinstance(data, dict):
                    continue
                n = data.get("n_samples", "?")
                mean = data.get("mean_score", 0)
                std = data.get("std_score", 0)
                lines.append(f"| {rtype} | {n} | {mean:.4f} | {std:.4f} |")
            lines.append("")

        return "\n".join(lines)

    def save(self, path: Union[str, Path], format: str = "markdown") -> None:
        """Save the report to a file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        format : str
            ``"markdown"`` or ``"json"``.
        """
        import json as json_mod

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = {
                "population_bias": self.population_bias,
                "gc_bias": self.gc_bias,
                "chromosome_bias": self.chromosome_bias,
                "repeat_bias": self.repeat_bias,
                "summary": self.summary,
            }
            path.write_text(json_mod.dumps(data, indent=2, default=str), encoding="utf-8")
        else:
            path.write_text(self.to_markdown(), encoding="utf-8")

        logger.info("Bias report saved to {}", path)

    def generate_plots(self, output_dir: Union[str, Path] = ".") -> List[Path]:
        """Generate bias audit plots (requires matplotlib).

        Returns a list of saved figure paths.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; skipping plot generation.")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        # Population bias bar chart
        if self.population_bias:
            fig, ax = plt.subplots(figsize=(10, 5))
            pops = sorted(self.population_bias.keys())
            scores = [
                self.population_bias[p].get("mean_score", 0)
                for p in pops
                if isinstance(self.population_bias[p], dict)
            ]
            pops_filtered = [
                p for p in pops if isinstance(self.population_bias[p], dict)
            ]
            ax.bar(pops_filtered, scores)
            ax.set_xlabel("Population")
            ax.set_ylabel("Mean Score")
            ax.set_title("Performance by Population")
            plt.tight_layout()
            p = output_dir / "population_bias.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            saved.append(p)

        # GC bias line plot
        if self.gc_bias:
            fig, ax = plt.subplots(figsize=(8, 5))
            bins = sorted(self.gc_bias.keys())
            means = [
                self.gc_bias[b].get("mean_score", 0)
                for b in bins
                if isinstance(self.gc_bias[b], dict)
            ]
            stds = [
                self.gc_bias[b].get("std_score", 0)
                for b in bins
                if isinstance(self.gc_bias[b], dict)
            ]
            bins_f = [
                b for b in bins if isinstance(self.gc_bias[b], dict)
            ]
            ax.errorbar(range(len(bins_f)), means, yerr=stds, marker="o", capsize=3)
            ax.set_xticks(range(len(bins_f)))
            ax.set_xticklabels(bins_f, rotation=45)
            ax.set_xlabel("GC Content Bin")
            ax.set_ylabel("Mean Score")
            ax.set_title("Performance by GC Content")
            plt.tight_layout()
            p = output_dir / "gc_bias.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            saved.append(p)

        # Chromosome bias
        if self.chromosome_bias:
            fig, ax = plt.subplots(figsize=(12, 5))
            chroms = sorted(
                [c for c in self.chromosome_bias if isinstance(self.chromosome_bias[c], dict)],
                key=_chrom_sort_key,
            )
            scores = [
                self.chromosome_bias[c].get("mean_score", 0) for c in chroms
            ]
            ax.bar(range(len(chroms)), scores)
            ax.set_xticks(range(len(chroms)))
            ax.set_xticklabels(chroms, rotation=45, fontsize=8)
            ax.set_xlabel("Chromosome")
            ax.set_ylabel("Mean Score")
            ax.set_title("Performance by Chromosome")
            plt.tight_layout()
            p = output_dir / "chromosome_bias.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            saved.append(p)

        return saved


def _chrom_sort_key(chrom: str) -> Tuple[int, str]:
    """Sort chromosomes numerically then alphabetically."""
    c = chrom.replace("chr", "")
    try:
        return (0, str(int(c)).zfill(3))
    except ValueError:
        return (1, c)


# ---------------------------------------------------------------------------
# BiasAuditor
# ---------------------------------------------------------------------------

class BiasAuditor:
    """Audit a genomic model for systematic biases.

    Parameters
    ----------
    model : object
        A model or :class:`~genova.api.inference.InferenceEngine` with
        ``embed()``, ``predict_variant()``, etc.
    score_fn : callable, optional
        Custom scoring function ``(model, sequences) -> np.ndarray`` that
        returns a 1-D array of scalar scores.  Defaults to L2-norm of
        embeddings.
    """

    def __init__(
        self,
        model: Any = None,
        score_fn: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self._score_fn = score_fn or self._default_score
        self._reports: Dict[str, Any] = {}

    @staticmethod
    def _default_score(model: Any, sequences: List[str]) -> np.ndarray:
        """Compute L2 norm of embeddings as a generic score."""
        if hasattr(model, "embed"):
            embeddings = model.embed(sequences)
            return np.array([np.linalg.norm(e) for e in embeddings])
        return np.zeros(len(sequences))

    # -- Population bias -----------------------------------------------------

    def audit_population_bias(
        self,
        model: Any,
        data_by_population: Dict[str, List[str]],
        labels_by_population: Optional[Dict[str, np.ndarray]] = None,
    ) -> BiasReport:
        """Evaluate performance disparities across populations.

        Parameters
        ----------
        model : object
            Model to evaluate.
        data_by_population : dict
            Mapping from population code (e.g. ``"EUR"``, ``"AFR"``) to
            lists of DNA sequences.
        labels_by_population : dict, optional
            Ground-truth labels per population for metric computation.

        Returns
        -------
        BiasReport
        """
        results: Dict[str, Any] = {}
        all_means: List[float] = []

        for pop, sequences in data_by_population.items():
            if not sequences:
                continue

            scores = self._score_fn(model, sequences)
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            all_means.append(mean_score)

            pop_result: Dict[str, Any] = {
                "n_samples": len(sequences),
                "mean_score": mean_score,
                "std_score": std_score,
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
            }

            # If labels are provided, compute accuracy-like metrics
            if labels_by_population and pop in labels_by_population:
                labels = labels_by_population[pop]
                preds = (scores > np.median(scores)).astype(int)
                if len(labels) == len(preds):
                    accuracy = float(np.mean(preds == labels))
                    pop_result["accuracy"] = accuracy

            results[pop] = pop_result

        # Disparity metric
        if len(all_means) > 1:
            max_gap = max(all_means) - min(all_means)
            cv = float(np.std(all_means) / np.mean(all_means)) if np.mean(all_means) != 0 else 0
        else:
            max_gap = 0.0
            cv = 0.0

        report = BiasReport(
            population_bias=results,
            summary=(
                f"Population bias audit across {len(results)} populations. "
                f"Max performance gap: {max_gap:.4f}. "
                f"Coefficient of variation: {cv:.4f}."
            ),
        )
        self._reports["population"] = report
        return report

    # -- GC content bias -----------------------------------------------------

    def audit_gc_bias(
        self,
        model: Any,
        data: List[str],
        gc_bins: Optional[List[float]] = None,
    ) -> BiasReport:
        """Evaluate performance across GC-content bins.

        Parameters
        ----------
        model : object
            Model to evaluate.
        data : list of str
            DNA sequences.
        gc_bins : list of float, optional
            Bin edges for GC fraction (default 0.0 to 1.0 in steps of 0.1).

        Returns
        -------
        BiasReport
        """
        if gc_bins is None:
            gc_bins = [i / 10.0 for i in range(11)]

        # Compute GC content for each sequence
        gc_fractions = []
        for seq in data:
            seq_upper = seq.upper()
            total = len(seq_upper)
            if total == 0:
                gc_fractions.append(0.0)
            else:
                gc = sum(1 for c in seq_upper if c in "GC")
                gc_fractions.append(gc / total)

        gc_fractions = np.array(gc_fractions)

        # Bin sequences
        results: Dict[str, Any] = {}
        for i in range(len(gc_bins) - 1):
            lo, hi = gc_bins[i], gc_bins[i + 1]
            bin_label = f"{lo:.1f}-{hi:.1f}"
            mask = (gc_fractions >= lo) & (gc_fractions < hi)
            if i == len(gc_bins) - 2:
                mask = (gc_fractions >= lo) & (gc_fractions <= hi)

            bin_seqs = [s for s, m in zip(data, mask) if m]
            if not bin_seqs:
                results[bin_label] = {"n_samples": 0, "mean_score": 0, "std_score": 0}
                continue

            scores = self._score_fn(model, bin_seqs)
            results[bin_label] = {
                "n_samples": len(bin_seqs),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "gc_range": (lo, hi),
            }

        report = BiasReport(
            gc_bias=results,
            summary=f"GC content bias audit across {len(results)} bins.",
        )
        self._reports["gc"] = report
        return report

    # -- Repeat region bias --------------------------------------------------

    def audit_repeat_bias(
        self,
        model: Any,
        repetitive_sequences: List[str],
        unique_sequences: List[str],
    ) -> BiasReport:
        """Compare performance in repetitive vs unique regions.

        Parameters
        ----------
        model : object
            Model to evaluate.
        repetitive_sequences : list of str
            Sequences from repetitive / low-complexity regions.
        unique_sequences : list of str
            Sequences from unique / non-repetitive regions.

        Returns
        -------
        BiasReport
        """
        results: Dict[str, Any] = {}

        for label, seqs in [("repetitive", repetitive_sequences),
                             ("unique", unique_sequences)]:
            if not seqs:
                results[label] = {"n_samples": 0, "mean_score": 0, "std_score": 0}
                continue
            scores = self._score_fn(model, seqs)
            results[label] = {
                "n_samples": len(seqs),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
            }

        report = BiasReport(
            repeat_bias=results,
            summary="Repeat region bias audit: repetitive vs unique regions.",
        )
        self._reports["repeat"] = report
        return report

    # -- Chromosome bias -----------------------------------------------------

    def audit_chromosome_bias(
        self,
        model: Any,
        data_by_chrom: Dict[str, List[str]],
    ) -> BiasReport:
        """Evaluate per-chromosome performance.

        Parameters
        ----------
        model : object
            Model to evaluate.
        data_by_chrom : dict
            Mapping from chromosome name to lists of DNA sequences.

        Returns
        -------
        BiasReport
        """
        results: Dict[str, Any] = {}

        for chrom, sequences in data_by_chrom.items():
            if not sequences:
                continue

            scores = self._score_fn(model, sequences)
            results[chrom] = {
                "n_samples": len(sequences),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
            }

        # Compute overall disparity
        means = [r["mean_score"] for r in results.values()]
        if len(means) > 1:
            max_gap = max(means) - min(means)
            best = max(results.items(), key=lambda x: x[1]["mean_score"])[0]
            worst = min(results.items(), key=lambda x: x[1]["mean_score"])[0]
        else:
            max_gap = 0
            best = worst = "N/A"

        report = BiasReport(
            chromosome_bias=results,
            summary=(
                f"Chromosome bias audit across {len(results)} chromosomes. "
                f"Best: {best}, Worst: {worst}, Max gap: {max_gap:.4f}."
            ),
        )
        self._reports["chromosome"] = report
        return report

    # -- Aggregate report ----------------------------------------------------

    def generate_report(self) -> BiasReport:
        """Combine all previously run audits into a single report.

        Returns
        -------
        BiasReport
            Aggregated report with all audit sections populated.
        """
        combined = BiasReport()
        summaries: List[str] = []

        for key, report in self._reports.items():
            if report.population_bias:
                combined.population_bias = report.population_bias
            if report.gc_bias:
                combined.gc_bias = report.gc_bias
            if report.chromosome_bias:
                combined.chromosome_bias = report.chromosome_bias
            if report.repeat_bias:
                combined.repeat_bias = report.repeat_bias
            if report.summary:
                summaries.append(report.summary)

        combined.summary = "\n\n".join(summaries) if summaries else "No audits have been run."
        return combined
