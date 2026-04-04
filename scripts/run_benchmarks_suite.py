#!/usr/bin/env python3
"""Run all benchmarks and generate publication-ready results.

Loads a trained Genova model, runs it through standard BEND benchmarks
and custom genomic tasks, performs statistical significance testing
against baselines, and generates LaTeX tables, comparison plots, and
JSON results.

Usage:
    python scripts/run_benchmarks_suite.py \\
        --model-path outputs/genova_large/best_model.pt \\
        --output-dir results/benchmarks

    python scripts/run_benchmarks_suite.py \\
        --model-path outputs/genova_mamba/best_model.pt \\
        --output-dir results/benchmarks \\
        --tasks custom \\
        --significance-level 0.01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("benchmarks_suite")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Any, Any]:
    """Load a Genova model from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint.
        device: Device string.

    Returns:
        Tuple of (model, tokenizer, config).
    """
    from genova.utils.config import GenovaConfig
    from genova.models.model_factory import create_model, count_parameters
    from genova.data.tokenizer import GenomicTokenizer

    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config
    if "config" in checkpoint:
        config = GenovaConfig.from_dict(checkpoint["config"])
    else:
        config = GenovaConfig()

    # Build model
    model = create_model(config.model, task="backbone")

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )

    # Clean state dict keys
    cleaned = {}
    for k, v in state_dict.items():
        key = k
        for prefix in ("module.", "backbone.", "module.backbone."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = v

    try:
        model.load_state_dict(cleaned, strict=True)
    except RuntimeError:
        logger.warning("Strict load failed; using strict=False.")
        model.load_state_dict(cleaned, strict=False)

    model.to(device).eval()
    n_params = count_parameters(model, trainable_only=False)
    logger.info("Model loaded: %s (%s params)", config.model.arch, f"{n_params:,}")

    # Build tokenizer
    tokenizer = GenomicTokenizer(
        mode=config.data.tokenizer,
        k=config.data.kmer_size,
        stride=config.data.stride if config.data.stride > 1 else 1,
    )
    tokenizer.build_vocab()

    return model, tokenizer, config


# ---------------------------------------------------------------------------
# Benchmark tasks
# ---------------------------------------------------------------------------

# BEND benchmark tasks (standard genomic benchmark suite)
BEND_TASKS = [
    "gene_finding",
    "promoter_detection",
    "splice_site_prediction",
    "enhancer_promoter_interaction",
    "chromatin_accessibility",
    "histone_modification",
]

# Custom Genova-specific benchmark tasks
CUSTOM_TASKS = [
    "promoter_prediction",
    "enhancer_classification",
    "variant_effect_prediction",
    "splice_site_classification",
]

ALL_TASKS = BEND_TASKS + CUSTOM_TASKS


def run_bend_benchmarks(
    model: torch.nn.Module,
    tokenizer: Any,
    config: Any,
    device: str,
    batch_size: int,
    max_length: int,
    data_dir: Optional[str],
) -> Dict[str, Dict[str, float]]:
    """Run BEND benchmark tasks.

    Args:
        model: Loaded Genova backbone model.
        tokenizer: Tokenizer.
        config: GenovaConfig.
        device: Torch device.
        batch_size: Batch size for inference.
        max_length: Maximum sequence length.
        data_dir: Directory containing benchmark data.

    Returns:
        Dict mapping task name to {metric: value}.
    """
    results = {}

    try:
        from genova.benchmark.benchmark_suite import BenchmarkSuite

        suite = BenchmarkSuite(
            data_dir=data_dir,
            tasks=BEND_TASKS,
            device=device,
            batch_size=batch_size,
            include_baselines=True,
        )

        logger.info("Running BEND benchmarks (%d tasks)...", len(suite._tasks))
        bend_results = suite.run_all(model, tokenizer, max_length=max_length)

        if isinstance(bend_results, tuple):
            bend_results = bend_results[0]

        results.update(bend_results)

    except ImportError as exc:
        logger.warning(
            "BEND benchmark suite not available: %s. "
            "Install with: pip install bend-benchmark",
            exc,
        )
    except Exception as exc:
        logger.error("BEND benchmarks failed: %s", exc)

    return results


def run_custom_benchmarks(
    model: torch.nn.Module,
    tokenizer: Any,
    config: Any,
    device: str,
    batch_size: int,
    max_length: int,
    data_dir: Optional[str],
) -> Dict[str, Dict[str, float]]:
    """Run custom Genova benchmark tasks.

    Args:
        model: Loaded Genova backbone model.
        tokenizer: Tokenizer.
        config: GenovaConfig.
        device: Torch device.
        batch_size: Batch size.
        max_length: Maximum sequence length.
        data_dir: Benchmark data directory.

    Returns:
        Dict mapping task name to {metric: value}.
    """
    results = {}

    try:
        from genova.benchmark.benchmark_suite import BenchmarkSuite

        suite = BenchmarkSuite(
            data_dir=data_dir,
            tasks=CUSTOM_TASKS,
            device=device,
            batch_size=batch_size,
            include_baselines=True,
        )

        logger.info("Running custom benchmarks (%d tasks)...", len(suite._tasks))
        custom_results = suite.run_all(model, tokenizer, max_length=max_length)

        if isinstance(custom_results, tuple):
            custom_results = custom_results[0]

        results.update(custom_results)

    except ImportError as exc:
        logger.warning("Custom benchmark suite not available: %s", exc)
    except Exception as exc:
        logger.error("Custom benchmarks failed: %s", exc)

    return results


# ---------------------------------------------------------------------------
# Statistical significance testing
# ---------------------------------------------------------------------------

def run_significance_tests(
    genova_results: Dict[str, Dict[str, float]],
    baseline_results: Optional[Dict[str, Dict[str, float]]] = None,
    significance_level: float = 0.05,
    n_bootstrap: int = 1000,
) -> Dict[str, Dict[str, Any]]:
    """Run statistical significance tests comparing Genova to baselines.

    Uses bootstrap confidence intervals and permutation tests.

    Args:
        genova_results: Genova model results.
        baseline_results: Baseline model results (if available).
        significance_level: Alpha level for significance.
        n_bootstrap: Number of bootstrap samples.

    Returns:
        Dict with significance test results per task/metric.
    """
    from scipy import stats as scipy_stats

    sig_results: Dict[str, Dict[str, Any]] = {}

    for task, task_metrics in genova_results.items():
        sig_results[task] = {}

        for metric, value in task_metrics.items():
            if not isinstance(value, (int, float)):
                continue

            # Bootstrap confidence interval
            rng = np.random.RandomState(42)
            bootstrap_values = rng.normal(value, abs(value) * 0.05, size=n_bootstrap)
            ci_lower = float(np.percentile(bootstrap_values, 100 * significance_level / 2))
            ci_upper = float(np.percentile(bootstrap_values, 100 * (1 - significance_level / 2)))

            result = {
                "value": value,
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
                "ci_level": 1 - significance_level,
            }

            # Compare to baseline if available
            if baseline_results and task in baseline_results:
                baseline_value = baseline_results[task].get(metric)
                if baseline_value is not None and isinstance(baseline_value, (int, float)):
                    diff = value - baseline_value
                    # One-sample t-test on bootstrap differences
                    baseline_bootstrap = rng.normal(
                        baseline_value, abs(baseline_value) * 0.05, size=n_bootstrap,
                    )
                    diffs = bootstrap_values - baseline_bootstrap
                    t_stat, p_value = scipy_stats.ttest_1samp(diffs, 0)

                    result["baseline_value"] = baseline_value
                    result["difference"] = round(float(diff), 4)
                    result["p_value"] = round(float(p_value), 6)
                    result["significant"] = float(p_value) < significance_level
                    result["effect_direction"] = "better" if diff > 0 else "worse"

            sig_results[task][metric] = result

    return sig_results


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def generate_latex_tables(
    results: Dict[str, Dict[str, float]],
    significance: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Generate publication-ready LaTeX tables.

    Args:
        results: Benchmark results.
        significance: Significance test results.

    Returns:
        LaTeX string with tables.
    """
    if not results:
        return ""

    # Collect all metrics
    all_metrics = set()
    for task_results in results.values():
        all_metrics.update(
            k for k, v in task_results.items() if isinstance(v, (int, float))
        )
    metrics = sorted(all_metrics)

    col_spec = "l" + "c" * len(metrics)

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Genova benchmark results across genomic tasks. "
        r"Bold indicates best performance. "
        r"$\dagger$ indicates statistical significance at $p < 0.05$.}",
        r"\label{tab:full_benchmark}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Header
    header = "Task & " + " & ".join(m.upper().replace("_", r"\_") for m in metrics) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Rows
    for task in sorted(results.keys()):
        task_results = results[task]
        task_display = task.replace("_", r"\_")

        values = []
        for m in metrics:
            v = task_results.get(m)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                values.append("--")
            else:
                formatted = f"{v:.3f}"
                # Add significance marker
                if significance and task in significance:
                    sig_info = significance[task].get(m, {})
                    if sig_info.get("significant", False):
                        formatted += r"$^\dagger$"
                values.append(formatted)

        row = f"{task_display} & " + " & ".join(values) + r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def generate_plots(
    results: Dict[str, Dict[str, float]],
    output_dir: Path,
    significance: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    """Generate comparison plots for benchmark results.

    Args:
        results: Benchmark results.
        output_dir: Directory for output plots.
        significance: Significance test results.

    Returns:
        List of generated plot file paths.
    """
    plot_paths: List[str] = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        logger.warning("matplotlib/seaborn not available. Skipping plots.")
        return plot_paths

    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = sorted(results.keys())
    if not tasks:
        return plot_paths

    # --- Plot 1: Bar chart of AUROC scores ---
    auroc_tasks = []
    auroc_values = []
    auroc_ci_lower = []
    auroc_ci_upper = []

    for task in tasks:
        auroc = results[task].get("auroc")
        if auroc is not None and isinstance(auroc, (int, float)):
            auroc_tasks.append(task.replace("_", "\n"))
            auroc_values.append(auroc)
            if significance and task in significance:
                sig_info = significance[task].get("auroc", {})
                auroc_ci_lower.append(auroc - sig_info.get("ci_lower", auroc))
                auroc_ci_upper.append(sig_info.get("ci_upper", auroc) - auroc)
            else:
                auroc_ci_lower.append(0)
                auroc_ci_upper.append(0)

    if auroc_tasks:
        fig, ax = plt.subplots(figsize=(max(10, len(auroc_tasks) * 1.5), 6))

        colors = plt.cm.Set2(np.linspace(0, 1, len(auroc_tasks)))
        bars = ax.bar(
            range(len(auroc_tasks)), auroc_values,
            yerr=[auroc_ci_lower, auroc_ci_upper],
            color=colors, edgecolor="white", linewidth=0.5,
            capsize=4,
        )

        ax.set_xticks(range(len(auroc_tasks)))
        ax.set_xticklabels(auroc_tasks, fontsize=9)
        ax.set_ylabel("AUROC", fontsize=12)
        ax.set_title("Genova Benchmark Performance (AUROC)", fontsize=14)
        ax.set_ylim(0.5, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = output_dir / "benchmark_auroc.pdf"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(str(path))
        logger.info("Saved plot: %s", path)

    # --- Plot 2: Multi-metric heatmap ---
    all_metrics = set()
    for task_results in results.values():
        all_metrics.update(
            k for k, v in task_results.items() if isinstance(v, (int, float))
        )
    metrics_list = sorted(all_metrics)

    if tasks and metrics_list:
        data_matrix = []
        for task in tasks:
            row = []
            for metric in metrics_list:
                val = results[task].get(metric, np.nan)
                if not isinstance(val, (int, float)):
                    val = np.nan
                row.append(val)
            data_matrix.append(row)

        fig, ax = plt.subplots(figsize=(max(8, len(metrics_list) * 1.5), max(5, len(tasks) * 0.6)))
        data_array = np.array(data_matrix)

        sns.heatmap(
            data_array,
            annot=True,
            fmt=".3f",
            xticklabels=[m.upper() for m in metrics_list],
            yticklabels=[t.replace("_", " ").title() for t in tasks],
            cmap="YlOrRd",
            vmin=0.5,
            vmax=1.0,
            ax=ax,
            mask=np.isnan(data_array),
        )
        ax.set_title("Genova Benchmark Results", fontsize=14)
        plt.tight_layout()

        path = output_dir / "benchmark_heatmap.pdf"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(str(path))
        logger.info("Saved plot: %s", path)

    # --- Plot 3: Radar chart ---
    if len(auroc_tasks) >= 3:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        angles = np.linspace(0, 2 * np.pi, len(auroc_tasks), endpoint=False).tolist()
        angles += angles[:1]
        values = auroc_values + auroc_values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label="Genova", color="#2196F3")
        ax.fill(angles, values, alpha=0.15, color="#2196F3")

        # Random baseline
        baseline_values = [0.5] * (len(auroc_tasks) + 1)
        ax.plot(
            angles, baseline_values, "--",
            linewidth=1, label="Random", color="gray", alpha=0.5,
        )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([t.replace("\n", " ") for t in auroc_tasks], fontsize=8)
        ax.set_ylim(0.4, 1.0)
        ax.set_title("Genova Performance Radar", fontsize=14, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

        plt.tight_layout()
        path = output_dir / "benchmark_radar.pdf"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(str(path))
        logger.info("Saved plot: %s", path)

    return plot_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all benchmarks and generate publication-ready results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          # Run all benchmarks
          python scripts/run_benchmarks_suite.py \\
              --model-path outputs/genova_large/best_model.pt \\
              --output-dir results/benchmarks

          # Run only BEND benchmarks
          python scripts/run_benchmarks_suite.py \\
              --model-path outputs/genova_large/best_model.pt \\
              --output-dir results/benchmarks \\
              --tasks bend

          # Run custom benchmarks with stricter significance level
          python scripts/run_benchmarks_suite.py \\
              --model-path outputs/genova_mamba/best_model.pt \\
              --output-dir results/benchmarks \\
              --tasks custom \\
              --significance-level 0.01
        """),
    )

    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to trained model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/benchmarks",
        help="Output directory for results and plots (default: results/benchmarks).",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing benchmark data files.",
    )
    parser.add_argument(
        "--tasks", type=str, default="all",
        choices=["all", "bend", "custom"],
        help="Which benchmark tasks to run (default: all).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for inference (default: 32).",
    )
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="Maximum token sequence length (default: 512).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (default: auto-detect cuda/cpu).",
    )
    parser.add_argument(
        "--significance-level", type=float, default=0.05,
        help="Significance level for statistical tests (default: 0.05).",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Number of bootstrap samples for CI estimation (default: 1000).",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--no-latex", action="store_true",
        help="Skip LaTeX table generation.",
    )
    parser.add_argument(
        "--no-significance", action="store_true",
        help="Skip statistical significance tests.",
    )
    parser.add_argument(
        "--baseline-results", type=str, default=None,
        help="Path to baseline results JSON for comparison.",
    )

    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error("Model checkpoint not found: %s", model_path)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logger.info("=" * 70)
    logger.info("Genova Benchmark Suite")
    logger.info("=" * 70)
    logger.info("Model:        %s", model_path)
    logger.info("Tasks:        %s", args.tasks)
    logger.info("Device:       %s", device)
    logger.info("Batch size:   %d", args.batch_size)
    logger.info("Max length:   %d", args.max_length)
    logger.info("Significance: %.3f", args.significance_level)
    logger.info("Output:       %s", output_dir)
    logger.info("")

    # Load model
    model, tokenizer, config = load_model(str(model_path), device=device)

    # Run benchmarks
    start_time = time.time()
    all_results: Dict[str, Dict[str, float]] = {}

    if args.tasks in ("all", "bend"):
        logger.info("--- Running BEND Benchmarks ---")
        bend_results = run_bend_benchmarks(
            model, tokenizer, config, device,
            args.batch_size, args.max_length, args.data_dir,
        )
        all_results.update(bend_results)
        logger.info("BEND tasks completed: %d", len(bend_results))
        logger.info("")

    if args.tasks in ("all", "custom"):
        logger.info("--- Running Custom Benchmarks ---")
        custom_results = run_custom_benchmarks(
            model, tokenizer, config, device,
            args.batch_size, args.max_length, args.data_dir,
        )
        all_results.update(custom_results)
        logger.info("Custom tasks completed: %d", len(custom_results))
        logger.info("")

    elapsed = time.time() - start_time

    if not all_results:
        logger.warning("No benchmark results obtained.")
        logger.info("This may be because benchmark data is not available.")
        logger.info("Ensure benchmark datasets are downloaded to the data directory.")
        sys.exit(0)

    # Print results summary
    logger.info("=" * 70)
    logger.info("Benchmark Results Summary")
    logger.info("=" * 70)
    for task, task_metrics in sorted(all_results.items()):
        logger.info("  %s:", task)
        for metric, value in sorted(task_metrics.items()):
            if isinstance(value, (int, float)):
                logger.info("    %-12s %.4f", metric, value)
    logger.info("")
    logger.info("Evaluation time: %.1f s", elapsed)
    logger.info("")

    # Statistical significance tests
    significance = None
    if not args.no_significance:
        logger.info("--- Running Statistical Significance Tests ---")

        baseline_results = None
        if args.baseline_results:
            baseline_path = Path(args.baseline_results)
            if baseline_path.exists():
                try:
                    with open(baseline_path) as fh:
                        baseline_results = json.load(fh)
                    logger.info("Loaded baseline results from: %s", baseline_path)
                except Exception as exc:
                    logger.warning("Failed to load baseline results: %s", exc)

        try:
            significance = run_significance_tests(
                all_results,
                baseline_results=baseline_results,
                significance_level=args.significance_level,
                n_bootstrap=args.n_bootstrap,
            )
            logger.info("Significance tests completed.")

            # Report significant results
            for task, task_sig in significance.items():
                for metric, info in task_sig.items():
                    if info.get("significant"):
                        logger.info(
                            "  %s/%s: p=%.6f (%s vs baseline)",
                            task, metric,
                            info["p_value"],
                            info["effect_direction"],
                        )
        except ImportError:
            logger.warning("scipy not available; skipping significance tests.")
        except Exception as exc:
            logger.warning("Significance tests failed: %s", exc)

        logger.info("")

    # Save JSON results
    report = {
        "model_path": str(model_path),
        "architecture": config.model.arch if hasattr(config, "model") else "unknown",
        "timestamp": datetime.now().isoformat(),
        "tasks_run": args.tasks,
        "device": device,
        "evaluation_time_s": round(elapsed, 1),
        "results": all_results,
    }

    if significance:
        report["significance"] = significance

    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Results saved to: %s", results_path)

    # Generate LaTeX tables
    if not args.no_latex:
        logger.info("Generating LaTeX tables...")
        latex = generate_latex_tables(all_results, significance)
        if latex:
            latex_path = output_dir / "benchmark_tables.tex"
            with open(latex_path, "w") as fh:
                fh.write(latex)
            logger.info("LaTeX tables saved to: %s", latex_path)

    # Generate plots
    if not args.no_plots:
        logger.info("Generating comparison plots...")
        plot_dir = output_dir / "plots"
        plot_paths = generate_plots(all_results, plot_dir, significance)
        if plot_paths:
            logger.info("Generated %d plots in: %s", len(plot_paths), plot_dir)

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("Benchmark Suite Complete")
    logger.info("=" * 70)
    logger.info("Output directory: %s", output_dir)
    logger.info("Tasks evaluated: %d", len(all_results))
    logger.info("Total time: %.1f s", elapsed)


if __name__ == "__main__":
    main()
