#!/usr/bin/env python3
"""Evaluate Genova model on standard genomic benchmarks.

Loads a trained Genova model checkpoint, runs it through the benchmark
suite (promoter detection, splice site prediction, enhancer classification,
variant effect prediction), and generates comparison plots, JSON reports,
and LaTeX tables.

Usage:
    python scripts/evaluate_model.py \\
        --model-path outputs/genova_pretrain/best_model.pt \\
        --output-dir results/evaluation

    python scripts/evaluate_model.py \\
        --model-path outputs/genova_pretrain/best_model.pt \\
        --tasks promoter_prediction splice_site_prediction \\
        --data-dir data/benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
logger = logging.getLogger("evaluate_genova")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple:
    """Load a Genova model and tokenizer from a training checkpoint.

    The checkpoint is expected to contain 'model_state_dict' and 'config'.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to load the model onto.

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
        logger.warning(
            "No config found in checkpoint. Using default config."
        )
        config = GenovaConfig()

    # Build model and load weights
    model = create_model(config.model, task="backbone")

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

    # Handle MLM-wrapped models: strip 'backbone.' prefix if present
    cleaned_state = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith("backbone."):
            new_key = k[len("backbone."):]
        if k.startswith("module."):
            new_key = k[len("module."):]
        if new_key.startswith("backbone."):
            new_key = new_key[len("backbone."):]
        cleaned_state[new_key] = v

    try:
        model.load_state_dict(cleaned_state, strict=True)
    except RuntimeError:
        logger.warning("Strict loading failed; trying non-strict.")
        model.load_state_dict(cleaned_state, strict=False)

    model.to(device).eval()

    n_params = count_parameters(model, trainable_only=False)
    logger.info("Model loaded: %s (%s parameters)", config.model.arch, f"{n_params:,}")

    # Build tokenizer
    tokenizer = GenomicTokenizer(
        mode=config.data.tokenizer,
        k=config.data.kmer_size,
        stride=config.data.stride if config.data.stride > 1 else 1,
    )
    tokenizer.build_vocab()
    logger.info("Tokenizer: mode=%s, vocab_size=%d", tokenizer.mode, tokenizer.vocab_size)

    return model, tokenizer, config


# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------

def run_benchmarks(
    model: torch.nn.Module,
    tokenizer: Any,
    config: Any,
    tasks: Optional[List[str]],
    data_dir: Optional[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> Dict[str, Any]:
    """Run benchmark tasks and return results.

    Args:
        model: Loaded Genova model (backbone).
        tokenizer: Tokenizer with built vocabulary.
        config: GenovaConfig.
        tasks: List of task names to run, or None for all.
        data_dir: Directory containing benchmark data.
        device: Torch device.
        batch_size: Inference batch size.
        max_length: Maximum sequence length.

    Returns:
        Nested dict of results.
    """
    from genova.benchmark.benchmark_suite import BenchmarkSuite

    suite = BenchmarkSuite(
        data_dir=data_dir,
        tasks=tasks,
        device=device,
        batch_size=batch_size,
        include_baselines=True,
    )

    logger.info("Running %d benchmark tasks...", len(suite._tasks))
    logger.info("")

    results = suite.run_all(model, tokenizer, max_length=max_length)

    return results, suite


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_latex_table(results: Dict[str, Dict[str, float]]) -> str:
    """Generate a LaTeX table from benchmark results.

    Args:
        results: Nested dict {task: {metric: value}}.

    Returns:
        LaTeX table string.
    """
    if not results:
        return ""

    # Collect all metrics across tasks
    all_metrics = set()
    for task_results in results.values():
        all_metrics.update(task_results.keys())
    metrics = sorted(all_metrics)

    # Build table
    n_cols = len(metrics) + 1
    col_spec = "l" + "c" * len(metrics)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Genova benchmark results across genomic tasks.}",
        r"\label{tab:benchmark}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Header
    header = "Task & " + " & ".join(m.upper() for m in metrics) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Rows
    for task, task_results in sorted(results.items()):
        task_display = task.replace("_", " ").title()
        values = []
        for m in metrics:
            v = task_results.get(m, float("nan"))
            if np.isnan(v):
                values.append("--")
            else:
                values.append(f"{v:.3f}")
        row = f"{task_display} & " + " & ".join(values) + r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_comparison_plots(
    suite: Any,
    output_dir: Path,
) -> List[str]:
    """Generate comparison plots against baseline models.

    Args:
        suite: BenchmarkSuite with results.
        output_dir: Directory to save plots.

    Returns:
        List of generated plot file paths.
    """
    plot_paths = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available. Skipping plots.")
        return plot_paths

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get comparison data
    try:
        comparison = suite.compare_models()
    except RuntimeError:
        logger.warning("No comparison data available.")
        return plot_paths

    if not comparison:
        return plot_paths

    # --- Plot 1: Grouped bar chart of AUROC across tasks and models ---
    fig, ax = plt.subplots(figsize=(12, 6))

    tasks = sorted(comparison.keys())
    models = set()
    for task_data in comparison.values():
        models.update(task_data.keys())
    models = sorted(models)

    x = np.arange(len(tasks))
    width = 0.8 / max(len(models), 1)

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, model_name in enumerate(models):
        auroc_values = []
        for task in tasks:
            task_models = comparison.get(task, {})
            model_metrics = task_models.get(model_name, {})
            auroc_values.append(model_metrics.get("auroc", 0))

        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, auroc_values, width,
            label=model_name,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Genova vs. Baseline Models: AUROC by Task", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [t.replace("_", "\n") for t in tasks],
        fontsize=9,
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0.6, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "benchmark_comparison_auroc.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(str(path))
    logger.info("Saved plot: %s", path)

    # --- Plot 2: Radar/spider chart ---
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for i, model_name in enumerate(models):
        values = []
        for task in tasks:
            task_models = comparison.get(task, {})
            model_metrics = task_models.get(model_name, {})
            values.append(model_metrics.get("auroc", 0))
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=8)
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Model Comparison (AUROC)", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    path = output_dir / "benchmark_radar.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(str(path))
    logger.info("Saved plot: %s", path)

    # --- Plot 3: Heatmap of all metrics ---
    all_metrics_set = set()
    for task_data in comparison.values():
        for model_data in task_data.values():
            all_metrics_set.update(model_data.keys())
    all_metrics = sorted(all_metrics_set)

    # Build matrix for Genova only
    genova_data = []
    for task in tasks:
        row = []
        for metric in all_metrics:
            val = comparison.get(task, {}).get("Genova", {}).get(metric, np.nan)
            row.append(val)
        genova_data.append(row)

    if genova_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        data_array = np.array(genova_data)

        sns.heatmap(
            data_array,
            annot=True,
            fmt=".3f",
            xticklabels=[m.upper() for m in all_metrics],
            yticklabels=[t.replace("_", " ").title() for t in tasks],
            cmap="YlOrRd",
            vmin=0.5,
            vmax=1.0,
            ax=ax,
        )
        ax.set_title("Genova Benchmark Metrics", fontsize=14)
        plt.tight_layout()

        path = output_dir / "benchmark_heatmap.pdf"
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
        description="Evaluate a trained Genova model on standard benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python scripts/evaluate_model.py \\
      --model-path outputs/genova_pretrain/best_model.pt

  # Run specific tasks
  python scripts/evaluate_model.py \\
      --model-path outputs/genova_pretrain/best_model.pt \\
      --tasks promoter_prediction variant_effect_prediction

  # Custom output directory and batch size
  python scripts/evaluate_model.py \\
      --model-path outputs/genova_pretrain/best_model.pt \\
      --output-dir results/my_eval \\
      --batch-size 128
        """,
    )

    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to trained model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing benchmark data files.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/evaluation",
        help="Output directory for results, plots, and tables.",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="*", default=None,
        help=(
            "Benchmark tasks to run. Default: all available. "
            "Options: promoter_prediction, enhancer_classification, "
            "variant_effect_prediction, splice_site_prediction"
        ),
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
        "--no-plots", action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--no-latex", action="store_true",
        help="Skip LaTeX table generation.",
    )

    args = parser.parse_args()

    # Validate inputs
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
    logger.info("Genova Model Evaluation")
    logger.info("=" * 70)
    logger.info("Model:      %s", model_path)
    logger.info("Device:     %s", device)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Output:     %s", output_dir)
    logger.info("")

    # --- Load model ---
    model, tokenizer, config = load_model_from_checkpoint(
        str(model_path), device=device,
    )

    # --- Run benchmarks ---
    start_time = time.time()

    results, suite = run_benchmarks(
        model=model,
        tokenizer=tokenizer,
        config=config,
        tasks=args.tasks,
        data_dir=args.data_dir,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    elapsed = time.time() - start_time

    # --- Print results ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("Benchmark Results")
    logger.info("=" * 70)
    for task_name, task_metrics in sorted(results.items()):
        logger.info("  %s:", task_name)
        for metric, value in sorted(task_metrics.items()):
            logger.info("    %-10s %.4f", metric, value)
    logger.info("")
    logger.info("Total evaluation time: %.1f s", elapsed)

    # --- Save JSON report ---
    report = suite.generate_report(output_path=output_dir / "benchmark_report.json")
    logger.info("JSON report saved: %s", output_dir / "benchmark_report.json")

    # --- Generate LaTeX table ---
    if not args.no_latex:
        latex_table = generate_latex_table(results)
        if latex_table:
            latex_path = output_dir / "benchmark_table.tex"
            with open(latex_path, "w") as fh:
                fh.write(latex_table)
            logger.info("LaTeX table saved: %s", latex_path)

    # --- Generate comparison plots ---
    if not args.no_plots:
        plot_dir = output_dir / "plots"
        plot_paths = generate_comparison_plots(suite, plot_dir)
        if plot_paths:
            logger.info("Generated %d comparison plots.", len(plot_paths))

    # --- Summary ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("Evaluation Complete")
    logger.info("=" * 70)
    logger.info("Output directory: %s", output_dir)
    logger.info(
        "Files: benchmark_report.json, benchmark_table.tex, plots/"
    )


if __name__ == "__main__":
    main()
