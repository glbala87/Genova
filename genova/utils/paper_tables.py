"""Publication-quality table generation for the Genova methods paper.

Produces LaTeX and Markdown tables for model comparisons, ablation
studies, and computational requirements.

Example::

    from genova.utils.paper_tables import generate_comparison_table

    latex = generate_comparison_table(results, fmt="latex")
    print(latex)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Table 1: Model comparison
# ---------------------------------------------------------------------------


def generate_comparison_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
    fmt: str = "latex",
    metrics: Optional[List[str]] = None,
    highlight_best: bool = True,
    caption: str = "Performance comparison across genomic benchmark tasks.",
    label: str = "tab:comparison",
) -> str:
    """Generate a model comparison table.

    Args:
        results: Nested dict ``{task: {model: {metric: value}}}``.
        fmt: Output format: ``"latex"`` or ``"markdown"``.
        metrics: Subset of metrics.  ``None`` means all found.
        highlight_best: Bold the best value per column.
        caption: LaTeX caption text.
        label: LaTeX label.

    Returns:
        Formatted table string.
    """
    tasks = sorted(results.keys())
    models = sorted(
        {m for task_data in results.values() for m in task_data.keys()}
    )

    # Gather all metrics
    all_metrics: set[str] = set()
    for task_data in results.values():
        for model_data in task_data.values():
            all_metrics.update(model_data.keys())
    metric_list = sorted(metrics or all_metrics)

    # Higher-is-better metrics
    lower_is_better = {"ece", "brier_score", "mse", "mae"}

    if fmt == "latex":
        return _comparison_latex(
            results, tasks, models, metric_list,
            highlight_best, lower_is_better, caption, label,
        )
    elif fmt == "markdown":
        return _comparison_markdown(
            results, tasks, models, metric_list,
            highlight_best, lower_is_better,
        )
    else:
        raise ValueError(f"Unsupported format {fmt!r}. Use 'latex' or 'markdown'.")


def _comparison_latex(
    results: Dict[str, Dict[str, Dict[str, float]]],
    tasks: List[str],
    models: List[str],
    metrics: List[str],
    highlight_best: bool,
    lower_is_better: set,
    caption: str,
    label: str,
) -> str:
    """Build LaTeX comparison table."""
    n_model_cols = len(models)
    col_spec = "ll" + "c" * n_model_cols

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Header
    header = "Task & Metric & " + " & ".join(
        rf"\textbf{{{m}}}" for m in models
    ) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for ti, task in enumerate(tasks):
        task_display = task.replace("_", " ").title()
        n_met = len(metrics)

        for mi, met in enumerate(metrics):
            values = []
            for model in models:
                v = results.get(task, {}).get(model, {}).get(met, float("nan"))
                values.append(v)

            # Find best
            valid_vals = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
            best_idx: Optional[int] = None
            if highlight_best and valid_vals:
                if met in lower_is_better:
                    best_idx = min(valid_vals, key=lambda x: x[1])[0]
                else:
                    best_idx = max(valid_vals, key=lambda x: x[1])[0]

            # Build row
            task_cell = rf"\multirow{{{n_met}}}{{*}}{{{task_display}}}" if mi == 0 else ""
            cells = [task_cell, met.upper()]
            for i, v in enumerate(values):
                if np.isnan(v):
                    cells.append("--")
                else:
                    cell = f"{v:.4f}"
                    if i == best_idx:
                        cell = rf"\textbf{{{cell}}}"
                    cells.append(cell)
            lines.append(" & ".join(cells) + r" \\")

        if ti < len(tasks) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    return "\n".join(lines)


def _comparison_markdown(
    results: Dict[str, Dict[str, Dict[str, float]]],
    tasks: List[str],
    models: List[str],
    metrics: List[str],
    highlight_best: bool,
    lower_is_better: set,
) -> str:
    """Build Markdown comparison table."""
    lines = []
    header = "| Task | Metric | " + " | ".join(models) + " |"
    separator = "|" + "|".join(["---"] * (2 + len(models))) + "|"
    lines.append(header)
    lines.append(separator)

    for task in tasks:
        task_display = task.replace("_", " ").title()
        for mi, met in enumerate(metrics):
            values = []
            for model in models:
                v = results.get(task, {}).get(model, {}).get(met, float("nan"))
                values.append(v)

            valid_vals = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
            best_idx: Optional[int] = None
            if highlight_best and valid_vals:
                if met in lower_is_better:
                    best_idx = min(valid_vals, key=lambda x: x[1])[0]
                else:
                    best_idx = max(valid_vals, key=lambda x: x[1])[0]

            task_cell = task_display if mi == 0 else ""
            cells = [task_cell, met.upper()]
            for i, v in enumerate(values):
                if np.isnan(v):
                    cells.append("--")
                else:
                    cell = f"{v:.4f}"
                    if i == best_idx:
                        cell = f"**{cell}**"
                    cells.append(cell)
            lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2: Ablation study
# ---------------------------------------------------------------------------


def generate_ablation_table(
    ablation_results: Dict[str, Dict[str, float]],
    fmt: str = "latex",
    metrics: Optional[List[str]] = None,
    caption: str = "Ablation study results.",
    label: str = "tab:ablation",
) -> str:
    """Generate an ablation study table.

    Args:
        ablation_results: Dict ``{variant_name: {metric: value}}``.
            Example variants: ``"Full model"``, ``"w/o Mamba"``,
            ``"w/o contrastive loss"``.
        fmt: ``"latex"`` or ``"markdown"``.
        metrics: Subset of metrics.
        caption: LaTeX caption.
        label: LaTeX label.

    Returns:
        Formatted table string.
    """
    variants = list(ablation_results.keys())

    all_metrics: set[str] = set()
    for met_dict in ablation_results.values():
        all_metrics.update(met_dict.keys())
    metric_list = sorted(metrics or all_metrics)

    if fmt == "latex":
        return _ablation_latex(ablation_results, variants, metric_list, caption, label)
    elif fmt == "markdown":
        return _ablation_markdown(ablation_results, variants, metric_list)
    else:
        raise ValueError(f"Unsupported format {fmt!r}.")


def _ablation_latex(
    results: Dict[str, Dict[str, float]],
    variants: List[str],
    metrics: List[str],
    caption: str,
    label: str,
) -> str:
    """Build LaTeX ablation table."""
    col_spec = "l" + "c" * len(metrics)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header = r"\textbf{Configuration} & " + " & ".join(
        rf"\textbf{{{m.upper()}}}" for m in metrics
    ) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Find full-model values for delta computation
    full_model_key = None
    for v in variants:
        if "full" in v.lower():
            full_model_key = v
            break

    for variant in variants:
        cells = [variant]
        for met in metrics:
            val = results[variant].get(met, float("nan"))
            if np.isnan(val):
                cells.append("--")
            else:
                cell = f"{val:.4f}"
                # Add delta annotation for non-full variants
                if full_model_key and variant != full_model_key:
                    full_val = results[full_model_key].get(met, float("nan"))
                    if not np.isnan(full_val):
                        delta = val - full_val
                        sign = "+" if delta >= 0 else ""
                        cell += rf" \textcolor{{gray}}{{\scriptsize ({sign}{delta:.3f})}}"
                if variant == full_model_key:
                    cell = rf"\textbf{{{cell}}}"
                cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def _ablation_markdown(
    results: Dict[str, Dict[str, float]],
    variants: List[str],
    metrics: List[str],
) -> str:
    """Build Markdown ablation table."""
    lines = []
    header = "| Configuration | " + " | ".join(m.upper() for m in metrics) + " |"
    separator = "|" + "|".join(["---"] * (1 + len(metrics))) + "|"
    lines.append(header)
    lines.append(separator)

    full_model_key = None
    for v in variants:
        if "full" in v.lower():
            full_model_key = v
            break

    for variant in variants:
        cells = [variant]
        for met in metrics:
            val = results[variant].get(met, float("nan"))
            if np.isnan(val):
                cells.append("--")
            else:
                cell = f"{val:.4f}"
                if full_model_key and variant != full_model_key:
                    full_val = results[full_model_key].get(met, float("nan"))
                    if not np.isnan(full_val):
                        delta = val - full_val
                        sign = "+" if delta >= 0 else ""
                        cell += f" ({sign}{delta:.3f})"
                if variant == full_model_key:
                    cell = f"**{cell}**"
                cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 3: Computational requirements
# ---------------------------------------------------------------------------


def generate_compute_table(
    compute_data: Dict[str, Dict[str, Any]],
    fmt: str = "latex",
    caption: str = "Computational requirements for model training and inference.",
    label: str = "tab:compute",
) -> str:
    """Generate a computational requirements table.

    Args:
        compute_data: Dict ``{model_name: {field: value}}``.
            Expected fields: ``parameters``, ``training_time``,
            ``gpu_memory``, ``inference_time``, ``flops``.
        fmt: ``"latex"`` or ``"markdown"``.
        caption: LaTeX caption.
        label: LaTeX label.

    Returns:
        Formatted table string.
    """
    models = list(compute_data.keys())
    fields = [
        ("parameters", "Parameters"),
        ("training_time", "Training Time"),
        ("gpu_memory", "GPU Memory (GB)"),
        ("inference_time", "Inference (ms/seq)"),
        ("flops", "FLOPs"),
    ]

    # Filter to available fields
    available_fields = []
    for key, display in fields:
        if any(key in compute_data[m] for m in models):
            available_fields.append((key, display))

    if fmt == "latex":
        return _compute_latex(compute_data, models, available_fields, caption, label)
    elif fmt == "markdown":
        return _compute_markdown(compute_data, models, available_fields)
    else:
        raise ValueError(f"Unsupported format {fmt!r}.")


def _compute_latex(
    data: Dict[str, Dict[str, Any]],
    models: List[str],
    fields: List[Tuple[str, str]],
    caption: str,
    label: str,
) -> str:
    """Build LaTeX compute table."""
    col_spec = "l" + "r" * len(fields)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header = r"\textbf{Model} & " + " & ".join(
        rf"\textbf{{{display}}}" for _, display in fields
    ) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for model in models:
        cells = [rf"\textbf{{{model}}}" if model == "Genova" else model]
        for key, _ in fields:
            val = data[model].get(key, "--")
            if isinstance(val, float):
                cells.append(f"{val:,.2f}")
            elif isinstance(val, int):
                cells.append(f"{val:,}")
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def _compute_markdown(
    data: Dict[str, Dict[str, Any]],
    models: List[str],
    fields: List[Tuple[str, str]],
) -> str:
    """Build Markdown compute table."""
    lines = []
    header = "| Model | " + " | ".join(display for _, display in fields) + " |"
    separator = "|" + "|".join(["---"] * (1 + len(fields))) + "|"
    lines.append(header)
    lines.append(separator)

    for model in models:
        cells = [f"**{model}**" if model == "Genova" else model]
        for key, _ in fields:
            val = data[model].get(key, "--")
            if isinstance(val, float):
                cells.append(f"{val:,.2f}")
            elif isinstance(val, int):
                cells.append(f"{val:,}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Export utility
# ---------------------------------------------------------------------------


def export_latex(
    tables: Dict[str, str],
    output_path: Union[str, Path],
    preamble: bool = True,
) -> None:
    """Export multiple LaTeX tables to a single ``.tex`` file.

    Args:
        tables: Mapping of table name to LaTeX string.
        output_path: Output file path.
        preamble: If ``True``, include minimal LaTeX preamble for
            standalone compilation.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    parts: List[str] = []

    if preamble:
        parts.extend([
            r"\documentclass{article}",
            r"\usepackage{booktabs}",
            r"\usepackage{multirow}",
            r"\usepackage{xcolor}",
            r"\usepackage{graphicx}",
            r"\begin{document}",
            "",
        ])

    for name, table in tables.items():
        parts.append(f"% --- {name} ---")
        parts.append(table)
        parts.append("")

    if preamble:
        parts.append(r"\end{document}")

    with open(path, "w") as fh:
        fh.write("\n".join(parts))

    logger.info("LaTeX tables exported to {}", path)
