"""Model comparison utilities for benchmark analysis.

Provides statistical significance testing, comparison tables, and
visualisation for comparing Genova against baseline genomic models
(DNABERT, Enformer, Basenji, Evo, etc.).

Example::

    from genova.benchmark.comparison import ModelComparator

    comparator = ModelComparator()
    comparator.load_results("results/genova.json")
    comparator.load_results("results/dnabert.json")
    comparator.compare()
    comparator.plot_comparison("comparison.pdf")
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from loguru import logger

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    """Container for a single model's benchmark results.

    Attributes:
        model_name: Human-readable model name.
        task_metrics: Mapping of task name to metric dict.
        predictions: Optional raw predictions keyed by task name.
        labels: Optional ground truth labels keyed by task name.
    """

    model_name: str
    task_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    labels: Dict[str, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def _paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    metric_fn: Any,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Two-sided paired bootstrap test for comparing two models.

    Args:
        scores_a: Predicted scores from model A, shape ``(N,)``.
        scores_b: Predicted scores from model B, shape ``(N,)``.
        labels: Ground truth binary labels, shape ``(N,)``.
        metric_fn: Callable ``(scores, labels) -> float``.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        Tuple of ``(observed_diff, p_value, ci_95)`` where *observed_diff*
        is ``metric(A) - metric(B)``, *p_value* is the two-sided p-value,
        and *ci_95* is the half-width of the 95% confidence interval.
    """
    rng = np.random.RandomState(seed)
    n = len(labels)

    observed_a = metric_fn(scores_a, labels)
    observed_b = metric_fn(scores_b, labels)
    observed_diff = observed_a - observed_b

    count = 0
    diffs = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diff_i = metric_fn(scores_a[idx], labels[idx]) - metric_fn(
            scores_b[idx], labels[idx]
        )
        diffs[i] = diff_i
        if abs(diff_i) >= abs(observed_diff):
            count += 1

    p_value = (count + 1) / (n_bootstrap + 1)
    ci_95 = float(np.percentile(np.abs(diffs - observed_diff), 95))

    return float(observed_diff), float(p_value), ci_95


def _delong_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """DeLong test for comparing two AUROCs on the same dataset.

    Implements the fast O(N log N) version of the DeLong test.

    Args:
        scores_a: Predicted scores from model A.
        scores_b: Predicted scores from model B.
        labels: Binary ground truth labels.

    Returns:
        Tuple of ``(z_statistic, p_value)``.
    """
    labels = np.asarray(labels).ravel()
    scores_a = np.asarray(scores_a, dtype=np.float64).ravel()
    scores_b = np.asarray(scores_b, dtype=np.float64).ravel()

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)

    if m == 0 or n == 0:
        logger.warning("DeLong test requires both classes; returning NaN.")
        return float("nan"), float("nan")

    def _compute_placement_values(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute structural components for DeLong variance estimation."""
        v_pos = np.zeros(m, dtype=np.float64)
        v_neg = np.zeros(n, dtype=np.float64)

        for i, pi in enumerate(pos_idx):
            for j, nj in enumerate(neg_idx):
                if scores[pi] > scores[nj]:
                    v_pos[i] += 1.0
                    v_neg[j] += 1.0
                elif scores[pi] == scores[nj]:
                    v_pos[i] += 0.5
                    v_neg[j] += 0.5

        v_pos /= n
        v_neg /= m
        return v_pos, v_neg

    v10_a, v01_a = _compute_placement_values(scores_a)
    v10_b, v01_b = _compute_placement_values(scores_b)

    auc_a = v10_a.mean()
    auc_b = v10_b.mean()

    # Covariance matrix of (AUC_a, AUC_b)
    s10 = np.cov(np.stack([v10_a, v10_b])) / m if m > 1 else np.zeros((2, 2))
    s01 = np.cov(np.stack([v01_a, v01_b])) / n if n > 1 else np.zeros((2, 2))
    s = s10 + s01

    # Variance of difference
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var_diff <= 0:
        return 0.0, 1.0

    z = (auc_a - auc_b) / np.sqrt(var_diff)

    # Two-sided p-value from standard normal (without scipy)
    p_value = 2 * _normal_cdf(-abs(z))

    return float(z), float(p_value)


def _normal_cdf(x: float) -> float:
    """Approximate the standard normal CDF using the error function.

    Args:
        x: Input value.

    Returns:
        CDF value.
    """
    import math

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# ModelComparator
# ---------------------------------------------------------------------------


class ModelComparator:
    """Compare genomic models across benchmark tasks.

    Supports loading results from JSON or CSV, running statistical
    significance tests, and generating comparison plots and LaTeX tables.

    Example::

        comp = ModelComparator()
        comp.add_result("Genova", {"promoter": {"auroc": 0.95, "f1": 0.90}})
        comp.add_result("DNABERT", {"promoter": {"auroc": 0.88, "f1": 0.82}})
        table = comp.compare()
    """

    def __init__(self) -> None:
        self._results: Dict[str, ModelResult] = {}

    # -- data loading --------------------------------------------------------

    def add_result(
        self,
        model_name: str,
        task_metrics: Dict[str, Dict[str, float]],
        predictions: Optional[Dict[str, np.ndarray]] = None,
        labels: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Register benchmark results for a model.

        Args:
            model_name: Model identifier.
            task_metrics: ``{task_name: {metric_name: value}}``.
            predictions: Optional raw prediction arrays keyed by task.
            labels: Optional ground truth arrays keyed by task.
        """
        self._results[model_name] = ModelResult(
            model_name=model_name,
            task_metrics=task_metrics,
            predictions=predictions or {},
            labels=labels or {},
        )
        logger.info(
            "Registered results for {} ({} tasks)", model_name, len(task_metrics)
        )

    def load_results(
        self,
        path: Union[str, Path],
        model_name: Optional[str] = None,
    ) -> None:
        """Load model results from a JSON or CSV file.

        **JSON format**::

            {
              "model_name": "Genova",
              "tasks": {
                "promoter_prediction": {"auroc": 0.95, "auprc": 0.92, ...},
                ...
              }
            }

        **CSV format**: columns ``model_name,task,metric,value``.

        Args:
            path: Path to the results file.
            model_name: Override the model name from the file.
        """
        path = Path(path)

        if path.suffix == ".json":
            with open(path) as fh:
                data = json.load(fh)
            name = model_name or data.get("model_name", path.stem)
            tasks = data.get("tasks", data.get("task_metrics", {}))
            self.add_result(name, tasks)

        elif path.suffix == ".csv":
            task_metrics: Dict[str, Dict[str, float]] = {}
            inferred_name: Optional[str] = None
            with open(path) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    inferred_name = inferred_name or row.get("model_name", path.stem)
                    task = row["task"]
                    metric = row["metric"]
                    value = float(row["value"])
                    task_metrics.setdefault(task, {})[metric] = value
            name = model_name or inferred_name or path.stem
            self.add_result(name, task_metrics)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        logger.info("Loaded results from {}", path)

    # -- comparison ----------------------------------------------------------

    def compare(
        self,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Generate a comparison table across all registered models.

        Args:
            metrics: Subset of metrics to include.  ``None`` means all.

        Returns:
            Nested dict ``{task: {model: {metric: value}}}``.
        """
        if not self._results:
            raise RuntimeError("No results registered. Call add_result() first.")

        # Collect all tasks and metrics
        all_tasks: set[str] = set()
        all_metrics: set[str] = set()
        for res in self._results.values():
            for task, mets in res.task_metrics.items():
                all_tasks.add(task)
                all_metrics.update(mets.keys())

        if metrics is not None:
            all_metrics = all_metrics & set(metrics)

        comparison: Dict[str, Dict[str, Dict[str, float]]] = {}
        for task in sorted(all_tasks):
            comparison[task] = {}
            for model_name, res in self._results.items():
                task_mets = res.task_metrics.get(task, {})
                comparison[task][model_name] = {
                    m: task_mets.get(m, float("nan")) for m in sorted(all_metrics)
                }

        logger.info(
            "Comparison table: {} tasks x {} models x {} metrics",
            len(all_tasks),
            len(self._results),
            len(all_metrics),
        )
        return comparison

    def statistical_test(
        self,
        model_a: str,
        model_b: str,
        task: str,
        test: str = "bootstrap",
        metric_fn: Optional[Any] = None,
        n_bootstrap: int = 10_000,
    ) -> Dict[str, float]:
        """Run a statistical significance test between two models on a task.

        Requires that raw predictions and labels were provided via
        :meth:`add_result`.

        Args:
            model_a: Name of model A.
            model_b: Name of model B.
            task: Task name.
            test: ``"bootstrap"`` for paired bootstrap or ``"delong"`` for
                DeLong AUROC test.
            metric_fn: Metric function ``(scores, labels) -> float``.
                Defaults to AUROC for ``"delong"`` and ``"bootstrap"``.
            n_bootstrap: Number of bootstrap iterations (bootstrap only).

        Returns:
            Dictionary with test statistics (varies by test type).

        Raises:
            ValueError: If predictions/labels are missing for the task.
        """
        res_a = self._results[model_a]
        res_b = self._results[model_b]

        if task not in res_a.predictions or task not in res_b.predictions:
            raise ValueError(
                f"Raw predictions required for statistical test on {task!r}. "
                "Provide predictions via add_result()."
            )

        scores_a = res_a.predictions[task]
        scores_b = res_b.predictions[task]
        labels = res_a.labels.get(task) or res_b.labels.get(task)
        if labels is None:
            raise ValueError(f"Ground truth labels required for task {task!r}.")

        if test == "delong":
            z, p = _delong_test(scores_a, scores_b, labels)
            result = {
                "test": "delong",
                "z_statistic": z,
                "p_value": p,
                "significant_0.05": float(p < 0.05),
            }
        elif test == "bootstrap":
            from genova.evaluation.metrics import auroc as _auroc_fn

            fn = metric_fn or _auroc_fn
            diff, p, ci = _paired_bootstrap_test(
                scores_a, scores_b, labels, fn, n_bootstrap=n_bootstrap
            )
            result = {
                "test": "paired_bootstrap",
                "observed_diff": diff,
                "p_value": p,
                "ci_95": ci,
                "significant_0.05": float(p < 0.05),
            }
        else:
            raise ValueError(f"Unknown test {test!r}. Use 'bootstrap' or 'delong'.")

        logger.info(
            "Statistical test ({}) {} vs {} on {}: p={:.4f}",
            test,
            model_a,
            model_b,
            task,
            result["p_value"],
        )
        return result

    # -- visualisation -------------------------------------------------------

    def plot_comparison(
        self,
        output_path: Optional[Union[str, Path]] = None,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (12, 6),
        dpi: int = 300,
    ) -> Optional[Any]:
        """Generate a grouped bar chart comparing models across tasks.

        Args:
            output_path: Path to save the figure (PDF/PNG/SVG).
            metrics: Subset of metrics to plot.
            figsize: Figure size in inches.
            dpi: Resolution.

        Returns:
            Matplotlib figure object, or ``None`` if matplotlib is unavailable.
        """
        if not _HAS_PLOTTING:
            logger.warning("matplotlib/seaborn not available; skipping plot.")
            return None

        comparison = self.compare(metrics=metrics)

        # Flatten to plottable form
        tasks = sorted(comparison.keys())
        models = sorted(self._results.keys())
        metric_names = set()
        for task_data in comparison.values():
            for model_data in task_data.values():
                metric_names.update(model_data.keys())
        metric_names_sorted = sorted(metric_names)

        if metrics:
            metric_names_sorted = [m for m in metric_names_sorted if m in metrics]

        n_metrics = len(metric_names_sorted)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize, squeeze=False)

        palette = sns.color_palette("Set2", n_colors=len(models))

        for mi, metric in enumerate(metric_names_sorted):
            ax = axes[0, mi]
            x = np.arange(len(tasks))
            width = 0.8 / len(models)

            for ci, model in enumerate(models):
                values = []
                for task in tasks:
                    val = comparison.get(task, {}).get(model, {}).get(metric, 0.0)
                    values.append(val)
                offset = (ci - len(models) / 2 + 0.5) * width
                ax.bar(
                    x + offset,
                    values,
                    width=width,
                    label=model if mi == 0 else None,
                    color=palette[ci],
                    edgecolor="white",
                    linewidth=0.5,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [t.replace("_", "\n") for t in tasks],
                fontsize=7,
                rotation=30,
                ha="right",
            )
            ax.set_title(metric.upper(), fontsize=10, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if len(models) > 1:
            axes[0, 0].legend(
                fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.15)
            )

        fig.suptitle("Model Comparison", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()

        if output_path:
            fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
            logger.info("Comparison plot saved to {}", output_path)

        return fig

    # -- LaTeX export --------------------------------------------------------

    def generate_latex_table(
        self,
        metrics: Optional[List[str]] = None,
        caption: str = "Model comparison across genomic benchmark tasks.",
        label: str = "tab:comparison",
        highlight_best: bool = True,
    ) -> str:
        """Generate a LaTeX table comparing models.

        Args:
            metrics: Subset of metrics.
            caption: Table caption.
            label: LaTeX label.
            highlight_best: Bold the best value per task/metric.

        Returns:
            LaTeX table string.
        """
        comparison = self.compare(metrics=metrics)
        tasks = sorted(comparison.keys())
        models = sorted(self._results.keys())

        # Determine metrics
        all_mets: set[str] = set()
        for task_data in comparison.values():
            for model_data in task_data.values():
                all_mets.update(model_data.keys())
        met_list = sorted(all_mets)

        # Build header
        n_cols = 1 + len(models)  # task + models
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{l" + "c" * len(models) + "}",
            r"\toprule",
        ]

        # Sub-header per metric
        header = "Task / Metric & " + " & ".join(models) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")

        for task in tasks:
            task_display = task.replace("_", " ").title()
            lines.append(rf"\textbf{{{task_display}}} \\")

            for met in met_list:
                values = []
                float_vals = []
                for model in models:
                    v = comparison[task].get(model, {}).get(met, float("nan"))
                    float_vals.append(v)
                    values.append(v)

                # Determine best (highest for auroc/auprc/f1/mcc, lowest for ece)
                higher_is_better = met not in {"ece", "brier_score", "mse"}
                if highlight_best:
                    valid = [v for v in float_vals if not np.isnan(v)]
                    if valid:
                        best = max(valid) if higher_is_better else min(valid)
                    else:
                        best = None
                else:
                    best = None

                row_parts = [f"\\quad {met.upper()}"]
                for v in values:
                    cell = f"{v:.4f}" if not np.isnan(v) else "--"
                    if best is not None and not np.isnan(v) and abs(v - best) < 1e-8:
                        cell = rf"\textbf{{{cell}}}"
                    row_parts.append(cell)

                lines.append(" & ".join(row_parts) + r" \\")

            lines.append(r"\midrule")

        # Replace last midrule with bottomrule
        lines[-1] = r"\bottomrule"
        lines.extend([
            r"\end{tabular}}",
            r"\end{table}",
        ])

        return "\n".join(lines)
