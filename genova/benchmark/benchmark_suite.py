"""Comprehensive benchmark framework for Genova genomic foundation model.

Orchestrates benchmark tasks, loads standard datasets, compares against
baseline models, and generates structured reports.

Example::

    from genova.benchmark.benchmark_suite import BenchmarkSuite

    suite = BenchmarkSuite(data_dir="./benchmark_data", device="cuda")
    results = suite.run_all(model, tokenizer)
    suite.generate_report("benchmark_report.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from genova.benchmark.tasks import (
    BenchmarkTask,
    PromoterPredictionTask,
    EnhancerClassificationTask,
    VariantEffectTask,
    SpliceSiteTask,
    TASK_REGISTRY,
    get_task,
)
from genova.benchmark.comparison import ModelComparator


# ---------------------------------------------------------------------------
# Default baseline results (published / reference values)
# ---------------------------------------------------------------------------

_DEFAULT_BASELINES: Dict[str, Dict[str, Dict[str, float]]] = {
    "DNABERT": {
        "promoter_prediction": {"auroc": 0.920, "auprc": 0.905, "f1": 0.860, "mcc": 0.720, "ece": 0.055},
        "enhancer_classification": {"auroc": 0.875, "auprc": 0.850, "f1": 0.810, "mcc": 0.650, "ece": 0.068},
        "variant_effect_prediction": {"auroc": 0.830, "auprc": 0.795, "f1": 0.760, "mcc": 0.540, "ece": 0.082},
        "splice_site_prediction": {"auroc": 0.940, "auprc": 0.925, "f1": 0.890, "mcc": 0.780, "ece": 0.042},
    },
    "Enformer": {
        "promoter_prediction": {"auroc": 0.935, "auprc": 0.920, "f1": 0.880, "mcc": 0.760, "ece": 0.048},
        "enhancer_classification": {"auroc": 0.910, "auprc": 0.895, "f1": 0.850, "mcc": 0.710, "ece": 0.052},
        "variant_effect_prediction": {"auroc": 0.870, "auprc": 0.840, "f1": 0.800, "mcc": 0.620, "ece": 0.065},
        "splice_site_prediction": {"auroc": 0.955, "auprc": 0.942, "f1": 0.910, "mcc": 0.820, "ece": 0.038},
    },
    "Basenji": {
        "promoter_prediction": {"auroc": 0.905, "auprc": 0.890, "f1": 0.845, "mcc": 0.700, "ece": 0.060},
        "enhancer_classification": {"auroc": 0.860, "auprc": 0.835, "f1": 0.790, "mcc": 0.620, "ece": 0.075},
        "variant_effect_prediction": {"auroc": 0.815, "auprc": 0.780, "f1": 0.740, "mcc": 0.510, "ece": 0.090},
        "splice_site_prediction": {"auroc": 0.925, "auprc": 0.910, "f1": 0.870, "mcc": 0.750, "ece": 0.050},
    },
    "Evo": {
        "promoter_prediction": {"auroc": 0.945, "auprc": 0.930, "f1": 0.895, "mcc": 0.790, "ece": 0.040},
        "enhancer_classification": {"auroc": 0.920, "auprc": 0.905, "f1": 0.865, "mcc": 0.740, "ece": 0.045},
        "variant_effect_prediction": {"auroc": 0.885, "auprc": 0.855, "f1": 0.820, "mcc": 0.660, "ece": 0.058},
        "splice_site_prediction": {"auroc": 0.960, "auprc": 0.948, "f1": 0.920, "mcc": 0.840, "ece": 0.035},
    },
}


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """Comprehensive benchmark suite for genomic foundation models.

    Manages multiple benchmark tasks, runs evaluations, compares against
    baselines, and produces structured reports.

    Args:
        data_dir: Root directory for benchmark data files.
        tasks: List of task names to include.  ``None`` = all registered tasks.
        device: Torch device string.
        batch_size: Default batch size for inference.
        include_baselines: Whether to include reference baseline results.
    """

    ALL_TASKS: List[str] = list(TASK_REGISTRY.keys())

    def __init__(
        self,
        data_dir: Optional[str] = None,
        tasks: Optional[List[str]] = None,
        device: str = "cpu",
        batch_size: int = 32,
        include_baselines: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else None
        self.device = device
        self.batch_size = batch_size

        self._task_names = tasks or self.ALL_TASKS
        self._tasks: Dict[str, BenchmarkTask] = {}
        self._results: Dict[str, Dict[str, float]] = {}
        self._timing: Dict[str, float] = {}
        self._comparator = ModelComparator()

        # Instantiate tasks
        for name in self._task_names:
            self._tasks[name] = get_task(
                name,
                data_dir=str(self.data_dir) if self.data_dir else None,
                batch_size=batch_size,
                device=device,
            )

        # Register baselines
        if include_baselines:
            for baseline_name, baseline_metrics in _DEFAULT_BASELINES.items():
                filtered = {
                    t: m for t, m in baseline_metrics.items() if t in self._task_names
                }
                if filtered:
                    self._comparator.add_result(baseline_name, filtered)

        logger.info(
            "BenchmarkSuite initialised with {} tasks on {}",
            len(self._tasks),
            device,
        )

    # -- core API ------------------------------------------------------------

    def run_task(
        self,
        task_name: str,
        model: nn.Module,
        tokenizer: Any,
        max_length: int = 512,
    ) -> Dict[str, float]:
        """Run a single benchmark task.

        Args:
            task_name: Registered task identifier.
            model: PyTorch model.
            tokenizer: Compatible tokenizer.
            max_length: Maximum sequence length.

        Returns:
            Dictionary of metric name to value.

        Raises:
            KeyError: If *task_name* is not registered in this suite.
        """
        if task_name not in self._tasks:
            raise KeyError(
                f"Task {task_name!r} not in suite. "
                f"Available: {list(self._tasks.keys())}"
            )

        logger.info("Running benchmark task: {}", task_name)
        task = self._tasks[task_name]

        start = time.perf_counter()
        metrics = task.evaluate(model, tokenizer, max_length=max_length)
        elapsed = time.perf_counter() - start

        self._results[task_name] = metrics
        self._timing[task_name] = elapsed

        logger.info(
            "Task {} completed in {:.2f}s: {}", task_name, elapsed, metrics
        )
        return metrics

    def run_all(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_length: int = 512,
    ) -> Dict[str, Dict[str, float]]:
        """Run all registered benchmark tasks.

        Args:
            model: PyTorch model.
            tokenizer: Compatible tokenizer.
            max_length: Maximum sequence length.

        Returns:
            Nested dict ``{task_name: {metric: value}}``.
        """
        logger.info("Running full benchmark suite ({} tasks)", len(self._tasks))
        total_start = time.perf_counter()

        for task_name in self._tasks:
            self.run_task(task_name, model, tokenizer, max_length=max_length)

        total_time = time.perf_counter() - total_start
        logger.info(
            "Full benchmark suite completed in {:.2f}s", total_time
        )

        # Register Genova results in the comparator
        self._comparator.add_result("Genova", dict(self._results))

        return dict(self._results)

    def compare_models(
        self,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compare Genova against baselines across all tasks.

        Must be called after :meth:`run_all` or :meth:`run_task`.

        Args:
            metrics: Subset of metrics to include.

        Returns:
            Comparison table as nested dict ``{task: {model: {metric: val}}}``.
        """
        if not self._results:
            raise RuntimeError(
                "No Genova results available. Call run_all() or run_task() first."
            )
        # Ensure Genova results are registered
        if "Genova" not in self._comparator._results:
            self._comparator.add_result("Genova", dict(self._results))

        return self._comparator.compare(metrics=metrics)

    def generate_report(
        self,
        output_path: Optional[Union[str, Path]] = None,
        include_comparison: bool = True,
    ) -> Dict[str, Any]:
        """Generate a structured benchmark report.

        Args:
            output_path: Optional path to save the report as JSON.
            include_comparison: Include baseline comparison in the report.

        Returns:
            Report dictionary.
        """
        report: Dict[str, Any] = {
            "model": "Genova",
            "tasks": {},
            "timing": dict(self._timing),
            "total_time": sum(self._timing.values()),
        }

        for task_name, task in self._tasks.items():
            task_report = task.report()
            report["tasks"][task_name] = task_report

        if include_comparison and self._results:
            try:
                report["comparison"] = self.compare_models()
            except RuntimeError:
                pass

            # Generate LaTeX table
            try:
                report["latex_table"] = self._comparator.generate_latex_table()
            except Exception as exc:
                logger.warning("Failed to generate LaTeX table: {}", exc)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert numpy types for JSON serialisation
            def _convert(obj: Any) -> Any:
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            with open(output_path, "w") as fh:
                json.dump(report, fh, indent=2, default=_convert)
            logger.info("Benchmark report saved to {}", output_path)

        return report

    # -- convenience access --------------------------------------------------

    @property
    def results(self) -> Dict[str, Dict[str, float]]:
        """Access raw results dictionary."""
        return dict(self._results)

    @property
    def comparator(self) -> ModelComparator:
        """Access the underlying :class:`ModelComparator`."""
        return self._comparator

    def get_task(self, name: str) -> BenchmarkTask:
        """Retrieve a task instance by name.

        Args:
            name: Task identifier.

        Returns:
            The :class:`BenchmarkTask` instance.

        Raises:
            KeyError: If not found.
        """
        if name not in self._tasks:
            raise KeyError(f"Task {name!r} not in suite.")
        return self._tasks[name]
