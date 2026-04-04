"""Cross-validation framework for genomic model evaluation.

Provides K-fold, stratified K-fold, leave-one-chromosome-out, and
nested cross-validation with aggregated metrics.

Example::

    from genova.evaluation.cross_validation import CrossValidator

    cv = CrossValidator()
    result = cv.kfold(dataset, model_fn=train_and_eval, k=5)
    print(result.mean_metrics)
    print(result.std_metrics)

    # Chromosome-level CV (biologically appropriate)
    result = cv.chromosome_cv(dataset, model_fn=train_and_eval)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """Results from a single cross-validation fold.

    Attributes:
        fold_idx: Index of the fold (0-based).
        metrics: Dictionary of metric name to value.
        train_size: Number of training samples in this fold.
        val_size: Number of validation samples in this fold.
        fold_name: Optional human-readable name (e.g. chromosome name).
    """

    fold_idx: int
    metrics: Dict[str, float]
    train_size: int
    val_size: int
    fold_name: Optional[str] = None


@dataclass
class CVResult:
    """Aggregated cross-validation results.

    Attributes:
        fold_results: Per-fold result objects.
        mean_metrics: Mean of each metric across folds.
        std_metrics: Standard deviation of each metric across folds.
        cv_type: Type of CV performed (``"kfold"``, ``"stratified"``,
            ``"chromosome"``, ``"nested"``).
        n_folds: Number of folds.
        best_hyperparams: Best hyperparameters found (nested CV only).
    """

    fold_results: List[FoldResult]
    mean_metrics: Dict[str, float] = field(default_factory=dict)
    std_metrics: Dict[str, float] = field(default_factory=dict)
    cv_type: str = "kfold"
    n_folds: int = 0
    best_hyperparams: Optional[Dict[str, Any]] = None

    def summary(self) -> str:
        """Return a human-readable summary string.

        Returns:
            Multi-line string with per-metric mean +/- std.
        """
        lines = [f"Cross-Validation Results ({self.cv_type}, {self.n_folds} folds):"]
        for metric in sorted(self.mean_metrics.keys()):
            mean = self.mean_metrics[metric]
            std = self.std_metrics.get(metric, 0.0)
            lines.append(f"  {metric}: {mean:.4f} +/- {std:.4f}")
        if self.best_hyperparams:
            lines.append(f"  Best hyperparams: {self.best_hyperparams}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Index-based dataset protocol
# ---------------------------------------------------------------------------

# The framework operates on indices so it is dataset-agnostic.  The
# ``model_fn`` callback receives train/val index arrays and returns a
# metrics dict.  This avoids coupling to any specific dataset class.


def _aggregate_metrics(fold_results: List[FoldResult]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute mean and std of metrics across folds.

    Args:
        fold_results: Per-fold results.

    Returns:
        Tuple of ``(mean_metrics, std_metrics)``.
    """
    if not fold_results:
        return {}, {}

    all_metrics: Dict[str, List[float]] = defaultdict(list)
    for fr in fold_results:
        for k, v in fr.metrics.items():
            all_metrics[k].append(v)

    mean_metrics: Dict[str, float] = {}
    std_metrics: Dict[str, float] = {}
    for k, values in all_metrics.items():
        arr = np.array(values, dtype=np.float64)
        mean_metrics[k] = float(arr.mean())
        std_metrics[k] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    return mean_metrics, std_metrics


# ---------------------------------------------------------------------------
# Cross Validator
# ---------------------------------------------------------------------------


class CrossValidator:
    """Cross-validation framework for genomic models.

    All methods accept a ``model_fn`` callback with signature::

        def model_fn(train_indices, val_indices) -> Dict[str, float]

    The callback is responsible for subsetting data, training (if
    applicable), evaluating, and returning a metrics dictionary.

    Args:
        seed: Random seed for reproducible fold splits.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    # ------------------------------------------------------------------
    # Standard K-fold
    # ------------------------------------------------------------------

    def kfold(
        self,
        dataset_size: int,
        model_fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]],
        k: int = 5,
    ) -> CVResult:
        """Run standard K-fold cross-validation.

        Args:
            dataset_size: Total number of samples in the dataset.
            model_fn: Callback ``(train_idx, val_idx) -> metrics``.
            k: Number of folds.  Typical values are 5 or 10.

        Returns:
            :class:`CVResult` with per-fold and aggregate metrics.

        Raises:
            ValueError: If *k* < 2 or *k* > *dataset_size*.
        """
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        if k > dataset_size:
            raise ValueError(
                f"k={k} exceeds dataset size {dataset_size}"
            )

        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(dataset_size)

        fold_sizes = np.full(k, dataset_size // k, dtype=int)
        fold_sizes[: dataset_size % k] += 1

        fold_results: List[FoldResult] = []
        current = 0

        for fold_idx in range(k):
            val_start = current
            val_end = current + fold_sizes[fold_idx]
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate(
                [indices[:val_start], indices[val_end:]]
            )
            current = val_end

            logger.info(
                "K-fold {}/{}: train={}, val={}",
                fold_idx + 1,
                k,
                len(train_indices),
                len(val_indices),
            )

            metrics = model_fn(train_indices, val_indices)

            fold_results.append(
                FoldResult(
                    fold_idx=fold_idx,
                    metrics=metrics,
                    train_size=len(train_indices),
                    val_size=len(val_indices),
                )
            )

        mean_m, std_m = _aggregate_metrics(fold_results)

        result = CVResult(
            fold_results=fold_results,
            mean_metrics=mean_m,
            std_metrics=std_m,
            cv_type="kfold",
            n_folds=k,
        )
        logger.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # Stratified K-fold
    # ------------------------------------------------------------------

    def stratified_kfold(
        self,
        labels: np.ndarray,
        model_fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]],
        k: int = 5,
    ) -> CVResult:
        """Run stratified K-fold cross-validation.

        Maintains approximately equal class distribution across folds.

        Args:
            labels: Integer class labels, shape ``(N,)``.
            model_fn: Callback ``(train_idx, val_idx) -> metrics``.
            k: Number of folds.

        Returns:
            :class:`CVResult` with per-fold and aggregate metrics.

        Raises:
            ValueError: If *k* < 2 or exceeds the smallest class count.
        """
        labels = np.asarray(labels).ravel()
        n = len(labels)

        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")

        rng = np.random.RandomState(self.seed)

        # Group indices by class
        classes = np.unique(labels)
        class_indices: Dict[int, np.ndarray] = {}
        for cls in classes:
            cls_idx = np.where(labels == cls)[0]
            rng.shuffle(cls_idx)
            class_indices[int(cls)] = cls_idx
            if len(cls_idx) < k:
                raise ValueError(
                    f"Class {cls} has {len(cls_idx)} samples, need at "
                    f"least k={k} for stratified splitting."
                )

        # Distribute each class across k folds
        fold_indices: List[List[int]] = [[] for _ in range(k)]
        for cls in classes:
            idx = class_indices[int(cls)]
            fold_sizes = np.full(k, len(idx) // k, dtype=int)
            fold_sizes[: len(idx) % k] += 1
            current = 0
            for fold_idx in range(k):
                end = current + fold_sizes[fold_idx]
                fold_indices[fold_idx].extend(idx[current:end].tolist())
                current = end

        # Shuffle within each fold
        for fi in fold_indices:
            rng.shuffle(fi)

        fold_results: List[FoldResult] = []

        for fold_idx in range(k):
            val_indices = np.array(fold_indices[fold_idx])
            train_indices = np.concatenate(
                [np.array(fold_indices[j]) for j in range(k) if j != fold_idx]
            )

            logger.info(
                "Stratified K-fold {}/{}: train={}, val={}",
                fold_idx + 1,
                k,
                len(train_indices),
                len(val_indices),
            )

            metrics = model_fn(train_indices, val_indices)

            fold_results.append(
                FoldResult(
                    fold_idx=fold_idx,
                    metrics=metrics,
                    train_size=len(train_indices),
                    val_size=len(val_indices),
                )
            )

        mean_m, std_m = _aggregate_metrics(fold_results)

        result = CVResult(
            fold_results=fold_results,
            mean_metrics=mean_m,
            std_metrics=std_m,
            cv_type="stratified_kfold",
            n_folds=k,
        )
        logger.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # Leave-one-chromosome-out
    # ------------------------------------------------------------------

    def chromosome_cv(
        self,
        chromosomes: np.ndarray,
        model_fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]],
        held_out_chroms: Optional[List[str]] = None,
    ) -> CVResult:
        """Leave-one-chromosome-out cross-validation.

        Each fold holds out all samples from one chromosome.  This is
        biologically appropriate because it prevents data leakage from
        genomic proximity.

        Args:
            chromosomes: Array of chromosome labels (e.g. ``"chr1"``,
                ``"chr2"``, ...), shape ``(N,)``.
            model_fn: Callback ``(train_idx, val_idx) -> metrics``.
            held_out_chroms: If provided, only these chromosomes are
                used as held-out folds.  Otherwise every unique
                chromosome becomes a fold.

        Returns:
            :class:`CVResult` with per-fold and aggregate metrics.
        """
        chromosomes = np.asarray(chromosomes)
        unique_chroms = sorted(np.unique(chromosomes).tolist())

        if held_out_chroms is not None:
            fold_chroms = [c for c in held_out_chroms if c in unique_chroms]
        else:
            fold_chroms = unique_chroms

        if len(fold_chroms) < 2:
            raise ValueError(
                f"Need at least 2 chromosomes for CV, got {len(fold_chroms)}"
            )

        fold_results: List[FoldResult] = []

        for fold_idx, chrom in enumerate(fold_chroms):
            val_indices = np.where(chromosomes == chrom)[0]
            train_indices = np.where(chromosomes != chrom)[0]

            # If held_out_chroms is specified, also exclude non-fold chroms
            # from validation but keep them in training
            if held_out_chroms is not None:
                # Only exclude the current chrom from training
                train_mask = chromosomes != chrom
                train_indices = np.where(train_mask)[0]

            logger.info(
                "Chromosome CV fold {}/{} ({}): train={}, val={}",
                fold_idx + 1,
                len(fold_chroms),
                chrom,
                len(train_indices),
                len(val_indices),
            )

            metrics = model_fn(train_indices, val_indices)

            fold_results.append(
                FoldResult(
                    fold_idx=fold_idx,
                    metrics=metrics,
                    train_size=len(train_indices),
                    val_size=len(val_indices),
                    fold_name=str(chrom),
                )
            )

        mean_m, std_m = _aggregate_metrics(fold_results)

        result = CVResult(
            fold_results=fold_results,
            mean_metrics=mean_m,
            std_metrics=std_m,
            cv_type="chromosome_cv",
            n_folds=len(fold_chroms),
        )
        logger.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # Nested CV
    # ------------------------------------------------------------------

    def nested_cv(
        self,
        dataset_size: int,
        model_fn: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], Dict[str, float]],
        hyperparam_grid: List[Dict[str, Any]],
        outer_k: int = 5,
        inner_k: int = 3,
        labels: Optional[np.ndarray] = None,
    ) -> CVResult:
        """Nested cross-validation for unbiased hyperparameter selection.

        The outer loop estimates generalization performance.  The inner
        loop selects the best hyperparameters for each outer fold.

        Args:
            dataset_size: Total number of samples.
            model_fn: Callback
                ``(train_idx, val_idx, hyperparams) -> metrics``.
                The callback receives hyperparameters as the third
                argument.
            hyperparam_grid: List of hyperparameter dicts to search.
            outer_k: Number of outer folds.
            inner_k: Number of inner folds for hyperparameter selection.
            labels: Optional labels for stratified splitting.

        Returns:
            :class:`CVResult` with per-fold metrics and the best
            hyperparameters found across inner folds.
        """
        if outer_k < 2 or inner_k < 2:
            raise ValueError("Both outer_k and inner_k must be >= 2")

        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(dataset_size)

        # Outer fold splits
        fold_sizes = np.full(outer_k, dataset_size // outer_k, dtype=int)
        fold_sizes[: dataset_size % outer_k] += 1

        fold_results: List[FoldResult] = []
        best_params_per_fold: List[Dict[str, Any]] = []
        current = 0

        for outer_idx in range(outer_k):
            val_start = current
            val_end = current + fold_sizes[outer_idx]
            outer_val = indices[val_start:val_end]
            outer_train = np.concatenate(
                [indices[:val_start], indices[val_end:]]
            )
            current = val_end

            logger.info(
                "Nested CV outer fold {}/{}: train={}, val={}",
                outer_idx + 1,
                outer_k,
                len(outer_train),
                len(outer_val),
            )

            # Inner CV: select best hyperparameters
            best_inner_score = -float("inf")
            best_params: Dict[str, Any] = hyperparam_grid[0] if hyperparam_grid else {}

            for params in hyperparam_grid:
                inner_scores: List[float] = []

                inner_fold_sizes = np.full(
                    inner_k, len(outer_train) // inner_k, dtype=int
                )
                inner_fold_sizes[: len(outer_train) % inner_k] += 1

                inner_rng = np.random.RandomState(self.seed + outer_idx + 1)
                inner_perm = inner_rng.permutation(len(outer_train))
                inner_current = 0

                for inner_idx in range(inner_k):
                    iv_start = inner_current
                    iv_end = inner_current + inner_fold_sizes[inner_idx]
                    inner_val_rel = inner_perm[iv_start:iv_end]
                    inner_train_rel = np.concatenate(
                        [inner_perm[:iv_start], inner_perm[iv_end:]]
                    )
                    inner_current = iv_end

                    # Map relative indices back to dataset indices
                    inner_train_idx = outer_train[inner_train_rel]
                    inner_val_idx = outer_train[inner_val_rel]

                    inner_metrics = model_fn(
                        inner_train_idx, inner_val_idx, params
                    )

                    # Use the first metric as the selection criterion
                    score_key = next(iter(inner_metrics))
                    inner_scores.append(inner_metrics[score_key])

                mean_inner = float(np.mean(inner_scores))
                if mean_inner > best_inner_score:
                    best_inner_score = mean_inner
                    best_params = params

            logger.info(
                "  Best inner params: {} (score={:.4f})",
                best_params,
                best_inner_score,
            )
            best_params_per_fold.append(best_params)

            # Evaluate on outer fold with best params
            outer_metrics = model_fn(outer_train, outer_val, best_params)

            fold_results.append(
                FoldResult(
                    fold_idx=outer_idx,
                    metrics=outer_metrics,
                    train_size=len(outer_train),
                    val_size=len(outer_val),
                )
            )

        mean_m, std_m = _aggregate_metrics(fold_results)

        # Select overall best hyperparams (most frequently chosen)
        from collections import Counter

        param_strs = [str(sorted(p.items())) for p in best_params_per_fold]
        most_common_str = Counter(param_strs).most_common(1)[0][0]
        overall_best = best_params_per_fold[param_strs.index(most_common_str)]

        result = CVResult(
            fold_results=fold_results,
            mean_metrics=mean_m,
            std_metrics=std_m,
            cv_type="nested_cv",
            n_folds=outer_k,
            best_hyperparams=overall_best,
        )
        logger.info(result.summary())
        return result
