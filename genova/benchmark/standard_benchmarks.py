"""Standard genomics benchmarks for foundation model evaluation.

Implements the BEND benchmark tasks (gene finding, promoter detection,
splice site prediction, enhancer-promoter interaction, chromatin
accessibility) and Nucleotide Transformer tasks (histone modification
prediction, regulatory element classification).

Provides a linear probing evaluation protocol alongside fine-tuning and
few-shot evaluation modes.

Example::

    from genova.benchmark.standard_benchmarks import (
        run_standard_benchmarks,
        LinearProbe,
        GeneFinidingBenchmark,
    )

    results = run_standard_benchmarks(model, data_dir="./data", tasks="all")
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from loguru import logger

from genova.benchmark.tasks import (
    BenchmarkDataset,
    BenchmarkTask,
    _compute_all_metrics,
    TASK_REGISTRY,
)


# ---------------------------------------------------------------------------
# LinearProbe
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    """Linear probing head for evaluating frozen encoder representations.

    Freezes the encoder and trains a single linear layer on top of the
    mean-pooled hidden states.

    Args:
        input_dim: Dimensionality of the encoder output.
        num_classes: Number of output classes (1 for binary).
        dropout: Dropout probability before the linear layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Encoder output tensor of shape ``(B, D)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        return self.linear(self.dropout(x))


# ---------------------------------------------------------------------------
# Feature extraction utility
# ---------------------------------------------------------------------------


@torch.no_grad()
def _extract_features(
    model: nn.Module,
    input_ids: Tensor,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Extract mean-pooled features from a frozen encoder.

    Args:
        model: Encoder model.
        input_ids: ``(N, L)`` token id tensor.
        batch_size: Inference batch size.
        device: Computation device.

    Returns:
        Features array of shape ``(N, D)``.
    """
    model.eval()
    model.to(device)
    all_features: List[np.ndarray] = []

    for start in range(0, len(input_ids), batch_size):
        batch = input_ids[start: start + batch_size].to(device)
        attention_mask = (batch != 0).long()

        outputs = model(input_ids=batch, attention_mask=attention_mask)

        if isinstance(outputs, dict):
            hidden = outputs.get("last_hidden_state")
            if hidden is None:
                hidden = outputs.get("hidden_states", outputs.get("logits"))
                if isinstance(hidden, (list, tuple)):
                    hidden = hidden[-1]
        elif isinstance(outputs, Tensor):
            hidden = outputs
        else:
            hidden = getattr(outputs, "last_hidden_state", outputs)

        if isinstance(hidden, (list, tuple)):
            hidden = hidden[-1]

        # Mean-pool
        if hidden.dim() == 3:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = hidden

        all_features.append(pooled.cpu().float().numpy())

    return np.concatenate(all_features, axis=0)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _train_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 1,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> LinearProbe:
    """Train a linear probe on extracted features.

    Args:
        features: ``(N, D)`` feature array.
        labels: ``(N,)`` label array.
        num_classes: Number of output classes.
        lr: Learning rate.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        device: Device.

    Returns:
        Trained :class:`LinearProbe`.
    """
    input_dim = features.shape[1]
    probe = LinearProbe(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    features_t = torch.tensor(features, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    n = len(features_t)
    probe.train()

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0

        for start in range(0, n, batch_size):
            idx = perm[start: start + batch_size]
            x = features_t[idx].to(device)
            y = labels_t[idx].to(device)

            logits = probe(x)

            if num_classes == 1:
                loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), y.float()
                )
            else:
                loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    probe.eval()
    return probe


@torch.no_grad()
def _evaluate_probe(
    probe: LinearProbe,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Evaluate a linear probe on test features.

    Args:
        probe: Trained linear probe.
        features: ``(N, D)`` test features.
        labels: ``(N,)`` test labels.
        device: Device.

    Returns:
        Metrics dictionary.
    """
    features_t = torch.tensor(features, dtype=torch.float32, device=device)
    logits = probe(features_t)

    if logits.size(-1) == 1:
        scores = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
    else:
        scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    return _compute_all_metrics(scores, labels)


# ---------------------------------------------------------------------------
# Standard Benchmark base class
# ---------------------------------------------------------------------------


class StandardBenchmark(BenchmarkTask):
    """Base class for standard genomics benchmarks.

    Extends :class:`BenchmarkTask` with support for linear probing,
    fine-tuning, and few-shot evaluation protocols.

    Args:
        task_name: Human-readable task name.
        data_dir: Path to data directory.
        batch_size: Inference batch size.
        device: Torch device string.
        num_classes: Number of classification classes.
        sequence_length: Expected input sequence length.
    """

    def __init__(
        self,
        task_name: str,
        data_dir: Optional[str] = None,
        batch_size: int = 32,
        device: str = "cpu",
        num_classes: int = 1,
        sequence_length: int = 512,
    ) -> None:
        super().__init__(
            task_name=task_name,
            data_dir=data_dir,
            batch_size=batch_size,
            device=device,
        )
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        # Separate train / test datasets for probing
        self.train_dataset: Optional[BenchmarkDataset] = None
        self.test_dataset: Optional[BenchmarkDataset] = None

    def _load_split(
        self,
        filename: str,
        synthetic_size: int = 200,
        seq_len: int = 300,
    ) -> BenchmarkDataset:
        """Load a data split from TSV or generate synthetic data.

        Args:
            filename: TSV filename within the data directory.
            synthetic_size: Number of synthetic samples if file not found.
            seq_len: Length of synthetic sequences.

        Returns:
            Loaded or synthetic :class:`BenchmarkDataset`.
        """
        sequences: List[str] = []
        labels: List[int] = []

        data_file = self.data_dir / filename if self.data_dir else None

        if data_file and data_file.exists():
            logger.info("Loading {} from {}", self.task_name, data_file)
            with open(data_file) as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    sequences.append(row["sequence"])
                    labels.append(int(row["label"]))
        else:
            logger.warning(
                "{} data not found; generating synthetic dataset for {}.",
                filename,
                self.task_name,
            )
            rng = np.random.RandomState(hash(self.task_name) % (2**31))
            for _ in range(synthetic_size):
                seq = "".join(rng.choice(list("ACGT"), size=seq_len))
                sequences.append(seq)
                labels.append(int(rng.random() > 0.5))

        return BenchmarkDataset(
            sequences=sequences,
            labels=np.array(labels, dtype=np.int64),
        )

    @torch.no_grad()
    def evaluate_linear_probe(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_length: int = 512,
        lr: float = 1e-3,
        epochs: int = 50,
    ) -> Dict[str, float]:
        """Evaluate using the linear probing protocol.

        Freezes the encoder, extracts features, trains a linear head,
        and evaluates on the test set.

        Args:
            model: Encoder model (frozen during probing).
            tokenizer: Compatible tokenizer.
            max_length: Maximum sequence length.
            lr: Learning rate for the probe.
            epochs: Training epochs for the probe.

        Returns:
            Metrics dictionary.
        """
        if self.train_dataset is None or self.test_dataset is None:
            self.load_data()
        assert self.train_dataset is not None
        assert self.test_dataset is not None

        device = torch.device(self.device)

        # Extract features
        train_ids = self.preprocess(
            self.train_dataset.sequences, tokenizer, max_length
        )
        test_ids = self.preprocess(
            self.test_dataset.sequences, tokenizer, max_length
        )

        train_features = _extract_features(
            model, train_ids, self.batch_size, device
        )
        test_features = _extract_features(
            model, test_ids, self.batch_size, device
        )

        # Train probe
        probe = _train_linear_probe(
            train_features,
            self.train_dataset.labels,
            num_classes=self.num_classes,
            lr=lr,
            epochs=epochs,
            device=device,
        )

        # Evaluate
        metrics = _evaluate_probe(
            probe, test_features, self.test_dataset.labels, device
        )
        logger.info(
            "Linear probe {} metrics: {}", self.task_name, metrics
        )
        return metrics

    @torch.no_grad()
    def evaluate_few_shot(
        self,
        model: nn.Module,
        tokenizer: Any,
        n_shots: int = 16,
        max_length: int = 512,
        n_trials: int = 5,
    ) -> Dict[str, float]:
        """Evaluate using few-shot learning.

        Samples *n_shots* examples per class from the training set, trains
        a linear probe, and evaluates on the test set.  Repeats for
        *n_trials* and reports the mean metrics.

        Args:
            model: Encoder model.
            tokenizer: Compatible tokenizer.
            n_shots: Number of examples per class.
            max_length: Maximum sequence length.
            n_trials: Number of independent trials.

        Returns:
            Averaged metrics dictionary.
        """
        if self.train_dataset is None or self.test_dataset is None:
            self.load_data()
        assert self.train_dataset is not None
        assert self.test_dataset is not None

        device = torch.device(self.device)

        # Extract all train features
        train_ids = self.preprocess(
            self.train_dataset.sequences, tokenizer, max_length
        )
        test_ids = self.preprocess(
            self.test_dataset.sequences, tokenizer, max_length
        )

        all_train_features = _extract_features(
            model, train_ids, self.batch_size, device
        )
        test_features = _extract_features(
            model, test_ids, self.batch_size, device
        )

        all_metrics: List[Dict[str, float]] = []

        for trial in range(n_trials):
            rng = np.random.RandomState(trial)
            # Sample n_shots per class
            selected_indices: List[int] = []
            for cls in range(max(2, self.num_classes)):
                cls_indices = np.where(self.train_dataset.labels == cls)[0]
                if len(cls_indices) == 0:
                    continue
                chosen = rng.choice(
                    cls_indices, size=min(n_shots, len(cls_indices)), replace=False
                )
                selected_indices.extend(chosen.tolist())

            if not selected_indices:
                continue

            few_features = all_train_features[selected_indices]
            few_labels = self.train_dataset.labels[selected_indices]

            probe = _train_linear_probe(
                few_features,
                few_labels,
                num_classes=self.num_classes,
                lr=1e-3,
                epochs=100,
                device=device,
            )

            metrics = _evaluate_probe(
                probe, test_features, self.test_dataset.labels, device
            )
            all_metrics.append(metrics)

        # Average across trials
        avg_metrics: Dict[str, float] = {}
        if all_metrics:
            for key in all_metrics[0]:
                avg_metrics[key] = float(
                    np.mean([m[key] for m in all_metrics])
                )

        logger.info(
            "Few-shot ({}-shot, {} trials) {} metrics: {}",
            n_shots,
            n_trials,
            self.task_name,
            avg_metrics,
        )
        return avg_metrics


# ---------------------------------------------------------------------------
# BEND benchmark tasks
# ---------------------------------------------------------------------------


class GeneFindingBenchmark(StandardBenchmark):
    """Gene finding benchmark from BEND.

    Binary classification of coding vs non-coding sequences.

    Args:
        data_dir: Directory containing ``gene_finding_train.tsv`` and
            ``gene_finding_test.tsv``.
        batch_size: Inference batch size.
        device: Torch device string.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            task_name="bend_gene_finding",
            data_dir=data_dir,
            batch_size=batch_size,
            device=device,
            num_classes=1,
            sequence_length=512,
        )

    def load_data(self) -> BenchmarkDataset:
        """Load gene finding dataset."""
        self.train_dataset = self._load_split("gene_finding_train.tsv")
        self.test_dataset = self._load_split("gene_finding_test.tsv")
        self.dataset = self.test_dataset
        logger.info(
            "Loaded gene finding: {} train, {} test",
            len(self.train_dataset),
            len(self.test_dataset),
        )
        return self.dataset


class PromoterDetectionBenchmark(StandardBenchmark):
    """Promoter detection benchmark from BEND.

    Supports TATA and non-TATA promoter subtypes.

    Args:
        data_dir: Data directory.
        promoter_type: ``"all"``, ``"tata"``, or ``"non_tata"``.
        batch_size: Inference batch size.
        device: Torch device string.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        promoter_type: str = "all",
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        if promoter_type not in ("all", "tata", "non_tata"):
            raise ValueError(
                f"promoter_type must be 'all', 'tata', or 'non_tata', "
                f"got {promoter_type!r}"
            )
        self.promoter_type = promoter_type
        super().__init__(
            task_name=f"bend_promoter_{promoter_type}",
            data_dir=data_dir,
            batch_size=batch_size,
            device=device,
            num_classes=1,
            sequence_length=300,
        )

    def load_data(self) -> BenchmarkDataset:
        """Load promoter detection dataset."""
        suffix = f"_{self.promoter_type}" if self.promoter_type != "all" else ""
        self.train_dataset = self._load_split(
            f"promoter{suffix}_train.tsv", seq_len=300
        )
        self.test_dataset = self._load_split(
            f"promoter{suffix}_test.tsv", seq_len=300
        )
        self.dataset = self.test_dataset
        logger.info(
            "Loaded promoter detection ({}): {} train, {} test",
            self.promoter_type,
            len(self.train_dataset),
            len(self.test_dataset),
        )
        return self.dataset


class SpliceSiteBenchmark(StandardBenchmark):
    """Splice site prediction benchmark from BEND.

    Supports donor and acceptor subtypes.

    Args:
        data_dir: Data directory.
        site_type: ``"all"``, ``"donor"``, or ``"acceptor"``.
        batch_size: Inference batch size.
        device: Torch device string.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        site_type: str = "all",
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        if site_type not in ("all", "donor", "acceptor"):
            raise ValueError(
                f"site_type must be 'all', 'donor', or 'acceptor', "
                f"got {site_type!r}"
            )
        self.site_type = site_type
        super().__init__(
            task_name=f"bend_splice_{site_type}",
            data_dir=data_dir,
            batch_size=batch_size,
            device=device,
            num_classes=1,
            sequence_length=400,
        )

    def load_data(self) -> BenchmarkDataset:
        """Load splice site dataset."""
        suffix = f"_{self.site_type}" if self.site_type != "all" else ""
        self.train_dataset = self._load_split(
            f"splice{suffix}_train.tsv", seq_len=400
        )
        self.test_dataset = self._load_split(
            f"splice{suffix}_test.tsv", seq_len=400
        )
        self.dataset = self.test_dataset
        logger.info(
            "Loaded splice site ({}): {} train, {} test",
            self.site_type,
            len(self.train_dataset),
            len(self.test_dataset),
        )
        return self.dataset


class EnhancerPromoterBenchmark(StandardBenchmark):
    """Enhancer-promoter interaction prediction from BEND.

    Binary classification of whether an enhancer-promoter pair interacts.

    Args:
        data_dir: Data directory.
        batch_size: Inference batch size.
        device: Torch device string.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            task_name="bend_enhancer_promoter",
            data_dir=data_dir,
            batch_size=batch_size,
            device=device,
            num_classes=1,
            sequence_length=1000,
        )

    def load_data(self) -> BenchmarkDataset:
        """Load enhancer-promoter interaction dataset."""
        self.train_dataset = self._load_split(
            "enhancer_promoter_train.tsv", seq_len=1000
        )
        self.test_dataset = self._load_split(
            "enhancer_promoter_test.tsv", seq_len=1000
        )
        self.dataset = self.test_dataset
        return self.dataset


class ChromatinAccessibilityBenchmark(StandardBenchmark):
    """Chromatin accessibility prediction from BEND.

    Binary classification of open vs closed chromatin regions.

    Args:
        data_dir: Data directory.
        batch_size: Inference batch size.
        device: Torch device string.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            task_name="bend_chromatin_accessibility",
            data_dir=data_dir,
            batch_size=batch_size,
            device=device,
            num_classes=1,
            sequence_length=500,
        )

    def load_data(self) -> BenchmarkDataset:
        """Load chromatin accessibility dataset."""
        self.train_dataset = self._load_split(
            "chromatin_accessibility_train.tsv", seq_len=500
        )
        self.test_dataset = self._load_split(
            "chromatin_accessibility_test.tsv", seq_len=500
        )
        self.dataset = self.test_dataset
        return self.dataset


# ---------------------------------------------------------------------------
# Nucleotide Transformer benchmark tasks
# ---------------------------------------------------------------------------


class HistoneModificationBenchmark(StandardBenchmark):
    """Histone modification prediction from Nucleotide Transformer benchmarks.

    Predicts presence/absence of a specific histone mark.

    Args:
        data_dir: Data directory.
        mark: Histone mark name (e.g. ``"H3K4me3"``, ``"H3K27ac"``).
        batch_size: Inference batch size.
        device: Torch device string.
    """

    SUPPORTED_MARKS = (
        "H3K4me3", "H3K27ac", "H3K4me1", "H3K36me3",
        "H3K9me3", "H3K27me3", "H3K9ac", "H4K20me1",
    )

    def __init__(
        self,
        data_dir: Optional[str] = None,
        mark: str = "H3K4me3",
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        self.mark = mark
        super().__init__(
            task_name=f"nt_histone_{mark}",
            data_dir=data_dir,
            batch_size=batch_size,
            device=device,
            num_classes=1,
            sequence_length=512,
        )

    def load_data(self) -> BenchmarkDataset:
        """Load histone modification dataset."""
        mark_lower = self.mark.lower()
        self.train_dataset = self._load_split(
            f"histone_{mark_lower}_train.tsv"
        )
        self.test_dataset = self._load_split(
            f"histone_{mark_lower}_test.tsv"
        )
        self.dataset = self.test_dataset
        logger.info(
            "Loaded histone {} benchmark: {} train, {} test",
            self.mark,
            len(self.train_dataset),
            len(self.test_dataset),
        )
        return self.dataset


class RegulatoryElementBenchmark(StandardBenchmark):
    """Regulatory element classification from Nucleotide Transformer benchmarks.

    Multi-class classification of genomic sequences into regulatory element
    categories (promoter, enhancer, silencer, insulator, etc.).

    Args:
        data_dir: Data directory.
        batch_size: Inference batch size.
        device: Torch device string.
        num_classes: Number of regulatory element classes.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        batch_size: int = 32,
        device: str = "cpu",
        num_classes: int = 5,
    ) -> None:
        super().__init__(
            task_name="nt_regulatory_elements",
            data_dir=data_dir,
            batch_size=batch_size,
            device=device,
            num_classes=num_classes,
            sequence_length=512,
        )

    def load_data(self) -> BenchmarkDataset:
        """Load regulatory element dataset."""
        self.train_dataset = self._load_split(
            "regulatory_elements_train.tsv"
        )
        self.test_dataset = self._load_split(
            "regulatory_elements_test.tsv"
        )
        self.dataset = self.test_dataset
        return self.dataset


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STANDARD_BENCHMARK_REGISTRY: Dict[str, type] = {
    "bend_gene_finding": GeneFindingBenchmark,
    "bend_promoter_all": PromoterDetectionBenchmark,
    "bend_promoter_tata": lambda **kw: PromoterDetectionBenchmark(promoter_type="tata", **kw),
    "bend_promoter_non_tata": lambda **kw: PromoterDetectionBenchmark(promoter_type="non_tata", **kw),
    "bend_splice_all": SpliceSiteBenchmark,
    "bend_splice_donor": lambda **kw: SpliceSiteBenchmark(site_type="donor", **kw),
    "bend_splice_acceptor": lambda **kw: SpliceSiteBenchmark(site_type="acceptor", **kw),
    "bend_enhancer_promoter": EnhancerPromoterBenchmark,
    "bend_chromatin_accessibility": ChromatinAccessibilityBenchmark,
    "nt_histone_H3K4me3": lambda **kw: HistoneModificationBenchmark(mark="H3K4me3", **kw),
    "nt_histone_H3K27ac": lambda **kw: HistoneModificationBenchmark(mark="H3K27ac", **kw),
    "nt_regulatory_elements": RegulatoryElementBenchmark,
}


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_standard_benchmarks(
    model: nn.Module,
    data_dir: str,
    tasks: Union[str, List[str]] = "all",
    tokenizer: Any = None,
    evaluation_mode: str = "linear_probe",
    device: str = "cpu",
    batch_size: int = 32,
    max_length: int = 512,
) -> Dict[str, Dict[str, float]]:
    """Run standard genomics benchmarks on a model.

    Args:
        model: Encoder model to evaluate.
        data_dir: Root directory for benchmark data.
        tasks: ``"all"`` to run all benchmarks, or a list of task names.
        tokenizer: Compatible tokenizer.
        evaluation_mode: One of ``"linear_probe"``, ``"fine_tune"``, or
            ``"few_shot"``.
        device: Torch device.
        batch_size: Inference batch size.
        max_length: Maximum sequence length.

    Returns:
        Nested dict ``{task_name: {metric: value}}``.
    """
    if tasks == "all":
        task_names = list(STANDARD_BENCHMARK_REGISTRY.keys())
    elif isinstance(tasks, str):
        task_names = [tasks]
    else:
        task_names = tasks

    results: Dict[str, Dict[str, float]] = {}

    for name in task_names:
        if name not in STANDARD_BENCHMARK_REGISTRY:
            logger.warning("Unknown benchmark task: {}. Skipping.", name)
            continue

        factory = STANDARD_BENCHMARK_REGISTRY[name]
        if callable(factory) and isinstance(factory, type):
            task = factory(
                data_dir=data_dir, batch_size=batch_size, device=device
            )
        else:
            task = factory(
                data_dir=data_dir, batch_size=batch_size, device=device
            )

        try:
            task.load_data()

            if evaluation_mode == "linear_probe" and hasattr(task, "evaluate_linear_probe"):
                metrics = task.evaluate_linear_probe(
                    model, tokenizer, max_length=max_length
                )
            elif evaluation_mode == "few_shot" and hasattr(task, "evaluate_few_shot"):
                metrics = task.evaluate_few_shot(
                    model, tokenizer, max_length=max_length
                )
            else:
                # Fine-tune mode: standard evaluate
                metrics = task.evaluate(
                    model, tokenizer, max_length=max_length
                )

            results[name] = metrics
            logger.info("Benchmark {} complete: {}", name, metrics)

        except Exception as e:
            logger.error("Benchmark {} failed: {}", name, e)
            results[name] = {"error": -1.0}

    logger.info(
        "Standard benchmarks complete: {}/{} tasks succeeded.",
        sum(1 for v in results.values() if "error" not in v),
        len(task_names),
    )
    return results
