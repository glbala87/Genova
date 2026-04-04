"""Individual benchmark task definitions for genomic model evaluation.

Each task provides a standard interface for loading data, running evaluation,
and generating reports.  Tasks are designed to work with any model that accepts
tokenised DNA sequences and returns logits or probabilities.

Example::

    from genova.benchmark.tasks import PromoterPredictionTask

    task = PromoterPredictionTask(data_dir="./data/promoters")
    task.load_data()
    results = task.evaluate(model, tokenizer)
"""

from __future__ import annotations

import abc
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from genova.evaluation.metrics import (
    auroc,
    auprc,
    expected_calibration_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f1_score(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
    """Compute binary F1 score.

    Args:
        predictions: Predicted probabilities, shape ``(N,)``.
        targets: Binary ground truth labels, shape ``(N,)``.
        threshold: Decision threshold.

    Returns:
        F1 score.
    """
    preds_binary = (np.asarray(predictions).ravel() >= threshold).astype(np.int64)
    targets = np.asarray(targets).ravel().astype(np.int64)
    tp = int(((preds_binary == 1) & (targets == 1)).sum())
    fp = int(((preds_binary == 1) & (targets == 0)).sum())
    fn = int(((preds_binary == 0) & (targets == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _mcc(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
    """Compute Matthews Correlation Coefficient.

    Args:
        predictions: Predicted probabilities, shape ``(N,)``.
        targets: Binary ground truth labels, shape ``(N,)``.
        threshold: Decision threshold.

    Returns:
        MCC value in ``[-1, 1]``.
    """
    preds_binary = (np.asarray(predictions).ravel() >= threshold).astype(np.int64)
    targets = np.asarray(targets).ravel().astype(np.int64)
    tp = int(((preds_binary == 1) & (targets == 1)).sum())
    tn = int(((preds_binary == 0) & (targets == 0)).sum())
    fp = int(((preds_binary == 1) & (targets == 0)).sum())
    fn = int(((preds_binary == 0) & (targets == 1)).sum())
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def _compute_all_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute the full standard metric set for a binary classification task.

    Args:
        scores: Predicted probabilities, shape ``(N,)``.
        targets: Binary labels, shape ``(N,)``.
        threshold: Decision threshold for F1 / MCC.
        n_bins: Number of bins for ECE.

    Returns:
        Dictionary with keys ``auroc``, ``auprc``, ``ece``, ``f1``, ``mcc``.
    """
    return {
        "auroc": auroc(scores, targets),
        "auprc": auprc(scores, targets),
        "ece": expected_calibration_error(scores, targets, n_bins=n_bins),
        "f1": _f1_score(scores, targets, threshold=threshold),
        "mcc": _mcc(scores, targets, threshold=threshold),
    }


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkDataset:
    """Container for benchmark sequences and labels.

    Attributes:
        sequences: List of raw DNA strings.
        labels: Integer label array of shape ``(N,)``.
        metadata: Optional per-sample metadata dictionaries.
        split: Dataset split name (``train``, ``val``, ``test``).
    """

    sequences: List[str]
    labels: np.ndarray
    metadata: Optional[List[Dict[str, Any]]] = None
    split: str = "test"

    def __len__(self) -> int:
        return len(self.sequences)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BenchmarkTask(abc.ABC):
    """Base class for all benchmark tasks.

    Subclasses must implement :meth:`load_data` and may override
    :meth:`preprocess` for task-specific preprocessing.

    Args:
        task_name: Human-readable task identifier.
        data_dir: Path to directory containing task data files.
        batch_size: Batch size for model inference.
        device: Torch device string.
    """

    def __init__(
        self,
        task_name: str,
        data_dir: Optional[str] = None,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        self.task_name = task_name
        self.data_dir = Path(data_dir) if data_dir else None
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.dataset: Optional[BenchmarkDataset] = None
        self._metrics: Optional[Dict[str, float]] = None
        logger.info("Initialised benchmark task: {}", self.task_name)

    # -- abstract interface --------------------------------------------------

    @abc.abstractmethod
    def load_data(self) -> BenchmarkDataset:
        """Load and return the benchmark dataset.

        Returns:
            Populated :class:`BenchmarkDataset`.
        """
        ...

    # -- shared logic --------------------------------------------------------

    def preprocess(
        self,
        sequences: List[str],
        tokenizer: Any,
        max_length: int = 512,
    ) -> torch.Tensor:
        """Tokenise a list of DNA sequences.

        Args:
            sequences: Raw DNA strings.
            tokenizer: Tokenizer with an ``encode`` or ``__call__`` method.
            max_length: Maximum token length (sequences are truncated/padded).

        Returns:
            ``(N, max_length)`` tensor of token IDs.
        """
        encoded: List[List[int]] = []
        for seq in sequences:
            if hasattr(tokenizer, "encode"):
                ids = tokenizer.encode(seq)
            elif callable(tokenizer):
                ids = tokenizer(seq)
            else:
                raise TypeError(
                    f"Tokenizer must be callable or have an encode method, "
                    f"got {type(tokenizer).__name__}"
                )
            if isinstance(ids, dict):
                ids = ids.get("input_ids", ids)
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if isinstance(ids[0], list):
                ids = ids[0]
            # Truncate or pad
            ids = ids[:max_length]
            ids = ids + [0] * (max_length - len(ids))
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_length: int = 512,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Run the benchmark evaluation.

        Args:
            model: PyTorch model returning logits or a dict with ``"logits"``
                key.
            tokenizer: Compatible tokenizer.
            max_length: Maximum sequence length in tokens.
            threshold: Decision threshold for binary metrics.

        Returns:
            Dictionary of metric name to value.
        """
        if self.dataset is None:
            self.load_data()
        assert self.dataset is not None

        model.eval()
        model.to(self.device)

        all_scores: List[np.ndarray] = []
        input_ids = self.preprocess(self.dataset.sequences, tokenizer, max_length)

        for start in range(0, len(input_ids), self.batch_size):
            batch = input_ids[start : start + self.batch_size].to(self.device)
            output = model(batch)
            if isinstance(output, dict):
                logits = output.get("logits", output.get("output"))
            elif isinstance(output, (tuple, list)):
                logits = output[0]
            else:
                logits = output
            # Binary: take last dim via sigmoid
            if logits.dim() >= 2 and logits.size(-1) == 2:
                probs = torch.softmax(logits, dim=-1)[:, 1]
            elif logits.dim() >= 2 and logits.size(-1) == 1:
                probs = torch.sigmoid(logits.squeeze(-1))
            else:
                probs = torch.sigmoid(logits)
            all_scores.append(probs.cpu().numpy())

        scores = np.concatenate(all_scores)
        targets = self.dataset.labels

        self._metrics = _compute_all_metrics(scores, targets, threshold=threshold)
        logger.info("Task {} metrics: {}", self.task_name, self._metrics)
        return self._metrics

    def report(self) -> Dict[str, Any]:
        """Return a summary report dictionary.

        Returns:
            Dict with task name, metrics, and dataset size.
        """
        return {
            "task": self.task_name,
            "n_samples": len(self.dataset) if self.dataset else 0,
            "metrics": self._metrics or {},
        }


# ---------------------------------------------------------------------------
# Concrete tasks
# ---------------------------------------------------------------------------

class PromoterPredictionTask(BenchmarkTask):
    """Binary classification of promoter vs non-promoter DNA sequences.

    Expects a TSV/CSV file with columns ``sequence`` and ``label`` (0/1)
    located at ``<data_dir>/promoter_test.tsv``.

    Args:
        data_dir: Directory containing ``promoter_test.tsv``.
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
            task_name="promoter_prediction",
            data_dir=data_dir,
            batch_size=batch_size,
            device=device,
        )

    def load_data(self) -> BenchmarkDataset:
        """Load promoter prediction dataset.

        Reads ``promoter_test.tsv`` from the data directory.  Falls back to
        synthetic data if the file is not found (useful for smoke tests).

        Returns:
            Populated :class:`BenchmarkDataset`.
        """
        sequences: List[str] = []
        labels: List[int] = []

        data_file = self.data_dir / "promoter_test.tsv" if self.data_dir else None

        if data_file and data_file.exists():
            logger.info("Loading promoter data from {}", data_file)
            with open(data_file) as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    sequences.append(row["sequence"])
                    labels.append(int(row["label"]))
        else:
            logger.warning(
                "Promoter data file not found; generating synthetic dataset "
                "for smoke testing."
            )
            rng = np.random.RandomState(42)
            for _ in range(200):
                seq = "".join(rng.choice(list("ACGT"), size=300))
                sequences.append(seq)
                labels.append(int(rng.random() > 0.5))

        self.dataset = BenchmarkDataset(
            sequences=sequences,
            labels=np.array(labels, dtype=np.int64),
            split="test",
        )
        logger.info(
            "Loaded {} promoter samples ({} positive)",
            len(self.dataset),
            int(self.dataset.labels.sum()),
        )
        return self.dataset


class EnhancerClassificationTask(BenchmarkTask):
    """Binary or multi-class enhancer prediction.

    Expects ``enhancer_test.tsv`` with columns ``sequence``, ``label``.
    For multi-class mode, labels may be integers ``{0, 1, 2, ...}``; the
    evaluation reduces to one-vs-rest binary metrics.

    Args:
        data_dir: Directory containing ``enhancer_test.tsv``.
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
            task_name="enhancer_classification",
            data_dir=data_dir,
            batch_size=batch_size,
            device=device,
        )

    def load_data(self) -> BenchmarkDataset:
        """Load enhancer classification dataset.

        Returns:
            Populated :class:`BenchmarkDataset`.
        """
        sequences: List[str] = []
        labels: List[int] = []

        data_file = self.data_dir / "enhancer_test.tsv" if self.data_dir else None

        if data_file and data_file.exists():
            logger.info("Loading enhancer data from {}", data_file)
            with open(data_file) as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    sequences.append(row["sequence"])
                    labels.append(int(row["label"]))
        else:
            logger.warning(
                "Enhancer data file not found; generating synthetic dataset."
            )
            rng = np.random.RandomState(43)
            for _ in range(200):
                seq = "".join(rng.choice(list("ACGT"), size=500))
                sequences.append(seq)
                labels.append(int(rng.random() > 0.5))

        self.dataset = BenchmarkDataset(
            sequences=sequences,
            labels=np.array(labels, dtype=np.int64),
            split="test",
        )
        logger.info(
            "Loaded {} enhancer samples ({} positive)",
            len(self.dataset),
            int(self.dataset.labels.sum()),
        )
        return self.dataset


class VariantEffectTask(BenchmarkTask):
    """ClinVar-style variant pathogenicity prediction.

    Expects ``variant_test.tsv`` with columns ``ref_sequence``,
    ``alt_sequence``, and ``label`` (0 = benign, 1 = pathogenic).
    Alternatively, ``sequence``, ``position``, ``ref``, ``alt``, ``label``.

    Args:
        data_dir: Directory containing ``variant_test.tsv``.
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
            task_name="variant_effect_prediction",
            data_dir=data_dir,
            batch_size=batch_size,
            device=device,
        )
        self._ref_sequences: List[str] = []
        self._alt_sequences: List[str] = []

    def load_data(self) -> BenchmarkDataset:
        """Load variant effect dataset.

        Returns:
            Populated :class:`BenchmarkDataset`.
        """
        ref_seqs: List[str] = []
        alt_seqs: List[str] = []
        labels: List[int] = []
        metadata: List[Dict[str, Any]] = []

        data_file = self.data_dir / "variant_test.tsv" if self.data_dir else None

        if data_file and data_file.exists():
            logger.info("Loading variant data from {}", data_file)
            with open(data_file) as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    if "ref_sequence" in row and "alt_sequence" in row:
                        ref_seqs.append(row["ref_sequence"])
                        alt_seqs.append(row["alt_sequence"])
                    elif "sequence" in row:
                        seq = row["sequence"]
                        pos = int(row["position"])
                        alt_base = row["alt"]
                        ref_seqs.append(seq)
                        alt_seq = seq[:pos] + alt_base + seq[pos + 1 :]
                        alt_seqs.append(alt_seq)
                    labels.append(int(row["label"]))
                    metadata.append({k: v for k, v in row.items() if k != "label"})
        else:
            logger.warning(
                "Variant data file not found; generating synthetic dataset."
            )
            rng = np.random.RandomState(44)
            bases = list("ACGT")
            for _ in range(200):
                seq = "".join(rng.choice(bases, size=200))
                pos = rng.randint(50, 150)
                alt_base = rng.choice([b for b in bases if b != seq[pos]])
                ref_seqs.append(seq)
                alt_seqs.append(seq[:pos] + alt_base + seq[pos + 1 :])
                labels.append(int(rng.random() > 0.5))

        self._ref_sequences = ref_seqs
        self._alt_sequences = alt_seqs

        # Store concatenated ref sequences as the primary sequences list
        self.dataset = BenchmarkDataset(
            sequences=ref_seqs,
            labels=np.array(labels, dtype=np.int64),
            metadata=metadata if metadata else None,
            split="test",
        )
        logger.info(
            "Loaded {} variant samples ({} pathogenic)",
            len(self.dataset),
            int(self.dataset.labels.sum()),
        )
        return self.dataset

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_length: int = 512,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Evaluate variant effect prediction.

        Computes the difference in model output between reference and
        alternate allele sequences as the variant effect score.

        Args:
            model: PyTorch model.
            tokenizer: Compatible tokenizer.
            max_length: Maximum sequence length.
            threshold: Decision threshold.

        Returns:
            Dictionary of metric name to value.
        """
        if self.dataset is None:
            self.load_data()
        assert self.dataset is not None

        model.eval()
        model.to(self.device)

        def _get_scores(sequences: List[str]) -> np.ndarray:
            input_ids = self.preprocess(sequences, tokenizer, max_length)
            all_logits: List[np.ndarray] = []
            for start in range(0, len(input_ids), self.batch_size):
                batch = input_ids[start : start + self.batch_size].to(self.device)
                output = model(batch)
                if isinstance(output, dict):
                    logits = output.get("logits", output.get("output"))
                elif isinstance(output, (tuple, list)):
                    logits = output[0]
                else:
                    logits = output
                if logits.dim() >= 2 and logits.size(-1) == 2:
                    vals = torch.softmax(logits, dim=-1)[:, 1]
                elif logits.dim() >= 2 and logits.size(-1) == 1:
                    vals = logits.squeeze(-1)
                else:
                    vals = logits
                all_logits.append(vals.cpu().numpy())
            return np.concatenate(all_logits)

        ref_scores = _get_scores(self._ref_sequences)
        alt_scores = _get_scores(self._alt_sequences)

        # Variant effect score: absolute difference mapped via sigmoid
        diff = np.abs(alt_scores - ref_scores)
        # Normalise to [0, 1] via sigmoid-like rescaling
        scores = 1.0 / (1.0 + np.exp(-diff))

        targets = self.dataset.labels
        self._metrics = _compute_all_metrics(scores, targets, threshold=threshold)
        logger.info("Task {} metrics: {}", self.task_name, self._metrics)
        return self._metrics


class SpliceSiteTask(BenchmarkTask):
    """Splice donor/acceptor prediction.

    Expects ``splice_test.tsv`` with columns ``sequence`` and ``label``
    (0 = non-splice, 1 = donor, or binary 0/1).

    Args:
        data_dir: Directory containing ``splice_test.tsv``.
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
            task_name="splice_site_prediction",
            data_dir=data_dir,
            batch_size=batch_size,
            device=device,
        )

    def load_data(self) -> BenchmarkDataset:
        """Load splice site dataset.

        Returns:
            Populated :class:`BenchmarkDataset`.
        """
        sequences: List[str] = []
        labels: List[int] = []

        data_file = self.data_dir / "splice_test.tsv" if self.data_dir else None

        if data_file and data_file.exists():
            logger.info("Loading splice site data from {}", data_file)
            with open(data_file) as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    sequences.append(row["sequence"])
                    labels.append(int(row["label"]))
        else:
            logger.warning(
                "Splice data file not found; generating synthetic dataset."
            )
            rng = np.random.RandomState(45)
            donor_motif = "GT"
            acceptor_motif = "AG"
            for i in range(200):
                if i < 100:
                    # Positive: embed splice motif
                    prefix = "".join(rng.choice(list("ACGT"), size=98))
                    suffix = "".join(rng.choice(list("ACGT"), size=100))
                    motif = donor_motif if rng.random() > 0.5 else acceptor_motif
                    seq = prefix + motif + suffix
                    sequences.append(seq)
                    labels.append(1)
                else:
                    seq = "".join(rng.choice(list("ACGT"), size=200))
                    sequences.append(seq)
                    labels.append(0)

        self.dataset = BenchmarkDataset(
            sequences=sequences,
            labels=np.array(labels, dtype=np.int64),
            split="test",
        )
        logger.info(
            "Loaded {} splice site samples ({} positive)",
            len(self.dataset),
            int(self.dataset.labels.sum()),
        )
        return self.dataset


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, type] = {
    "promoter_prediction": PromoterPredictionTask,
    "enhancer_classification": EnhancerClassificationTask,
    "variant_effect_prediction": VariantEffectTask,
    "splice_site_prediction": SpliceSiteTask,
}


def get_task(name: str, **kwargs: Any) -> BenchmarkTask:
    """Retrieve a benchmark task by name.

    Args:
        name: Task identifier (see :data:`TASK_REGISTRY`).
        **kwargs: Forwarded to the task constructor.

    Returns:
        Instantiated :class:`BenchmarkTask`.

    Raises:
        KeyError: If *name* is not found in the registry.
    """
    if name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task {name!r}. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[name](**kwargs)
