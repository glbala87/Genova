"""Curriculum learning for genomic model training.

Implements difficulty scoring, pacing functions, and a curriculum
sampler that orders training data from easy to hard sequences based
on configurable difficulty criteria.

Example::

    from genova.training.curriculum import CurriculumScheduler, CurriculumSampler

    scheduler = CurriculumScheduler()
    difficulties = [scheduler.score_difficulty(seq) for seq in sequences]

    sampler = CurriculumSampler(
        difficulties=difficulties,
        dataset_size=len(sequences),
        pacing="sqrt",
    )
    dataloader = DataLoader(dataset, sampler=sampler)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch, total_epochs=num_epochs)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
from loguru import logger

try:
    import torch
    from torch.utils.data import Sampler

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

    class Sampler:  # type: ignore[no-redef]
        """Stub for when torch is unavailable."""
        pass


# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------


def _gc_content(seq: str) -> float:
    """Compute GC fraction of a DNA sequence.

    Args:
        seq: DNA string.

    Returns:
        GC fraction in ``[0, 1]``.
    """
    if not seq:
        return 0.0
    s = seq.upper()
    gc = s.count("G") + s.count("C")
    total = len(s) - s.count("N")
    return gc / max(total, 1)


def _gc_content_variance(seq: str, window: int = 100) -> float:
    """Compute variance of local GC content across sliding windows.

    Higher variance indicates more heterogeneous GC composition, which
    is generally harder for models to learn.

    Args:
        seq: DNA string.
        window: Sliding window size in bases.

    Returns:
        Variance of GC content across windows.
    """
    if len(seq) < window:
        return 0.0

    gc_values = []
    for i in range(0, len(seq) - window + 1, window // 2):
        chunk = seq[i : i + window]
        gc_values.append(_gc_content(chunk))

    if len(gc_values) < 2:
        return 0.0

    arr = np.array(gc_values, dtype=np.float64)
    return float(arr.var())


def _repeat_fraction(seq: str, min_repeat_len: int = 3) -> float:
    """Estimate the fraction of a sequence covered by simple repeats.

    Detects tandem repeats by checking for repeated substrings.

    Args:
        seq: DNA string.
        min_repeat_len: Minimum repeat unit length.

    Returns:
        Fraction of sequence covered by repeats in ``[0, 1]``.
    """
    if len(seq) < min_repeat_len * 2:
        return 0.0

    s = seq.upper()
    repeat_mask = [False] * len(s)

    for unit_len in range(min_repeat_len, min(8, len(s) // 2 + 1)):
        for start in range(len(s) - unit_len):
            unit = s[start : start + unit_len]
            # Check if the unit repeats immediately after
            end = start + unit_len
            if end + unit_len <= len(s) and s[end : end + unit_len] == unit:
                # Mark both copies
                for j in range(start, min(end + unit_len, len(s))):
                    repeat_mask[j] = True

    return sum(repeat_mask) / len(s)


def _n_content(seq: str) -> float:
    """Compute fraction of N (ambiguous) bases.

    Args:
        seq: DNA string.

    Returns:
        N fraction in ``[0, 1]``.
    """
    if not seq:
        return 0.0
    return seq.upper().count("N") / len(seq)


# ---------------------------------------------------------------------------
# Pacing functions
# ---------------------------------------------------------------------------


def _linear_pacing(progress: float) -> float:
    """Linear pacing: competence grows linearly with training progress.

    Args:
        progress: Training progress in ``[0, 1]``.

    Returns:
        Competence level in ``[0, 1]``.
    """
    return min(1.0, max(0.0, progress))


def _sqrt_pacing(progress: float) -> float:
    """Square-root pacing: competence grows faster initially.

    Args:
        progress: Training progress in ``[0, 1]``.

    Returns:
        Competence level in ``[0, 1]``.
    """
    return min(1.0, math.sqrt(max(0.0, progress)))


def _exponential_pacing(progress: float, base: float = 10.0) -> float:
    """Exponential pacing: competence grows slowly then rapidly.

    Args:
        progress: Training progress in ``[0, 1]``.
        base: Base for the exponential.

    Returns:
        Competence level in ``[0, 1]``.
    """
    if progress >= 1.0:
        return 1.0
    return min(1.0, (base ** progress - 1.0) / (base - 1.0))


_PACING_FUNCTIONS = {
    "linear": _linear_pacing,
    "sqrt": _sqrt_pacing,
    "exponential": _exponential_pacing,
}


# ---------------------------------------------------------------------------
# Curriculum Scheduler
# ---------------------------------------------------------------------------


class CurriculumScheduler:
    """Difficulty scoring and competence scheduling for curriculum learning.

    Scores each sequence on multiple difficulty criteria and manages a
    competence level that determines what fraction of the difficulty
    range is accessible at any training stage.

    The competence-based approach follows Platanios et al. (2019):
    *Competence-based Curriculum Learning for Neural Machine Translation*.

    Args:
        gc_weight: Weight for GC content variance in difficulty score.
        repeat_weight: Weight for repeat fraction in difficulty score.
        n_weight: Weight for N content in difficulty score.
        gc_window: Window size for local GC variance calculation.
        pacing: Pacing function name (``"linear"``, ``"sqrt"``,
            ``"exponential"``).
        initial_competence: Starting competence level in ``[0, 1]``.
            Determines the fraction of easiest samples available at
            the start of training.
    """

    def __init__(
        self,
        gc_weight: float = 1.0,
        repeat_weight: float = 1.0,
        n_weight: float = 1.5,
        gc_window: int = 100,
        pacing: str = "sqrt",
        initial_competence: float = 0.1,
    ) -> None:
        self.gc_weight = gc_weight
        self.repeat_weight = repeat_weight
        self.n_weight = n_weight
        self.gc_window = gc_window
        self.initial_competence = max(0.01, min(1.0, initial_competence))

        if pacing not in _PACING_FUNCTIONS:
            raise ValueError(
                f"Unknown pacing function {pacing!r}. "
                f"Choose from {list(_PACING_FUNCTIONS.keys())}."
            )
        self.pacing_name = pacing
        self._pacing_fn = _PACING_FUNCTIONS[pacing]

        # Current competence level
        self._competence: float = self.initial_competence

        logger.info(
            "CurriculumScheduler: pacing={}, initial_competence={:.2f}, "
            "weights=(gc={}, repeat={}, n={})",
            pacing,
            self.initial_competence,
            gc_weight,
            repeat_weight,
            n_weight,
        )

    @property
    def competence(self) -> float:
        """Current competence level in ``[0, 1]``."""
        return self._competence

    def score_difficulty(self, sequence: str) -> float:
        """Compute a difficulty score for a DNA sequence.

        Higher scores indicate harder sequences.  The score is a
        weighted combination of:

        - **GC content variance**: measures compositional heterogeneity.
        - **Repeat fraction**: measures simple tandem repeat content.
        - **N content**: measures ambiguous base content.

        All components are in ``[0, 1]`` and the result is normalized.

        Args:
            sequence: DNA string.

        Returns:
            Difficulty score in ``[0, 1]``.
        """
        gc_var = _gc_content_variance(sequence, window=self.gc_window)
        # Normalize GC variance (typical range 0-0.05, cap at 0.1)
        gc_var_norm = min(gc_var / 0.1, 1.0)

        repeat_frac = _repeat_fraction(sequence)
        n_frac = _n_content(sequence)

        total_weight = self.gc_weight + self.repeat_weight + self.n_weight
        if total_weight == 0:
            return 0.0

        difficulty = (
            self.gc_weight * gc_var_norm
            + self.repeat_weight * repeat_frac
            + self.n_weight * n_frac
        ) / total_weight

        return float(min(1.0, max(0.0, difficulty)))

    def score_batch(self, sequences: Sequence[str]) -> np.ndarray:
        """Score difficulty for a batch of sequences.

        Args:
            sequences: List of DNA strings.

        Returns:
            Array of difficulty scores, shape ``(N,)``.
        """
        return np.array(
            [self.score_difficulty(seq) for seq in sequences],
            dtype=np.float64,
        )

    def update_competence(
        self,
        epoch: int,
        total_epochs: int,
    ) -> float:
        """Update the competence level based on training progress.

        The competence grows from ``initial_competence`` to ``1.0``
        over the course of training, following the configured pacing
        function.

        Args:
            epoch: Current epoch (0-based).
            total_epochs: Total number of training epochs.

        Returns:
            Updated competence level.
        """
        if total_epochs <= 1:
            self._competence = 1.0
            return self._competence

        progress = epoch / max(1, total_epochs - 1)
        raw = self._pacing_fn(progress)

        # Scale from initial_competence to 1.0
        self._competence = self.initial_competence + (
            1.0 - self.initial_competence
        ) * raw

        self._competence = min(1.0, max(self.initial_competence, self._competence))

        logger.debug(
            "Competence updated: epoch={}/{}, progress={:.3f}, competence={:.3f}",
            epoch,
            total_epochs,
            progress,
            self._competence,
        )
        return self._competence

    def get_sampler(
        self,
        difficulties: np.ndarray,
        dataset_size: int,
        epoch: int = 0,
        total_epochs: int = 1,
    ) -> "CurriculumSampler":
        """Create a :class:`CurriculumSampler` for the current epoch.

        Convenience method that updates competence and constructs a
        sampler.

        Args:
            difficulties: Pre-computed difficulty scores, shape ``(N,)``.
            dataset_size: Total dataset size.
            epoch: Current epoch.
            total_epochs: Total epochs.

        Returns:
            A configured :class:`CurriculumSampler`.
        """
        self.update_competence(epoch, total_epochs)
        return CurriculumSampler(
            difficulties=difficulties,
            dataset_size=dataset_size,
            competence=self._competence,
        )


# ---------------------------------------------------------------------------
# Curriculum Sampler (PyTorch Sampler)
# ---------------------------------------------------------------------------


class CurriculumSampler(Sampler):
    """PyTorch-compatible sampler that orders data by difficulty.

    At each epoch, only the fraction of easiest samples determined by
    the current competence level is available.  Within the accessible
    range, samples are shuffled for stochasticity.

    Compatible with :class:`torch.utils.data.DataLoader`.

    Args:
        difficulties: Pre-computed difficulty scores, shape ``(N,)``.
        dataset_size: Total dataset size (must match ``len(difficulties)``).
        competence: Current competence level in ``[0, 1]``.
        seed: Random seed for shuffling.
    """

    def __init__(
        self,
        difficulties: np.ndarray,
        dataset_size: int,
        competence: float = 1.0,
        seed: int = 42,
    ) -> None:
        self._difficulties = np.asarray(difficulties, dtype=np.float64)
        self._dataset_size = dataset_size
        self._competence = min(1.0, max(0.0, competence))
        self._seed = seed
        self._epoch = 0

        # Sort indices by difficulty (ascending = easy first)
        self._sorted_indices = np.argsort(self._difficulties)

    @property
    def competence(self) -> float:
        """Current competence level."""
        return self._competence

    @competence.setter
    def competence(self, value: float) -> None:
        self._competence = min(1.0, max(0.0, value))

    def set_epoch(
        self,
        epoch: int,
        total_epochs: Optional[int] = None,
        pacing: str = "sqrt",
    ) -> None:
        """Update the epoch and optionally recompute competence.

        Args:
            epoch: Current epoch (0-based).
            total_epochs: If provided, recompute competence using the
                specified pacing function.
            pacing: Pacing function name.
        """
        self._epoch = epoch
        if total_epochs is not None and total_epochs > 1:
            progress = epoch / max(1, total_epochs - 1)
            pacing_fn = _PACING_FUNCTIONS.get(pacing, _sqrt_pacing)
            self._competence = min(1.0, pacing_fn(progress))

    def __len__(self) -> int:
        """Return the number of accessible samples at current competence."""
        accessible = max(1, int(self._competence * self._dataset_size))
        return accessible

    def __iter__(self) -> Iterator[int]:
        """Yield sample indices ordered by difficulty, shuffled within
        the accessible range.

        Returns:
            Iterator of dataset indices.
        """
        accessible = max(1, int(self._competence * self._dataset_size))
        accessible_indices = self._sorted_indices[:accessible].copy()

        # Shuffle within accessible range
        rng = np.random.RandomState(self._seed + self._epoch)
        rng.shuffle(accessible_indices)

        return iter(accessible_indices.tolist())
