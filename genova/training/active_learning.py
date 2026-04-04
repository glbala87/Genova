"""Active learning strategies for Genova.

Provides uncertainty-based, diversity-based, committee-based, and
expected-model-change sampling strategies for selecting the most
informative samples from an unlabelled pool.

Example::

    from genova.training.active_learning import ActiveLearner

    learner = ActiveLearner(model, strategy="uncertainty")
    indices = learner.select_samples(pool_data, budget=100)
    learner.update(new_labels)
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------


class SamplingStrategy(str, Enum):
    """Supported active learning sampling strategies."""

    UNCERTAINTY = "uncertainty"
    DIVERSITY = "diversity"
    COMMITTEE = "committee"
    RANDOM = "random"


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class ActiveLearningState:
    """Internal state for an active learning loop.

    Attributes:
        labelled_indices: Indices of samples that have been labelled.
        unlabelled_indices: Indices of remaining unlabelled samples.
        round_number: Current active learning round.
        history: Per-round metadata (selected indices, metrics, etc.).
    """

    labelled_indices: List[int] = field(default_factory=list)
    unlabelled_indices: List[int] = field(default_factory=list)
    round_number: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sampling functions
# ---------------------------------------------------------------------------


def _uncertainty_sampling(
    model: nn.Module,
    pool: Tensor,
    budget: int,
    device: torch.device,
    batch_size: int = 64,
) -> List[int]:
    """Select samples with highest predictive uncertainty (entropy).

    Args:
        model: Trained model returning logits.
        pool: Input tensor ``(N, L)`` of unlabelled samples.
        budget: Number of samples to select.
        device: Torch device.
        batch_size: Batch size for inference.

    Returns:
        List of selected indices (into *pool*).
    """
    model.eval()
    entropies: List[float] = []

    with torch.no_grad():
        for start in range(0, len(pool), batch_size):
            batch = pool[start : start + batch_size].to(device)
            output = model(batch)
            if isinstance(output, dict):
                logits = output.get("logits", output.get("output", output.get("last_hidden_state")))
            elif isinstance(output, (list, tuple)):
                logits = output[0]
            else:
                logits = output

            # Handle different output shapes
            if logits.dim() == 3:
                # Per-token predictions: average entropy across positions
                probs = torch.softmax(logits, dim=-1)
                token_entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
                entropy = token_entropy.mean(dim=-1)
            elif logits.dim() == 2 and logits.size(-1) > 1:
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
            else:
                # Binary: use sigmoid entropy
                p = torch.sigmoid(logits.squeeze(-1))
                entropy = -(
                    p * (p + 1e-10).log() + (1 - p) * (1 - p + 1e-10).log()
                )

            entropies.extend(entropy.cpu().tolist())

    # Select top-k by entropy
    indices = np.argsort(entropies)[::-1][:budget].tolist()
    return indices


def _diversity_sampling(
    model: nn.Module,
    pool: Tensor,
    budget: int,
    device: torch.device,
    batch_size: int = 64,
) -> List[int]:
    """Select diverse samples using k-center greedy (coreset).

    Greedily selects the sample farthest from the already-selected set
    in embedding space.

    Args:
        model: Model used to extract embeddings.
        pool: Input tensor ``(N, L)``.
        budget: Number of samples to select.
        device: Torch device.
        batch_size: Batch size for embedding extraction.

    Returns:
        List of selected indices.
    """
    model.eval()
    embeddings_list: List[Tensor] = []

    with torch.no_grad():
        for start in range(0, len(pool), batch_size):
            batch = pool[start : start + batch_size].to(device)
            output = model(batch)
            if isinstance(output, dict):
                hidden = output.get("last_hidden_state", output.get("hidden_states"))
            elif isinstance(output, Tensor):
                hidden = output
            else:
                hidden = getattr(output, "last_hidden_state", output)

            if isinstance(hidden, (list, tuple)):
                hidden = hidden[-1]

            # Mean pool to get sequence-level embeddings
            if hidden.dim() == 3:
                emb = hidden.mean(dim=1)
            else:
                emb = hidden
            embeddings_list.append(emb.cpu())

    embeddings = torch.cat(embeddings_list, dim=0)  # (N, D)
    N = embeddings.size(0)
    budget = min(budget, N)

    # K-center greedy
    selected: List[int] = []
    # Start with a random seed
    rng = np.random.RandomState(42)
    first = int(rng.randint(0, N))
    selected.append(first)

    # Min distance from each point to the selected set
    min_dist = torch.full((N,), float("inf"))
    dist_to_first = torch.cdist(
        embeddings, embeddings[first : first + 1]
    ).squeeze(-1)
    min_dist = torch.minimum(min_dist, dist_to_first)

    for _ in range(budget - 1):
        # Select the point with the largest minimum distance
        idx = int(torch.argmax(min_dist).item())
        selected.append(idx)
        dist_to_new = torch.cdist(
            embeddings, embeddings[idx : idx + 1]
        ).squeeze(-1)
        min_dist = torch.minimum(min_dist, dist_to_new)

    return selected


def _committee_sampling(
    models: List[nn.Module],
    pool: Tensor,
    budget: int,
    device: torch.device,
    batch_size: int = 64,
) -> List[int]:
    """Select samples with highest disagreement across a committee.

    Disagreement is measured as the variance of predicted probabilities
    across committee members.

    Args:
        models: List of committee models.
        pool: Input tensor ``(N, L)``.
        budget: Number of samples to select.
        device: Torch device.
        batch_size: Batch size.

    Returns:
        List of selected indices.
    """
    if len(models) < 2:
        raise ValueError("Committee sampling requires at least 2 models.")

    all_predictions: List[np.ndarray] = []
    for m in models:
        m.eval()
        model_preds: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(pool), batch_size):
                batch = pool[start : start + batch_size].to(device)
                output = m(batch)
                if isinstance(output, dict):
                    logits = output.get("logits", output.get("output", output.get("last_hidden_state")))
                elif isinstance(output, (list, tuple)):
                    logits = output[0]
                else:
                    logits = output

                if logits.dim() == 3:
                    probs = torch.softmax(logits, dim=-1).mean(dim=1)
                elif logits.dim() == 2 and logits.size(-1) > 1:
                    probs = torch.softmax(logits, dim=-1)
                else:
                    p = torch.sigmoid(logits.squeeze(-1))
                    probs = torch.stack([1 - p, p], dim=-1)

                model_preds.append(probs.cpu().numpy())
        all_predictions.append(np.concatenate(model_preds, axis=0))

    # Stack: (num_models, N, C)
    stacked = np.stack(all_predictions, axis=0)
    # Variance across models, summed across classes
    disagreement = stacked.var(axis=0).sum(axis=-1)  # (N,)

    indices = np.argsort(disagreement)[::-1][:budget].tolist()
    return indices


def _random_sampling(pool_size: int, budget: int) -> List[int]:
    """Select random samples from the pool.

    Args:
        pool_size: Total number of samples in the pool.
        budget: Number of samples to select.

    Returns:
        List of selected indices.
    """
    rng = np.random.RandomState(None)
    return rng.choice(pool_size, size=min(budget, pool_size), replace=False).tolist()


# ---------------------------------------------------------------------------
# ActiveLearner
# ---------------------------------------------------------------------------


class ActiveLearner:
    """Active learning loop manager.

    Supports multiple acquisition strategies and manages the labelled /
    unlabelled pool split across rounds.

    Args:
        model: PyTorch model for predictions and embedding extraction.
        strategy: Sampling strategy name (``"uncertainty"``,
            ``"diversity"``, ``"committee"``, ``"random"``).
        committee_models: List of models for committee-based sampling.
            Required only when ``strategy="committee"``.
        device: Torch device for inference.
        batch_size: Batch size for model inference during selection.
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: Union[str, SamplingStrategy] = "uncertainty",
        committee_models: Optional[List[nn.Module]] = None,
        device: Union[str, torch.device] = "cpu",
        batch_size: int = 64,
    ) -> None:
        self.model = model
        self.strategy = SamplingStrategy(strategy)
        self.committee_models = committee_models or []
        self.device = torch.device(device)
        self.batch_size = batch_size

        self._state = ActiveLearningState()
        self._labels: Dict[int, Any] = {}

    @property
    def state(self) -> ActiveLearningState:
        """Current active learning state."""
        return self._state

    def initialise_pool(self, pool_size: int, initial_labels: Optional[Dict[int, Any]] = None) -> None:
        """Set up the initial labelled / unlabelled split.

        Args:
            pool_size: Total number of samples.
            initial_labels: Dict mapping sample index to its label.
                These indices are moved to the labelled set.
        """
        all_indices = list(range(pool_size))
        if initial_labels:
            self._labels.update(initial_labels)
            labelled = list(initial_labels.keys())
            unlabelled = [i for i in all_indices if i not in initial_labels]
        else:
            labelled = []
            unlabelled = all_indices

        self._state.labelled_indices = labelled
        self._state.unlabelled_indices = unlabelled
        self._state.round_number = 0

    def select_samples(
        self,
        pool: Tensor,
        budget: int,
        strategy: Optional[str] = None,
    ) -> List[int]:
        """Select the most informative samples from the pool.

        Args:
            pool: Full pool tensor ``(N, L)``.  If the learner has an
                active unlabelled set, only those indices are considered.
            budget: Number of samples to select.
            strategy: Override the default strategy for this round.

        Returns:
            List of selected sample indices (into the original pool).
        """
        strat = SamplingStrategy(strategy) if strategy else self.strategy

        # If we have a tracked unlabelled set, restrict to those indices
        if self._state.unlabelled_indices:
            unlabelled_indices = self._state.unlabelled_indices
            sub_pool = pool[unlabelled_indices]
        else:
            unlabelled_indices = list(range(len(pool)))
            sub_pool = pool

        budget = min(budget, len(unlabelled_indices))

        if strat == SamplingStrategy.UNCERTAINTY:
            local_indices = _uncertainty_sampling(
                self.model, sub_pool, budget, self.device, self.batch_size
            )
        elif strat == SamplingStrategy.DIVERSITY:
            local_indices = _diversity_sampling(
                self.model, sub_pool, budget, self.device, self.batch_size
            )
        elif strat == SamplingStrategy.COMMITTEE:
            if not self.committee_models:
                raise ValueError(
                    "Committee sampling requires committee_models to be set."
                )
            local_indices = _committee_sampling(
                self.committee_models, sub_pool, budget, self.device, self.batch_size
            )
        elif strat == SamplingStrategy.RANDOM:
            local_indices = _random_sampling(len(sub_pool), budget)
        else:
            raise ValueError(f"Unknown strategy: {strat}")

        # Map local indices back to global pool indices
        global_indices = [unlabelled_indices[i] for i in local_indices]

        # Record in history
        self._state.history.append({
            "round": self._state.round_number,
            "strategy": strat.value,
            "budget": budget,
            "selected_indices": global_indices,
        })

        return global_indices

    def update(self, new_labels: Dict[int, Any]) -> None:
        """Update the learner with newly obtained labels.

        Moves labelled samples from the unlabelled to the labelled set.

        Args:
            new_labels: Dict mapping sample index to its label.
        """
        self._labels.update(new_labels)
        newly_labelled = set(new_labels.keys())
        self._state.labelled_indices.extend(
            [i for i in newly_labelled if i not in set(self._state.labelled_indices)]
        )
        self._state.unlabelled_indices = [
            i for i in self._state.unlabelled_indices if i not in newly_labelled
        ]
        self._state.round_number += 1

    def train_step(
        self,
        pool: Tensor,
        labels: Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[Callable[..., Tensor]] = None,
    ) -> float:
        """Run one training step on the currently labelled data.

        Args:
            pool: Full pool tensor ``(N, L)``.
            labels: Full label tensor ``(N, ...)``.
            optimizer: PyTorch optimizer.
            loss_fn: Loss function.  Defaults to cross-entropy.

        Returns:
            Training loss value.
        """
        if not self._state.labelled_indices:
            return 0.0

        self.model.train()
        indices = self._state.labelled_indices
        batch_x = pool[indices].to(self.device)
        batch_y = labels[indices].to(self.device)

        optimizer.zero_grad()
        output = self.model(batch_x)
        if isinstance(output, dict):
            logits = output.get("logits", output.get("output"))
        elif isinstance(output, (list, tuple)):
            logits = output[0]
        else:
            logits = output

        if loss_fn is not None:
            loss = loss_fn(logits, batch_y)
        else:
            if logits.dim() == 3:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), batch_y.view(-1), ignore_index=-100
                )
            elif logits.dim() == 2 and logits.size(-1) > 1:
                loss = F.cross_entropy(logits, batch_y)
            else:
                loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), batch_y.float()
                )

        loss.backward()
        optimizer.step()

        return float(loss.item())

    def get_labels(self) -> Dict[int, Any]:
        """Return all collected labels.

        Returns:
            Dictionary mapping sample index to label.
        """
        return dict(self._labels)
