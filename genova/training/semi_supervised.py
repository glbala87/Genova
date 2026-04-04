"""Semi-supervised learning strategies for Genova.

Implements pseudo-labeling, self-training, consistency regularisation,
and FixMatch-style training for leveraging unlabelled genomic data.

Example::

    from genova.training.semi_supervised import SemiSupervisedTrainer

    trainer = SemiSupervisedTrainer(model, optimizer)
    trainer.train(labelled_loader, unlabelled_loader, epochs=10)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SemiSupervisedMetrics:
    """Metrics collected during semi-supervised training.

    Attributes:
        epoch: Epoch number.
        supervised_loss: Loss on labelled data.
        unsupervised_loss: Loss on unlabelled data (pseudo-label or
            consistency).
        total_loss: Combined loss.
        num_pseudo_labels: Number of pseudo-labels generated.
        pseudo_label_accuracy: Accuracy of pseudo-labels vs true labels
            (if available, else ``-1``).
    """

    epoch: int = 0
    supervised_loss: float = 0.0
    unsupervised_loss: float = 0.0
    total_loss: float = 0.0
    num_pseudo_labels: int = 0
    pseudo_label_accuracy: float = -1.0


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------


def _default_weak_augmentation(x: Tensor) -> Tensor:
    """Apply weak augmentation (identity + small noise).

    Args:
        x: Input tensor ``(B, L)`` of token IDs.

    Returns:
        Augmented tensor of same shape.
    """
    # For token IDs, weak augmentation is identity (no change)
    return x.clone()


def _default_strong_augmentation(x: Tensor, mask_prob: float = 0.15, mask_token: int = 4) -> Tensor:
    """Apply strong augmentation via random masking.

    Randomly replaces tokens with a mask token.

    Args:
        x: Input tensor ``(B, L)`` of token IDs.
        mask_prob: Probability of masking each token.
        mask_token: Token ID to use for masking.

    Returns:
        Augmented tensor of same shape.
    """
    x_aug = x.clone()
    mask = torch.rand_like(x.float()) < mask_prob
    x_aug[mask] = mask_token
    return x_aug


def _mixup(
    x1: Tensor,
    y1: Tensor,
    x2: Tensor,
    y2: Tensor,
    alpha: float = 0.75,
) -> Tuple[Tensor, Tensor]:
    """Apply MixUp interpolation between two sample sets.

    For discrete inputs (token IDs), we interpolate in embedding space
    when possible.  This function operates on continuous representations.

    Args:
        x1: First input batch ``(B, ...)``.
        y1: First label batch ``(B, C)`` (one-hot or soft labels).
        x2: Second input batch ``(B, ...)``.
        y2: Second label batch ``(B, C)``.
        alpha: Beta distribution parameter.  ``alpha=1`` gives uniform.

    Returns:
        Tuple of (mixed_x, mixed_y).
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam = max(lam, 1 - lam)  # ensure lambda >= 0.5
    mixed_x = lam * x1.float() + (1 - lam) * x2.float()
    mixed_y = lam * y1.float() + (1 - lam) * y2.float()
    return mixed_x, mixed_y


# ---------------------------------------------------------------------------
# Pseudo-labeling
# ---------------------------------------------------------------------------


@torch.no_grad()
def pseudo_label(
    model: nn.Module,
    unlabelled_data: Tensor,
    threshold: float = 0.95,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 64,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate pseudo-labels for unlabelled data.

    Runs the model on unlabelled data and retains predictions that
    exceed the confidence threshold.

    Args:
        model: Trained model returning logits.
        unlabelled_data: Input tensor ``(N, L)``.
        threshold: Minimum confidence to accept a pseudo-label.
        device: Torch device.
        batch_size: Batch size for inference.

    Returns:
        Tuple of:
        - ``selected_data``: Inputs that passed the threshold ``(M, L)``.
        - ``pseudo_labels``: Predicted labels ``(M,)``.
        - ``confidences``: Confidence scores ``(M,)``.
    """
    device = torch.device(device)
    model.eval()
    model.to(device)

    all_probs: List[Tensor] = []
    all_preds: List[Tensor] = []

    for start in range(0, len(unlabelled_data), batch_size):
        batch = unlabelled_data[start : start + batch_size].to(device)
        output = model(batch)
        if isinstance(output, dict):
            logits = output.get("logits", output.get("output"))
        elif isinstance(output, (list, tuple)):
            logits = output[0]
        else:
            logits = output

        if logits.dim() == 3:
            # Per-token: take mean logits across positions
            logits = logits.mean(dim=1)

        if logits.dim() == 2 and logits.size(-1) > 1:
            probs = torch.softmax(logits, dim=-1)
            max_probs, preds = probs.max(dim=-1)
        else:
            p = torch.sigmoid(logits.squeeze(-1))
            max_probs = torch.where(p > 0.5, p, 1 - p)
            preds = (p > 0.5).long()

        all_probs.append(max_probs.cpu())
        all_preds.append(preds.cpu())

    confidences = torch.cat(all_probs, dim=0)
    predictions = torch.cat(all_preds, dim=0)

    mask = confidences >= threshold
    selected_data = unlabelled_data[mask]
    pseudo_labels_out = predictions[mask]
    selected_confidences = confidences[mask]

    return selected_data, pseudo_labels_out, selected_confidences


# ---------------------------------------------------------------------------
# Consistency loss
# ---------------------------------------------------------------------------


def consistency_loss(
    model: nn.Module,
    x_aug1: Tensor,
    x_aug2: Tensor,
    loss_type: str = "mse",
) -> Tensor:
    """Compute consistency regularisation loss between two augmented views.

    Both views should be of the same underlying sample.  The loss
    encourages the model to produce the same predictions for both.

    Args:
        model: Model returning logits.
        x_aug1: First augmented view ``(B, L)``.
        x_aug2: Second augmented view ``(B, L)``.
        loss_type: ``"mse"`` or ``"kl"`` divergence.

    Returns:
        Scalar loss tensor.
    """
    out1 = model(x_aug1)
    out2 = model(x_aug2)

    def _extract_probs(output: Any) -> Tensor:
        if isinstance(output, dict):
            logits = output.get("logits", output.get("output"))
        elif isinstance(output, (list, tuple)):
            logits = output[0]
        else:
            logits = output

        if logits.dim() == 3:
            logits = logits.mean(dim=1)

        if logits.dim() == 2 and logits.size(-1) > 1:
            return torch.softmax(logits, dim=-1)
        return torch.sigmoid(logits.squeeze(-1))

    p1 = _extract_probs(out1)
    p2 = _extract_probs(out2)

    if loss_type == "kl":
        log_p1 = (p1 + 1e-10).log()
        log_p2 = (p2 + 1e-10).log()
        loss = 0.5 * (
            F.kl_div(log_p1, p2, reduction="batchmean")
            + F.kl_div(log_p2, p1, reduction="batchmean")
        )
    else:
        loss = F.mse_loss(p1, p2)

    return loss


# ---------------------------------------------------------------------------
# SemiSupervisedTrainer
# ---------------------------------------------------------------------------


class SemiSupervisedTrainer:
    """Semi-supervised training manager.

    Supports multiple training paradigms:
    - Pseudo-labeling with confidence thresholding
    - Self-training (iterative pseudo-labeling)
    - Consistency regularisation
    - FixMatch-style (weak + strong augmentation consistency)
    - MixMatch-style (augmentation + pseudo-labels + mixup)

    Args:
        model: PyTorch model to train.
        optimizer: Optimizer.  If ``None``, creates Adam with lr=1e-4.
        device: Torch device.
        pseudo_label_threshold: Confidence threshold for pseudo-labels.
        consistency_weight: Weight for the consistency / unsupervised loss.
        weak_augmentation: Callable for weak augmentation.
        strong_augmentation: Callable for strong augmentation.
        mask_token: Token ID used for masking in strong augmentation.
        ema_decay: Exponential moving average decay for teacher model
            (used in self-training).  Set to 0 to disable EMA.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Union[str, torch.device] = "cpu",
        pseudo_label_threshold: float = 0.95,
        consistency_weight: float = 1.0,
        weak_augmentation: Optional[Callable[[Tensor], Tensor]] = None,
        strong_augmentation: Optional[Callable[[Tensor], Tensor]] = None,
        mask_token: int = 4,
        ema_decay: float = 0.999,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        else:
            self.optimizer = optimizer

        self.pseudo_label_threshold = pseudo_label_threshold
        self.consistency_weight = consistency_weight
        self.mask_token = mask_token
        self.ema_decay = ema_decay

        self.weak_aug = weak_augmentation or _default_weak_augmentation
        self.strong_aug = strong_augmentation or (
            lambda x: _default_strong_augmentation(x, mask_token=mask_token)
        )

        # EMA teacher model
        self._teacher: Optional[nn.Module] = None
        if ema_decay > 0:
            self._teacher = copy.deepcopy(model)
            self._teacher.to(self.device)
            self._teacher.eval()
            for p in self._teacher.parameters():
                p.requires_grad_(False)

        self._history: List[SemiSupervisedMetrics] = []

    @property
    def history(self) -> List[SemiSupervisedMetrics]:
        """Training history across epochs."""
        return self._history

    def _update_teacher(self) -> None:
        """Update the EMA teacher model."""
        if self._teacher is None:
            return
        for t_param, s_param in zip(
            self._teacher.parameters(), self.model.parameters()
        ):
            t_param.data.mul_(self.ema_decay).add_(
                s_param.data, alpha=1 - self.ema_decay
            )

    def _supervised_loss(self, batch_x: Tensor, batch_y: Tensor) -> Tensor:
        """Compute supervised loss on labelled data.

        Args:
            batch_x: ``(B, L)``
            batch_y: ``(B,)`` or ``(B, C)``

        Returns:
            Scalar loss.
        """
        output = self.model(batch_x)
        if isinstance(output, dict):
            logits = output.get("logits", output.get("output"))
        elif isinstance(output, (list, tuple)):
            logits = output[0]
        else:
            logits = output

        if logits.dim() == 3:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), batch_y.view(-1), ignore_index=-100
            )
        elif logits.dim() == 2 and logits.size(-1) > 1:
            return F.cross_entropy(logits, batch_y)
        else:
            return F.binary_cross_entropy_with_logits(
                logits.squeeze(-1), batch_y.float()
            )

    def train(
        self,
        labelled_data: Union[DataLoader, Tuple[Tensor, Tensor]],
        unlabelled_data: Union[DataLoader, Tensor],
        model: Optional[nn.Module] = None,
        epochs: int = 10,
        method: str = "fixmatch",
        log_interval: int = 1,
    ) -> List[SemiSupervisedMetrics]:
        """Run semi-supervised training.

        Args:
            labelled_data: Either a DataLoader yielding ``(x, y)`` tuples
                or a tuple ``(inputs, labels)``.
            unlabelled_data: Either a DataLoader yielding input tensors
                or a single tensor ``(N, L)``.
            model: Optional model override (uses ``self.model`` if None).
            epochs: Number of training epochs.
            method: Training method -- ``"pseudo_label"``,
                ``"consistency"``, ``"fixmatch"``, or ``"mixmatch"``.
            log_interval: Log metrics every N epochs.

        Returns:
            List of :class:`SemiSupervisedMetrics` per epoch.
        """
        if model is not None:
            self.model = model
            self.model.to(self.device)

        # Normalise data inputs
        if isinstance(labelled_data, tuple):
            lab_x, lab_y = labelled_data
            lab_loader = DataLoader(
                TensorDataset(lab_x, lab_y), batch_size=32, shuffle=True
            )
        else:
            lab_loader = labelled_data

        if isinstance(unlabelled_data, Tensor):
            unlab_loader = DataLoader(
                TensorDataset(unlabelled_data), batch_size=32, shuffle=True
            )
        else:
            unlab_loader = unlabelled_data

        metrics_list: List[SemiSupervisedMetrics] = []

        for epoch in range(epochs):
            self.model.train()
            epoch_sup_loss = 0.0
            epoch_unsup_loss = 0.0
            epoch_pseudo_count = 0
            n_batches = 0

            unlab_iter = iter(unlab_loader)

            for batch in lab_loader:
                if isinstance(batch, (list, tuple)):
                    batch_x, batch_y = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue

                # Get unlabelled batch
                try:
                    unlab_batch = next(unlab_iter)
                except StopIteration:
                    unlab_iter = iter(unlab_loader)
                    unlab_batch = next(unlab_iter)

                if isinstance(unlab_batch, (list, tuple)):
                    unlab_x = unlab_batch[0].to(self.device)
                else:
                    unlab_x = unlab_batch.to(self.device)

                # Supervised loss
                sup_loss = self._supervised_loss(batch_x, batch_y)

                # Unsupervised loss depends on method
                if method == "pseudo_label":
                    unsup_loss, n_pseudo = self._pseudo_label_step(unlab_x)
                elif method == "consistency":
                    unsup_loss = self._consistency_step(unlab_x)
                    n_pseudo = 0
                elif method == "fixmatch":
                    unsup_loss, n_pseudo = self._fixmatch_step(unlab_x)
                elif method == "mixmatch":
                    unsup_loss, n_pseudo = self._mixmatch_step(
                        batch_x, batch_y, unlab_x
                    )
                else:
                    raise ValueError(f"Unknown method: {method!r}")

                total_loss = sup_loss + self.consistency_weight * unsup_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self._update_teacher()

                epoch_sup_loss += sup_loss.item()
                epoch_unsup_loss += unsup_loss.item()
                epoch_pseudo_count += n_pseudo
                n_batches += 1

            n_batches = max(n_batches, 1)
            metrics = SemiSupervisedMetrics(
                epoch=epoch,
                supervised_loss=epoch_sup_loss / n_batches,
                unsupervised_loss=epoch_unsup_loss / n_batches,
                total_loss=(epoch_sup_loss + epoch_unsup_loss) / n_batches,
                num_pseudo_labels=epoch_pseudo_count,
            )
            metrics_list.append(metrics)
            self._history.append(metrics)

        return metrics_list

    def _pseudo_label_step(self, unlab_x: Tensor) -> Tuple[Tensor, int]:
        """Pseudo-labeling step: predict on unlabelled, train on confident."""
        teacher = self._teacher if self._teacher is not None else self.model
        teacher.eval()

        with torch.no_grad():
            output = teacher(unlab_x)
            if isinstance(output, dict):
                logits = output.get("logits", output.get("output"))
            elif isinstance(output, (list, tuple)):
                logits = output[0]
            else:
                logits = output

            if logits.dim() == 3:
                logits = logits.mean(dim=1)

            if logits.dim() == 2 and logits.size(-1) > 1:
                probs = torch.softmax(logits, dim=-1)
                max_probs, pseudo_y = probs.max(dim=-1)
            else:
                p = torch.sigmoid(logits.squeeze(-1))
                max_probs = torch.where(p > 0.5, p, 1 - p)
                pseudo_y = (p > 0.5).long()

        mask = max_probs >= self.pseudo_label_threshold
        n_pseudo = int(mask.sum().item())

        if n_pseudo == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        self.model.train()
        selected_x = unlab_x[mask]
        selected_y = pseudo_y[mask]

        return self._supervised_loss(selected_x, selected_y), n_pseudo

    def _consistency_step(self, unlab_x: Tensor) -> Tensor:
        """Consistency regularisation between two augmented views."""
        self.model.train()
        aug1 = self.weak_aug(unlab_x)
        aug2 = self.strong_aug(unlab_x)
        return consistency_loss(self.model, aug1, aug2, loss_type="mse")

    def _fixmatch_step(self, unlab_x: Tensor) -> Tuple[Tensor, int]:
        """FixMatch: pseudo-label from weak aug, train on strong aug.

        Uses weak augmentation to generate pseudo-labels, then trains
        the model on the strongly augmented version with those labels.
        """
        teacher = self._teacher if self._teacher is not None else self.model
        teacher.eval()

        weak_x = self.weak_aug(unlab_x)
        strong_x = self.strong_aug(unlab_x)

        with torch.no_grad():
            output = teacher(weak_x)
            if isinstance(output, dict):
                logits = output.get("logits", output.get("output"))
            elif isinstance(output, (list, tuple)):
                logits = output[0]
            else:
                logits = output

            if logits.dim() == 3:
                logits = logits.mean(dim=1)

            if logits.dim() == 2 and logits.size(-1) > 1:
                probs = torch.softmax(logits, dim=-1)
                max_probs, pseudo_y = probs.max(dim=-1)
            else:
                p = torch.sigmoid(logits.squeeze(-1))
                max_probs = torch.where(p > 0.5, p, 1 - p)
                pseudo_y = (p > 0.5).long()

        mask = max_probs >= self.pseudo_label_threshold
        n_pseudo = int(mask.sum().item())

        if n_pseudo == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        self.model.train()
        selected_x = strong_x[mask]
        selected_y = pseudo_y[mask]

        return self._supervised_loss(selected_x, selected_y), n_pseudo

    def _mixmatch_step(
        self,
        lab_x: Tensor,
        lab_y: Tensor,
        unlab_x: Tensor,
    ) -> Tuple[Tensor, int]:
        """MixMatch: augmentation + pseudo-labels + mixup.

        Generates pseudo-labels from multiple augmented views,
        averages them (sharpening), then applies MixUp between
        labelled and pseudo-labelled data.
        """
        teacher = self._teacher if self._teacher is not None else self.model
        teacher.eval()
        n_augmentations = 2

        # Generate pseudo-labels from multiple augmentations
        all_probs: List[Tensor] = []
        with torch.no_grad():
            for _ in range(n_augmentations):
                aug_x = self.weak_aug(unlab_x)
                output = teacher(aug_x)
                if isinstance(output, dict):
                    logits = output.get("logits", output.get("output"))
                elif isinstance(output, (list, tuple)):
                    logits = output[0]
                else:
                    logits = output

                if logits.dim() == 3:
                    logits = logits.mean(dim=1)

                if logits.dim() == 2 and logits.size(-1) > 1:
                    probs = torch.softmax(logits, dim=-1)
                else:
                    p = torch.sigmoid(logits.squeeze(-1))
                    probs = torch.stack([1 - p, p], dim=-1)

                all_probs.append(probs)

        # Average and sharpen
        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        temperature = 0.5
        sharpened = avg_probs.pow(1.0 / temperature)
        sharpened = sharpened / sharpened.sum(dim=-1, keepdim=True)

        # Convert labelled targets to one-hot if needed
        num_classes = sharpened.size(-1)
        if lab_y.dim() == 1:
            lab_y_onehot = F.one_hot(lab_y, num_classes=num_classes).float()
        else:
            lab_y_onehot = lab_y.float()

        # MixUp
        self.model.train()
        mixed_x, mixed_y = _mixup(
            lab_x.float(), lab_y_onehot, unlab_x.float(), sharpened
        )

        output = self.model(mixed_x.long())
        if isinstance(output, dict):
            logits = output.get("logits", output.get("output"))
        elif isinstance(output, (list, tuple)):
            logits = output[0]
        else:
            logits = output

        if logits.dim() == 3:
            logits = logits.mean(dim=1)

        if logits.dim() == 2 and logits.size(-1) > 1:
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(mixed_y * log_probs).sum(dim=-1).mean()
        else:
            loss = F.mse_loss(torch.sigmoid(logits.squeeze(-1)), mixed_y[:, 1] if mixed_y.dim() == 2 else mixed_y)

        n_pseudo = int(unlab_x.size(0))
        return loss, n_pseudo
