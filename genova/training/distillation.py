"""Knowledge distillation utilities for Genova.

Provides loss functions and a trainer for distilling a large *teacher*
model into a smaller *student* model.  Two forms of distillation are
supported:

1. **Logit distillation** -- KL divergence between temperature-scaled
   teacher and student output distributions.
2. **Feature distillation** -- MSE loss between intermediate hidden
   representations of teacher and student.

Example::

    from genova.training.distillation import DistillationLoss, DistillationTrainer

    criterion = DistillationLoss(temperature=4.0, alpha=0.5)
    trainer = DistillationTrainer(teacher, student, criterion, optimizer)

    for batch in loader:
        loss = trainer.train_step(batch)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Distillation losses
# ---------------------------------------------------------------------------


class DistillationLoss(nn.Module):
    """Combined task loss and KL-divergence distillation loss.

    The total loss is::

        L = alpha * L_distill + (1 - alpha) * L_task

    where ``L_distill`` is the KL divergence between the temperature-scaled
    teacher and student logit distributions, and ``L_task`` is the standard
    cross-entropy loss against ground-truth labels.

    Args:
        temperature: Temperature for softening logits before KL divergence.
            Higher values produce softer probability distributions.
        alpha: Weighting factor.  ``alpha=1`` uses only distillation loss;
            ``alpha=0`` uses only task loss.  Default is 0.5.
        reduction: Reduction mode for the losses (``"mean"`` or ``"sum"``).
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Optional[Tensor] = None,
        ignore_index: int = -100,
    ) -> Dict[str, Tensor]:
        """Compute the combined distillation + task loss.

        Args:
            student_logits: ``(B, L, V)`` or ``(B, V)`` student output logits.
            teacher_logits: ``(B, L, V)`` or ``(B, V)`` teacher output logits
                (detached -- no gradients flow to teacher).
            labels: Optional ground-truth labels for the task loss.  If
                ``None``, only the distillation loss is computed.
            ignore_index: Label value to ignore in cross-entropy.

        Returns:
            Dictionary with keys ``"loss"`` (combined), ``"distill_loss"``,
            and optionally ``"task_loss"``.
        """
        T = self.temperature

        # Soft distributions
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits.detach() / T, dim=-1)

        # KL divergence: KL(teacher || student)
        distill_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction="batchmean" if self.reduction == "mean" else "sum",
        ) * (T * T)

        result: Dict[str, Tensor] = {"distill_loss": distill_loss}

        if labels is not None:
            task_loss = F.cross_entropy(
                student_logits.reshape(-1, student_logits.size(-1)),
                labels.reshape(-1),
                ignore_index=ignore_index,
                reduction=self.reduction,
            )
            result["task_loss"] = task_loss
            result["loss"] = self.alpha * distill_loss + (1.0 - self.alpha) * task_loss
        else:
            result["loss"] = distill_loss

        return result


class FeatureDistillationLoss(nn.Module):
    """MSE loss between intermediate representations of teacher and student.

    When teacher and student have different hidden dimensions, a learnable
    linear projection is used to map the student features to the teacher
    feature space before computing MSE.

    Args:
        student_dim: Student hidden dimension.
        teacher_dim: Teacher hidden dimension.
        normalize: Whether to L2-normalise features before computing MSE.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.proj: Optional[nn.Linear] = None
        if student_dim != teacher_dim:
            self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(
        self,
        student_features: Tensor,
        teacher_features: Tensor,
    ) -> Tensor:
        """Compute MSE between student and teacher features.

        Args:
            student_features: ``(B, L, D_s)`` student hidden states.
            teacher_features: ``(B, L, D_t)`` teacher hidden states
                (detached).

        Returns:
            Scalar MSE loss.
        """
        sf = student_features
        if self.proj is not None:
            sf = self.proj(sf)

        tf = teacher_features.detach()

        if self.normalize:
            sf = F.normalize(sf, dim=-1)
            tf = F.normalize(tf, dim=-1)

        return F.mse_loss(sf, tf)


# ---------------------------------------------------------------------------
# Distillation Trainer
# ---------------------------------------------------------------------------


class DistillationTrainer:
    """Trains a student model using knowledge distillation from a teacher.

    Combines logit-level distillation (via :class:`DistillationLoss`) with
    optional feature-level distillation (via :class:`FeatureDistillationLoss`).

    The teacher model is kept in ``eval()`` mode and ``torch.no_grad()``
    throughout.

    Args:
        teacher: Pre-trained teacher model.
        student: Student model to train.
        criterion: :class:`DistillationLoss` for logit distillation.
        optimizer: Optimizer for the student parameters.
        feature_loss: Optional :class:`FeatureDistillationLoss`.
        feature_weight: Weight for the feature distillation loss.
        device: Device to run on.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        criterion: DistillationLoss,
        optimizer: torch.optim.Optimizer,
        feature_loss: Optional[FeatureDistillationLoss] = None,
        feature_weight: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.criterion = criterion
        self.optimizer = optimizer
        self.feature_loss = feature_loss
        self.feature_weight = feature_weight
        self.device = device or next(student.parameters()).device

        # Teacher stays frozen in eval mode
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def train_step(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Run one training step with distillation.

        Args:
            input_ids: ``(B, L)`` input token ids.
            attention_mask: ``(B, L)`` attention mask.
            labels: ``(B, L)`` ground-truth labels for the task loss.

        Returns:
            Dictionary with ``"loss"`` and component losses.
        """
        self.student.train()

        need_hidden = self.feature_loss is not None

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = self.teacher(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=need_hidden,
            )

        # Student forward
        student_out = self.student(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=need_hidden,
        )

        # Logit distillation
        losses = self.criterion(
            student_logits=student_out["logits"],
            teacher_logits=teacher_out["logits"],
            labels=labels,
        )
        total_loss = losses["loss"]

        # Feature distillation
        if (
            self.feature_loss is not None
            and "hidden_states" in student_out
            and "hidden_states" in teacher_out
        ):
            s_hidden = student_out["hidden_states"]
            t_hidden = teacher_out["hidden_states"]
            # Use the last shared layer (min of the two lists)
            n_layers = min(len(s_hidden), len(t_hidden))
            feat_loss = torch.tensor(0.0, device=self.device)
            for i in range(n_layers):
                feat_loss = feat_loss + self.feature_loss(s_hidden[i], t_hidden[i])
            feat_loss = feat_loss / max(1, n_layers)
            losses["feature_loss"] = feat_loss
            total_loss = total_loss + self.feature_weight * feat_loss

        losses["loss"] = total_loss

        # Backward + step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {k: v.detach() for k, v in losses.items()}
