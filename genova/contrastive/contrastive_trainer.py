"""Training pipeline for contrastive learning on genomic sequences.

Orchestrates data augmentation, positive/negative pair construction via
in-batch negatives, training loop, embedding extraction, and downstream
evaluation (clustering of promoters, enhancers, and intergenic regions).

Example::

    from genova.contrastive.contrastive_trainer import ContrastiveTrainer

    trainer = ContrastiveTrainer(
        model=contrastive_model,
        augmenter=augmenter,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir="outputs/contrastive",
    )
    trainer.train(num_epochs=50)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

from genova.contrastive.augmentations import GenomicAugmenter
from genova.contrastive.contrastive_model import ContrastiveGenovaModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _silhouette_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute silhouette score; falls back to 0.0 if sklearn is unavailable."""
    try:
        from sklearn.metrics import silhouette_score
        return float(silhouette_score(embeddings, labels))
    except ImportError:
        logger.warning(
            "scikit-learn is not installed; silhouette score will be 0.0."
        )
        return 0.0


def _kmeans_cluster_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> float:
    """Compute cluster purity via k-means; returns 0.0 if sklearn absent."""
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        preds = km.fit_predict(embeddings)
        # Purity: fraction of correctly assigned samples
        total = 0
        for k in range(n_clusters):
            cluster_mask = preds == k
            if cluster_mask.sum() == 0:
                continue
            counts = np.bincount(labels[cluster_mask].astype(int))
            total += counts.max()
        return float(total) / len(labels)
    except ImportError:
        logger.warning(
            "scikit-learn is not installed; cluster accuracy will be 0.0."
        )
        return 0.0


# ---------------------------------------------------------------------------
# Contrastive Trainer
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Configuration for the contrastive training loop.

    Args:
        output_dir: Directory for checkpoints and logs.
        num_epochs: Number of training epochs.
        log_every_n_steps: Logging frequency.
        eval_every_n_steps: Evaluation frequency.
        save_every_n_steps: Checkpoint saving frequency.
        max_grad_norm: Gradient clipping norm.
        mixed_precision: Enable AMP (``"fp16"`` or ``"bf16"``; empty to disable).
        gradient_accumulation_steps: Number of micro-batches per optimiser step.
    """

    output_dir: str = "outputs/contrastive"
    num_epochs: int = 50
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 1000
    save_every_n_steps: int = 5000
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"
    gradient_accumulation_steps: int = 1


class ContrastiveTrainer:
    """End-to-end training pipeline for SimCLR-style contrastive learning.

    Handles:
    - Positive pair generation via :class:`GenomicAugmenter`.
    - In-batch negative sampling (no explicit negatives needed).
    - Mixed-precision training with gradient scaling.
    - Periodic checkpoint saving and evaluation.
    - Embedding extraction for downstream tasks.
    - Evaluation via k-means clustering of functional regions.

    Args:
        model: The contrastive model to train.
        augmenter: Genomic augmenter that produces two views per sample.
        train_loader: DataLoader yielding batches of ``(input_ids,)`` or
            ``(input_ids, attention_mask)`` or dicts with ``"input_ids"``.
        val_loader: Optional validation DataLoader.
        optimizer: PyTorch optimiser.
        scheduler: Optional learning-rate scheduler.
        config: Training configuration.
        device: Target device (auto-detected if ``None``).
    """

    def __init__(
        self,
        model: ContrastiveGenovaModel,
        augmenter: GenomicAugmenter,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        config: Optional[TrainerConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model = model.to(self.device)
        self.augmenter = augmenter
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.01
        )
        self.scheduler = scheduler

        # Mixed precision
        self._use_amp = self.config.mixed_precision in ("fp16", "bf16")
        self._amp_dtype = (
            torch.bfloat16
            if self.config.mixed_precision == "bf16"
            else torch.float16
        )
        self.scaler = GradScaler(enabled=(self.config.mixed_precision == "fp16"))

        # State
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")
        self._output_dir = Path(self.config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Batch unpacking
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_batch(
        batch: Any,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Extract input_ids and optional attention_mask from a batch.

        Supports tuples ``(input_ids,)`` / ``(input_ids, mask)`` and dicts
        with ``"input_ids"`` / ``"attention_mask"`` keys.
        """
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
        elif isinstance(batch, (list, tuple)):
            input_ids = batch[0]
            attention_mask = batch[1] if len(batch) > 1 else None
        else:
            input_ids = batch
            attention_mask = None
        return input_ids, attention_mask

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train_step(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
    ) -> float:
        """Execute a single training step and return the scalar loss."""
        # Augment: generate two views (positive pair)
        view1, view2 = self.augmenter(input_ids)
        view1 = view1.to(self.device)
        view2 = view2.to(self.device)
        mask1 = attention_mask.to(self.device) if attention_mask is not None else None
        mask2 = mask1  # same mask for both augmented views

        with autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self._use_amp,
        ):
            outputs = self.model(view1, view2, mask1, mask2)
            loss = outputs["loss"] / self.config.gradient_accumulation_steps

        self.scaler.scale(loss).backward()
        return loss.item() * self.config.gradient_accumulation_steps

    def _optimiser_step(self) -> None:
        """Clip gradients, step optimiser, and zero grads."""
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if self.scheduler is not None:
            self.scheduler.step()

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """Run the full contrastive training loop.

        Args:
            num_epochs: Override for number of epochs (uses config default
                if ``None``).

        Returns:
            Dict containing ``"train_losses"`` and optionally ``"val_losses"``
            tracked across epochs.
        """
        epochs = num_epochs or self.config.num_epochs
        history: Dict[str, List[float]] = {"train_losses": [], "val_losses": []}

        logger.info(
            "Starting contrastive training: %d epochs, %d batches/epoch",
            epochs,
            len(self.train_loader),
        )

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            t0 = time.time()

            self.optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(self.train_loader):
                input_ids, attention_mask = self._unpack_batch(batch)
                input_ids = input_ids.to(self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                loss_val = self._train_step(input_ids, attention_mask)
                epoch_loss += loss_val
                n_batches += 1

                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self._optimiser_step()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.log_every_n_steps == 0:
                        avg = epoch_loss / n_batches
                        lr = self.optimizer.param_groups[0]["lr"]
                        logger.info(
                            "Epoch %d | Step %d | Loss %.4f | LR %.2e",
                            epoch + 1,
                            self.global_step,
                            avg,
                            lr,
                        )

                    # Evaluation
                    if (
                        self.val_loader is not None
                        and self.global_step % self.config.eval_every_n_steps == 0
                    ):
                        val_loss = self.evaluate()
                        logger.info(
                            "Epoch %d | Step %d | Val Loss %.4f",
                            epoch + 1,
                            self.global_step,
                            val_loss,
                        )
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint("best.pt")

                    # Checkpoint
                    if self.global_step % self.config.save_every_n_steps == 0:
                        self.save_checkpoint(f"step_{self.global_step}.pt")

            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - t0
            history["train_losses"].append(avg_epoch_loss)

            logger.info(
                "Epoch %d complete | Avg Loss %.4f | Time %.1fs",
                epoch + 1,
                avg_epoch_loss,
                elapsed,
            )

            # End-of-epoch validation
            if self.val_loader is not None:
                val_loss = self.evaluate()
                history["val_losses"].append(val_loss)
                logger.info("Epoch %d | Val Loss %.4f", epoch + 1, val_loss)

        self.save_checkpoint("final.pt")
        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute average contrastive loss on the validation set.

        Returns:
            Mean NT-Xent loss over all validation batches.
        """
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            input_ids, attention_mask = self._unpack_batch(batch)
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            view1, view2 = self.augmenter(input_ids)
            view1 = view1.to(self.device)
            view2 = view2.to(self.device)
            mask = attention_mask

            with autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
                enabled=self._use_amp,
            ):
                outputs = self.model(view1, view2, mask, mask)

            total_loss += outputs["loss"].item()
            n_batches += 1

        self.model.train()
        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_embeddings(
        self,
        dataloader: DataLoader,
        use_projection_head: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract embeddings for all samples in a dataloader.

        Args:
            dataloader: DataLoader yielding batches.  If batches are dicts,
                a ``"label"`` key is extracted as well.
            use_projection_head: If ``True``, return projected embeddings
                (after the MLP head).  Otherwise return encoder-pooled
                embeddings (typically better for downstream tasks).

        Returns:
            Tuple of ``(embeddings, labels)`` as numpy arrays.
            *labels* is ``None`` if the dataloader does not supply labels.
        """
        self.model.eval()
        all_embeddings: List[Tensor] = []
        all_labels: List[Tensor] = []

        for batch in dataloader:
            input_ids, attention_mask = self._unpack_batch(batch)
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            if use_projection_head:
                emb = self.model.encode(input_ids, attention_mask)
            else:
                emb = self.model.get_embeddings(input_ids, attention_mask)

            all_embeddings.append(emb.cpu())

            # Try to extract labels
            if isinstance(batch, dict) and "label" in batch:
                all_labels.append(batch["label"])
            elif isinstance(batch, (list, tuple)) and len(batch) > 2:
                all_labels.append(batch[2])

        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        labels = (
            torch.cat(all_labels, dim=0).numpy()
            if all_labels
            else None
        )

        self.model.train()
        return embeddings, labels

    # ------------------------------------------------------------------
    # Downstream evaluation: cluster genomic regions
    # ------------------------------------------------------------------

    def evaluate_clustering(
        self,
        eval_loader: DataLoader,
        n_clusters: int = 3,
        region_names: Sequence[str] = ("promoter", "enhancer", "intergenic"),
    ) -> Dict[str, float]:
        """Evaluate learned representations by clustering functional regions.

        Extracts embeddings, runs k-means clustering, and measures cluster
        quality via silhouette score and cluster purity.

        Expected label encoding:
            - 0: promoter
            - 1: enhancer
            - 2: intergenic

        Args:
            eval_loader: DataLoader yielding labelled genomic region batches.
            n_clusters: Number of k-means clusters (should match number of
                region types).
            region_names: Names for logging.

        Returns:
            Dict with ``"silhouette_score"`` and ``"cluster_purity"`` keys.
        """
        logger.info("Evaluating clustering on %d region types...", n_clusters)

        embeddings, labels = self.extract_embeddings(eval_loader)

        if labels is None:
            logger.warning("No labels found in eval_loader; skipping clustering eval.")
            return {"silhouette_score": 0.0, "cluster_purity": 0.0}

        silhouette = _silhouette_score(embeddings, labels)
        purity = _kmeans_cluster_accuracy(embeddings, labels, n_clusters)

        logger.info(
            "Clustering results: silhouette=%.4f, purity=%.4f",
            silhouette,
            purity,
        )
        for i, name in enumerate(region_names[:n_clusters]):
            count = int((labels == i).sum())
            logger.info("  %s: %d samples", name, count)

        return {
            "silhouette_score": silhouette,
            "cluster_purity": purity,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str) -> Path:
        """Save a training checkpoint.

        Args:
            filename: Name of the checkpoint file.

        Returns:
            Path to the saved checkpoint.
        """
        path = self._output_dir / filename
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "scaler_state_dict": self.scaler.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s", path)
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint and restore state.

        Args:
            path: Path to the checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(
            "Checkpoint loaded from %s (step %d)", path, self.global_step
        )
