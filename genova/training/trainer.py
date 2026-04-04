"""Production training pipeline for Genova genomic foundation models.

Handles single-GPU, distributed (DDP), and FSDP training with mixed
precision, gradient accumulation, checkpointing, early stopping, and
logging to wandb / TensorBoard.

Example::

    from genova.utils.config import GenovaConfig
    from genova.training.trainer import GenovaTrainer

    config = GenovaConfig.from_yaml("configs/pretrain.yaml")
    trainer = GenovaTrainer(model, train_loader, val_loader, config)
    trainer.train()
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from loguru import logger

from genova.utils.config import GenovaConfig, TrainingConfig
from genova.training.scheduler import create_scheduler
from genova.training.distributed import (
    is_main_process,
    get_rank,
    get_world_size,
    reduce_metrics,
)
from genova.training.fsdp import (
    FSDPConfig,
    setup_fsdp,
    save_fsdp_checkpoint,
    load_fsdp_checkpoint,
    FSDP_AVAILABLE,
)


class GenovaTrainer:
    """End-to-end training driver for Genova models.

    Supports single-GPU, DDP, and FSDP parallelism strategies.

    Args:
        model: The model to train (unwrapped :class:`nn.Module`).
        train_loader: Training :class:`DataLoader`.
        val_loader: Validation :class:`DataLoader` (may be ``None``).
        config: Full :class:`GenovaConfig` (uses ``training``, ``data``,
            and ``model`` sections).
        optimizer: Optional pre-built optimizer.  When ``None``, AdamW is
            created from the config.
        scheduler: Optional pre-built LR scheduler.  When ``None``, one is
            created via :func:`~genova.training.scheduler.create_scheduler`.
        rank: DDP rank (``0`` for single-GPU).
        world_size: Number of DDP processes (``1`` for single-GPU).
        fsdp_config: Optional :class:`FSDPConfig`.  When provided (and
            enabled), the model is wrapped with FSDP instead of DDP.
            Alternatively, set ``config.training.fsdp = True`` to auto-create
            a default FSDP configuration.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: GenovaConfig,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        rank: int = 0,
        world_size: int = 1,
        fsdp_config: Optional[FSDPConfig] = None,
    ) -> None:
        self.config = config
        self.tcfg: TrainingConfig = config.training
        self.rank = rank
        self.world_size = world_size

        # FSDP configuration -----------------------------------------------
        # Detect FSDP from explicit arg, or from config.training.fsdp flag
        self.fsdp_config: Optional[FSDPConfig] = fsdp_config
        if self.fsdp_config is None and getattr(self.tcfg, "fsdp", False):
            self.fsdp_config = FSDPConfig(
                mixed_precision=self.tcfg.mixed_precision,
            )
        self.use_fsdp: bool = (
            self.fsdp_config is not None
            and self.fsdp_config.enabled
            and FSDP_AVAILABLE
        )

        # Device -----------------------------------------------------------
        if torch.cuda.is_available():
            self.device = torch.device("cuda", rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # Model ------------------------------------------------------------
        if self.use_fsdp:
            # FSDP handles device placement internally
            self.model = setup_fsdp(model, self.fsdp_config, device_id=rank)  # type: ignore[arg-type]
            self._unwrapped_model = model
        else:
            self.model = model.to(self.device)
            if self.tcfg.compile_model and hasattr(torch, "compile"):
                self.model = torch.compile(self.model)  # type: ignore[attr-defined]
            if world_size > 1:
                self.model = DDP(self.model, device_ids=[rank], output_device=rank)
            self._unwrapped_model: nn.Module = (
                self.model.module if isinstance(self.model, DDP) else self.model
            )

        # Data -------------------------------------------------------------
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer --------------------------------------------------------
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = self._build_optimizer()

        # Steps calculation ------------------------------------------------
        steps_per_epoch = math.ceil(
            len(self.train_loader) / self.tcfg.gradient_accumulation_steps
        )
        if self.tcfg.max_steps > 0:
            self.total_steps = self.tcfg.max_steps
        else:
            self.total_steps = steps_per_epoch * self.tcfg.epochs

        # Scheduler --------------------------------------------------------
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = create_scheduler(
                self.optimizer, self.tcfg, self.total_steps
            )

        # Mixed precision --------------------------------------------------
        self.use_amp = self.tcfg.mixed_precision in ("fp16", "bf16")
        self.amp_dtype = (
            torch.bfloat16
            if self.tcfg.mixed_precision == "bf16"
            else torch.float16
        )
        self.scaler: Optional[GradScaler] = (
            GradScaler() if self.tcfg.mixed_precision == "fp16" else None
        )

        # State tracking ---------------------------------------------------
        self.global_step: int = 0
        self.epoch: int = 0
        self.best_metric: float = float("inf")
        self._steps_without_improvement: int = 0

        # Output -----------------------------------------------------------
        self.output_dir = Path(self.tcfg.output_dir) / self.tcfg.run_name
        if is_main_process(self.rank):
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logging back-ends ------------------------------------------------
        self._wandb_run: Optional[Any] = None
        self._tb_writer: Optional[Any] = None
        if is_main_process(self.rank):
            self._init_loggers()

        # Resume -----------------------------------------------------------
        if self.tcfg.resume_from_checkpoint:
            self.load_checkpoint(self.tcfg.resume_from_checkpoint)

    # ------------------------------------------------------------------
    # Optimizer factory
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW optimizer with weight-decay filtering."""
        no_decay = {"bias", "LayerNorm.weight", "layernorm.weight", "norm.weight"}
        param_groups = [
            {
                "params": [
                    p
                    for n, p in self._unwrapped_model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.tcfg.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self._unwrapped_model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(param_groups, lr=self.tcfg.lr)

    # ------------------------------------------------------------------
    # Logger helpers
    # ------------------------------------------------------------------

    def _init_loggers(self) -> None:
        """Initialise wandb and/or TensorBoard if available."""
        # wandb
        try:
            import wandb

            if wandb.api.api_key:
                self._wandb_run = wandb.init(
                    project="genova",
                    name=self.tcfg.run_name,
                    config=self.config.to_dict(),
                    resume="allow",
                )
                logger.info("Weights & Biases logging enabled.")
        except Exception:
            logger.debug("wandb not available or not configured; skipping.")

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = self.output_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info("TensorBoard logging enabled at {}", tb_dir)
        except Exception:
            logger.debug("TensorBoard not available; skipping.")

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Write *metrics* to all active logging back-ends."""
        if not is_main_process(self.rank):
            return
        if self._wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step)
        if self._tb_writer is not None:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(k, v, global_step=step)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, float]:
        """Run the full training loop.

        Returns:
            Dictionary of final metrics from the last evaluation.
        """
        logger.info(
            "Starting training: epochs={}, total_steps={}, device={}",
            self.tcfg.epochs,
            self.total_steps,
            self.device,
        )

        # Progress bar (main process only)
        pbar = None
        if is_main_process(self.rank):
            try:
                from tqdm.auto import tqdm

                pbar = tqdm(
                    total=self.total_steps,
                    initial=self.global_step,
                    desc="Training",
                    unit="step",
                )
            except ImportError:
                pass

        last_eval_metrics: Dict[str, float] = {}

        for epoch in range(self.epoch, self.tcfg.epochs):
            self.epoch = epoch

            # Distributed sampler epoch
            if hasattr(self.train_loader, "sampler") and hasattr(
                self.train_loader.sampler, "set_epoch"
            ):
                self.train_loader.sampler.set_epoch(epoch)

            epoch_metrics = self._train_one_epoch(pbar)

            # End-of-epoch evaluation
            if self.val_loader is not None:
                eval_metrics = self.evaluate()
                last_eval_metrics = eval_metrics

                if is_main_process(self.rank):
                    logger.info(
                        "Epoch {} eval: {}",
                        epoch,
                        {k: f"{v:.4f}" for k, v in eval_metrics.items()},
                    )

                # Early stopping on validation loss
                val_loss = eval_metrics.get("val/loss", float("inf"))
                if val_loss < self.best_metric:
                    self.best_metric = val_loss
                    self._steps_without_improvement = 0
                    if self.use_fsdp or is_main_process(self.rank):
                        # FSDP requires all ranks to participate in state dict
                        # gathering; save_fsdp_checkpoint handles rank-gating
                        self.save_checkpoint(self.output_dir / "best_model.pt")
                else:
                    self._steps_without_improvement += 1
                    if (
                        self.tcfg.early_stopping_patience > 0
                        and self._steps_without_improvement
                        >= self.tcfg.early_stopping_patience
                    ):
                        logger.info(
                            "Early stopping triggered after {} epochs without improvement.",
                            self._steps_without_improvement,
                        )
                        break

            if self.global_step >= self.total_steps:
                break

        if pbar is not None:
            pbar.close()

        # Final checkpoint
        if self.use_fsdp or is_main_process(self.rank):
            self.save_checkpoint(self.output_dir / "final_model.pt")
        if is_main_process(self.rank):
            if self._tb_writer is not None:
                self._tb_writer.close()
            if self._wandb_run is not None:
                import wandb

                wandb.finish()

        logger.info("Training complete. Global step: {}", self.global_step)
        return last_eval_metrics

    def _train_one_epoch(self, pbar: Optional[Any] = None) -> Dict[str, float]:
        """Execute one training epoch.

        Returns:
            Dictionary of aggregated training metrics for the epoch.
        """
        self.model.train()
        accum_loss: float = 0.0
        accum_steps: int = 0
        epoch_loss: float = 0.0
        epoch_tokens: int = 0
        step_start = time.monotonic()

        self.optimizer.zero_grad(set_to_none=True)

        for micro_step, batch in enumerate(self.train_loader):
            if self.global_step >= self.total_steps:
                break

            # Move to device
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            # Forward
            with autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                )
                # Support models returning dict or object with .loss
                if isinstance(outputs, dict):
                    loss = outputs["loss"]
                else:
                    loss = outputs.loss

                loss = loss / self.tcfg.gradient_accumulation_steps

            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()
            accum_steps += 1

            # Mask count for perplexity tracking
            if "labels" in batch:
                epoch_tokens += (batch["labels"] != -100).sum().item()

            # Optimiser step at accumulation boundary
            if accum_steps % self.tcfg.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.tcfg.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    # FSDP exposes clip_grad_norm_ on the wrapped model
                    clip_params = (
                        self.model.parameters()
                        if self.use_fsdp
                        else self._unwrapped_model.parameters()
                    )
                    nn.utils.clip_grad_norm_(
                        clip_params,
                        self.tcfg.max_grad_norm,
                    )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                step_loss = accum_loss * self.tcfg.gradient_accumulation_steps
                epoch_loss += step_loss
                accum_loss = 0.0

                # Logging
                if (
                    self.global_step % self.tcfg.log_every_n_steps == 0
                    and is_main_process(self.rank)
                ):
                    elapsed = time.monotonic() - step_start
                    current_lr = self.scheduler.get_last_lr()[0]
                    perplexity = math.exp(min(step_loss, 20.0))
                    metrics = {
                        "train/loss": step_loss,
                        "train/perplexity": perplexity,
                        "train/lr": current_lr,
                        "train/epoch": self.epoch,
                        "train/step_time_s": elapsed / self.tcfg.log_every_n_steps,
                    }
                    self._log_metrics(metrics, self.global_step)
                    logger.info(
                        "step={} loss={:.4f} ppl={:.2f} lr={:.2e}",
                        self.global_step,
                        step_loss,
                        perplexity,
                        current_lr,
                    )
                    step_start = time.monotonic()

                # Progress bar
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{step_loss:.4f}")

                # Mid-epoch evaluation
                if (
                    self.val_loader is not None
                    and self.tcfg.eval_every_n_steps > 0
                    and self.global_step % self.tcfg.eval_every_n_steps == 0
                ):
                    eval_metrics = self.evaluate()
                    self._log_metrics(eval_metrics, self.global_step)
                    self.model.train()

                # Mid-epoch checkpoint
                if (
                    self.tcfg.save_every_n_steps > 0
                    and self.global_step % self.tcfg.save_every_n_steps == 0
                    and (self.use_fsdp or is_main_process(self.rank))
                ):
                    ckpt_path = (
                        self.output_dir / f"checkpoint-{self.global_step}.pt"
                    )
                    self.save_checkpoint(ckpt_path)

        num_steps = max(1, self.global_step)
        return {"train/epoch_loss": epoch_loss / num_steps}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the validation set.

        Returns:
            Dictionary with ``val/loss`` and ``val/perplexity``.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss: float = 0.0
        total_correct: int = 0
        total_tokens: int = 0
        num_batches: int = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            with autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                )

            if isinstance(outputs, dict):
                loss = outputs["loss"]
                logits = outputs.get("logits")
            else:
                loss = outputs.loss
                logits = getattr(outputs, "logits", None)

            total_loss += loss.item()
            num_batches += 1

            # Accuracy on masked positions
            if logits is not None and "labels" in batch:
                mask = batch["labels"] != -100
                if mask.any():
                    preds = logits.argmax(dim=-1)
                    total_correct += (preds[mask] == batch["labels"][mask]).sum().item()
                    total_tokens += mask.sum().item()

        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(min(avg_loss, 20.0))
        accuracy = total_correct / max(1, total_tokens)

        metrics = {
            "val/loss": avg_loss,
            "val/perplexity": perplexity,
            "val/accuracy": accuracy,
        }

        # Reduce across processes
        metrics = reduce_metrics(metrics, device=self.device)

        if is_main_process(self.rank):
            logger.info(
                "Eval: loss={:.4f} ppl={:.2f} acc={:.4f}",
                metrics["val/loss"],
                metrics["val/perplexity"],
                metrics["val/accuracy"],
            )

        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save full training state to *path*.

        Includes model weights, optimizer state, scheduler state, scaler
        state, epoch, global step, and best metric.  When FSDP is active,
        uses ``StateDictType.FULL_STATE_DICT`` for a consolidated checkpoint.

        Args:
            path: Destination file path.
        """
        path = Path(path)

        if self.use_fsdp:
            save_fsdp_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                path=path,
                epoch=self.epoch,
                global_step=self.global_step,
                best_metric=self.best_metric,
                config=self.config.to_dict(),
                scheduler_state=self.scheduler.state_dict(),
                scaler_state=self.scaler.state_dict() if self.scaler else None,
                rank=self.rank,
            )
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self._unwrapped_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "config": self.config.to_dict(),
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to {}", path)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Restore full training state from *path*.

        Automatically detects whether the checkpoint was saved with FSDP
        and uses the appropriate loading strategy.

        Args:
            path: Checkpoint file path.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if self.use_fsdp:
            metadata = load_fsdp_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                path=path,
                device=self.device,
            )
            self.epoch = metadata.get("epoch", 0)
            self.global_step = metadata.get("global_step", 0)
            self.best_metric = metadata.get("best_metric", float("inf"))

            if metadata.get("scheduler_state_dict") is not None:
                self.scheduler.load_state_dict(metadata["scheduler_state_dict"])
            if self.scaler is not None and metadata.get("scaler_state_dict") is not None:
                self.scaler.load_state_dict(metadata["scaler_state_dict"])

            logger.info(
                "FSDP checkpoint loaded from {} (epoch={}, step={}, best_metric={:.4f})",
                path,
                self.epoch,
                self.global_step,
                self.best_metric,
            )
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self._unwrapped_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", float("inf"))

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(
            "Checkpoint loaded from {} (epoch={}, step={}, best_metric={:.4f})",
            path,
            self.epoch,
            self.global_step,
            self.best_metric,
        )
