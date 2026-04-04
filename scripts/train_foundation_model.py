#!/usr/bin/env python3
"""Train Genova foundation model.

Launch script that wires configuration, data, model, and the training
loop -- supporting single-GPU, multi-GPU (DDP), mixed precision,
wandb integration, and checkpoint resumption.

Usage:
    # Single GPU with small config
    python scripts/train_foundation_model.py --config configs/train_small.yaml

    # Multi-GPU with large config
    python scripts/train_foundation_model.py \\
        --config configs/train_large.yaml \\
        --gpus 4 \\
        --wandb-project genova-pretrain

    # Resume from checkpoint
    python scripts/train_foundation_model.py \\
        --config configs/train_small.yaml \\
        --resume outputs/genova_small/checkpoint-10000.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.multiprocessing as mp

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_genova")


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (trades speed for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Training worker
# ---------------------------------------------------------------------------

def train_worker(
    rank: int,
    world_size: int,
    config_path: str,
    overrides: List[str],
    data_dir: Optional[str],
    output_dir: Optional[str],
    resume_path: Optional[str],
    wandb_project: Optional[str],
) -> None:
    """Worker function executed by each process (single or DDP).

    Args:
        rank: Process rank (0 for single-GPU).
        world_size: Total number of processes.
        config_path: Path to YAML config.
        overrides: CLI config overrides.
        data_dir: Override data directory.
        output_dir: Override output directory.
        resume_path: Path to checkpoint to resume from.
        wandb_project: Wandb project name.
    """
    from genova.utils.config import GenovaConfig
    from genova.models.model_factory import create_model, count_parameters, model_summary
    from genova.data.tokenizer import GenomicTokenizer
    from genova.data.dataloader import create_dataloaders
    from genova.training.trainer import GenovaTrainer
    from genova.training.distributed import setup_distributed, cleanup_distributed

    # --- Distributed setup ---
    if world_size > 1:
        setup_distributed(rank, world_size)

    # --- Load configuration ---
    config = GenovaConfig.from_yaml(config_path, overrides=overrides or None)

    # Apply CLI overrides for directories
    if data_dir:
        config.data.genome_fasta = str(Path(data_dir) / "hg38.fa")
        config.data.train_regions = str(Path(data_dir) / "train_regions.bed")
        config.data.val_regions = str(Path(data_dir) / "val_regions.bed")
        config.data.test_regions = str(Path(data_dir) / "test_regions.bed")

    if output_dir:
        config.training.output_dir = output_dir

    if resume_path:
        config.training.resume_from_checkpoint = resume_path

    # Set seed (offset by rank for DDP diversity)
    set_seed(config.training.seed + rank)

    # --- Logging (rank 0 only) ---
    if rank == 0:
        logger.info("=" * 70)
        logger.info("Genova Foundation Model Training")
        logger.info("=" * 70)
        logger.info("Config:      %s", config_path)
        logger.info("Architecture: %s", config.model.arch)
        logger.info("Hidden dim:  %d", config.model.d_model)
        logger.info("Layers:      %d", config.model.n_layers)
        logger.info("Heads:       %d", config.model.n_heads)
        logger.info("Vocab size:  %d", config.model.vocab_size)
        logger.info("Seq length:  %d", config.data.seq_length)
        logger.info("Batch size:  %d", config.data.batch_size)
        logger.info("Epochs:      %d", config.training.epochs)
        logger.info("LR:          %s", config.training.lr)
        logger.info("GPUs:        %d", world_size)
        logger.info("Mixed prec.: %s", config.training.mixed_precision)
        logger.info("Output:      %s", config.training.output_dir)
        if resume_path:
            logger.info("Resuming:    %s", resume_path)
        logger.info("")

    # --- Wandb setup (rank 0 only) ---
    if rank == 0 and wandb_project:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                name=config.training.run_name,
                config=config.to_dict(),
                resume="allow" if resume_path else None,
            )
            logger.info("Wandb logging enabled (project=%s).", wandb_project)
        except Exception as exc:
            logger.warning("Failed to initialise wandb: %s", exc)

    # --- Build tokenizer ---
    tokenizer = GenomicTokenizer(
        mode=config.data.tokenizer,
        k=config.data.kmer_size,
        stride=config.data.stride if config.data.stride > 1 else 1,
    )
    tokenizer.build_vocab()

    # Sync vocab size with model config
    config.model.vocab_size = tokenizer.vocab_size
    config.data.vocab_size = tokenizer.vocab_size

    if rank == 0:
        logger.info(
            "Tokenizer: mode=%s, k=%d, vocab_size=%d",
            tokenizer.mode, tokenizer.k, tokenizer.vocab_size,
        )

    # --- Build data loaders ---
    if rank == 0:
        logger.info("Building data loaders...")

    try:
        loaders = create_dataloaders(
            config, tokenizer, rank=rank, world_size=world_size,
        )
    except Exception as exc:
        logger.error("Failed to create data loaders: %s", exc)
        logger.error(
            "Make sure the FASTA and BED files exist. "
            "Run scripts/prepare_training_data.py first."
        )
        if world_size > 1:
            cleanup_distributed()
        sys.exit(1)

    train_loader = loaders.get("train")
    val_loader = loaders.get("val")

    if train_loader is None:
        logger.error(
            "No training data available. Check data paths in your config."
        )
        if world_size > 1:
            cleanup_distributed()
        sys.exit(1)

    if rank == 0:
        logger.info("Train batches: %d", len(train_loader))
        if val_loader:
            logger.info("Val batches:   %d", len(val_loader))
        logger.info("")

    # --- Build model ---
    if rank == 0:
        logger.info("Building model...")

    model = create_model(config.model, task="mlm")
    n_params = count_parameters(model)
    summary = model_summary(model)

    if rank == 0:
        logger.info("Model: %s", config.model.arch)
        logger.info("  Total parameters:     %s", f"{summary['total_params']:,}")
        logger.info("  Trainable parameters: %s", f"{summary['trainable_params']:,}")
        logger.info(
            "  Model size (approx):  %.1f MB",
            summary["total_params"] * 4 / 1024 / 1024,
        )
        logger.info("")

    # --- Save resolved config ---
    run_output_dir = Path(config.training.output_dir) / config.training.run_name
    if rank == 0:
        run_output_dir.mkdir(parents=True, exist_ok=True)
        config.save_yaml(run_output_dir / "config.yaml")
        tokenizer.save(run_output_dir / "tokenizer.json")
        logger.info("Config saved to %s", run_output_dir / "config.yaml")

    # --- Build trainer and train ---
    if rank == 0:
        logger.info("Starting training...")
        logger.info("")

    start_time = time.time()

    trainer = GenovaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        rank=rank,
        world_size=world_size,
    )

    final_metrics = trainer.train()

    elapsed = time.time() - start_time

    # --- Report results ---
    if rank == 0:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Training Complete")
        logger.info("=" * 70)
        logger.info("Total time:  %.1f s (%.1f hours)", elapsed, elapsed / 3600)
        logger.info("Final step:  %d", trainer.global_step)
        logger.info("Best metric: %.4f", trainer.best_metric)

        if final_metrics:
            logger.info("Final metrics:")
            for k, v in final_metrics.items():
                logger.info("  %s: %.4f", k, v)

        # Save final training report
        report = {
            "config_path": config_path,
            "architecture": config.model.arch,
            "total_params": summary["total_params"],
            "trainable_params": summary["trainable_params"],
            "total_steps": trainer.global_step,
            "best_metric": trainer.best_metric,
            "final_metrics": final_metrics,
            "training_time_s": round(elapsed, 1),
            "gpus": world_size,
        }
        report_path = run_output_dir / "training_report.json"
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Training report saved to %s", report_path)

        # Close wandb
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

    # --- Cleanup ---
    if world_size > 1:
        cleanup_distributed()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Genova genomic foundation model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with small config
  python scripts/train_foundation_model.py --config configs/train_small.yaml

  # Production training
  python scripts/train_foundation_model.py \\
      --config configs/train_large.yaml \\
      --gpus 4 \\
      --wandb-project genova

  # Resume from checkpoint
  python scripts/train_foundation_model.py \\
      --config configs/train_small.yaml \\
      --resume outputs/genova_small/checkpoint-5000.pt
        """,
    )

    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/train_small.yaml).",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory (expects hg38.fa and *_regions.bed files).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--gpus", type=int, default=None,
        help="Number of GPUs to use. Default: all available (or 1 if no GPU).",
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None,
        help="Wandb project name. If set, enables wandb logging.",
    )
    parser.add_argument(
        "--override", "-o", nargs="*", default=None,
        help="Config overrides in key=value form (e.g. training.lr=3e-4).",
    )

    args = parser.parse_args()

    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    # Validate resume checkpoint
    if args.resume and not Path(args.resume).exists():
        logger.error("Checkpoint not found: %s", args.resume)
        sys.exit(1)

    # Determine world size
    if args.gpus is not None:
        world_size = args.gpus
    elif torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    # Ensure we don't request more GPUs than available
    if torch.cuda.is_available():
        available = torch.cuda.device_count()
        if world_size > available:
            logger.warning(
                "Requested %d GPUs but only %d available. Using %d.",
                world_size, available, available,
            )
            world_size = available
    else:
        if world_size > 1:
            logger.warning("No CUDA GPUs found. Falling back to CPU training.")
        world_size = 1

    logger.info("Device: %s", "CUDA" if torch.cuda.is_available() else "CPU")
    logger.info("World size: %d", world_size)

    overrides = args.override or []

    if world_size > 1:
        # DDP multi-GPU training via mp.spawn
        logger.info("Launching DDP training on %d GPUs...", world_size)
        mp.spawn(
            train_worker,
            args=(
                world_size,
                str(config_path),
                overrides,
                args.data_dir,
                args.output_dir,
                args.resume,
                args.wandb_project,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        # Single GPU / CPU training
        train_worker(
            rank=0,
            world_size=1,
            config_path=str(config_path),
            overrides=overrides,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            resume_path=args.resume,
            wandb_project=args.wandb_project,
        )


if __name__ == "__main__":
    main()
