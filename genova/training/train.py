"""Training entry point for Genova.

Provides :func:`run_training` which wires up configuration, data,
model, and the trainer -- including optional DDP launch.

Can be called from the CLI or imported directly::

    from genova.training.train import run_training
    run_training("configs/pretrain.yaml", overrides=["training.lr=3e-4"])
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger

from genova.utils.config import GenovaConfig
from genova.models.model_factory import create_model, count_parameters
from genova.data.tokenizer import GenomicTokenizer
from genova.data.dataloader import create_dataloaders
from genova.training.trainer import GenovaTrainer
from genova.training.distributed import setup_distributed, cleanup_distributed


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Single-process worker
# ---------------------------------------------------------------------------


def _train_worker(
    rank: int,
    world_size: int,
    config: GenovaConfig,
) -> None:
    """Worker function executed by each DDP process.

    Args:
        rank: Process rank.
        world_size: Total number of processes.
        config: Fully resolved training configuration.
    """
    # Distributed init
    if world_size > 1:
        setup_distributed(rank, world_size)

    _set_seed(config.training.seed + rank)

    # Tokenizer
    tokenizer = GenomicTokenizer(
        mode=config.data.tokenizer,
        k=config.data.kmer_size,
        stride=config.data.stride,
    )
    tokenizer.build_vocab()

    # DataLoaders
    loaders = create_dataloaders(
        config,
        tokenizer,
        rank=rank,
        world_size=world_size,
    )

    train_loader = loaders.get("train")
    val_loader = loaders.get("val")

    if train_loader is None:
        raise RuntimeError(
            "No training data available. Check data.genome_fasta and "
            "data.train_regions in your config."
        )

    # Model
    model = create_model(config.model, task="mlm")
    n_params = count_parameters(model)
    logger.info("Model created with {:,} trainable parameters.", n_params)

    # Trainer
    trainer = GenovaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        rank=rank,
        world_size=world_size,
    )

    # Run
    final_metrics = trainer.train()
    logger.info("Final metrics: {}", final_metrics)

    # Cleanup
    if world_size > 1:
        cleanup_distributed()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_training(
    config_path: Union[str, Path],
    overrides: Optional[Sequence[str]] = None,
) -> None:
    """Launch Genova training from a YAML config file.

    Handles DDP spawning when ``config.training.ddp`` is ``True`` and
    multiple GPUs are available.

    Args:
        config_path: Path to a YAML configuration file.
        overrides: Optional CLI-style overrides, e.g.
            ``["training.lr=3e-4", "model.n_layers=24"]``.
    """
    config = GenovaConfig.from_yaml(config_path, overrides=overrides)

    logger.info("Configuration loaded from {}", config_path)
    logger.info("Run name: {}", config.training.run_name)
    logger.info("Output dir: {}", config.training.output_dir)

    # Save resolved config
    output_dir = Path(config.training.output_dir) / config.training.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(output_dir / "config.yaml")

    # Determine world size
    world_size = 1
    if config.training.ddp and torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        if world_size < 2:
            logger.warning(
                "DDP requested but only {} GPU(s) found; falling back to single-GPU.",
                world_size,
            )
            world_size = 1

    if world_size > 1:
        logger.info("Launching DDP training on {} GPUs.", world_size)
        mp.spawn(
            _train_worker,
            args=(world_size, config),
            nprocs=world_size,
            join=True,
        )
    else:
        _train_worker(rank=0, world_size=1, config=config)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a Genova model.")
    parser.add_argument("config", type=str, help="Path to YAML config file.")
    parser.add_argument(
        "--override",
        "-o",
        nargs="*",
        default=None,
        help="Config overrides in key=value form.",
    )
    args = parser.parse_args()
    run_training(args.config, overrides=args.override)
