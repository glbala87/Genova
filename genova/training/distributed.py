"""Distributed Data Parallel (DDP) utilities for Genova training.

Provides helpers for initialising and tearing down PyTorch distributed
process groups, checking rank, and reducing metrics across workers.

Example::

    from genova.training.distributed import setup_distributed, cleanup_distributed

    setup_distributed(rank=0, world_size=4)
    # ... training ...
    cleanup_distributed()
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Union

import torch
import torch.distributed as dist
from loguru import logger


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "29500",
) -> None:
    """Initialise the distributed process group.

    Args:
        rank: Global rank of the current process.
        world_size: Total number of processes.
        backend: Communication backend (``"nccl"`` for GPU, ``"gloo"`` for CPU).
        master_addr: Address of the rank-0 process.
        master_port: Port used for rendezvous.
    """
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        logger.info(
            "Distributed process group initialised: rank={}/{}, backend={}",
            rank,
            world_size,
            backend,
        )


def cleanup_distributed() -> None:
    """Destroy the distributed process group if active."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed.")


def is_main_process(rank: Optional[int] = None) -> bool:
    """Return ``True`` if the current process is rank 0.

    Args:
        rank: Explicit rank value.  If ``None``, queries the process group
            (falls back to ``True`` when not in a distributed context).
    """
    if rank is not None:
        return rank == 0
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_rank() -> int:
    """Return the global rank (0 when not distributed)."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Return the world size (1 when not distributed)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def reduce_metrics(
    metrics: Dict[str, float],
    device: Optional[torch.device] = None,
    average: bool = True,
) -> Dict[str, float]:
    """All-reduce a dictionary of scalar metrics across processes.

    Args:
        metrics: Mapping of metric name to scalar value.
        device: Device on which to perform the reduction.  Defaults to
            ``cuda`` when available, otherwise ``cpu``.
        average: If ``True``, divide summed values by world size.

    Returns:
        Dictionary with reduced (optionally averaged) values.  When not
        running in a distributed context the input is returned unchanged.
    """
    if not dist.is_initialized():
        return metrics

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_size = dist.get_world_size()
    reduced: Dict[str, float] = {}

    for key, value in metrics.items():
        tensor = torch.tensor(value, dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if average:
            tensor /= world_size
        reduced[key] = tensor.item()

    return reduced


def broadcast_object(obj: object, src: int = 0) -> object:
    """Broadcast a picklable Python object from *src* to all ranks.

    Args:
        obj: The object to broadcast (only meaningful on rank *src*).
        src: Source rank.

    Returns:
        The broadcast object on every rank.
    """
    if not dist.is_initialized():
        return obj
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]
