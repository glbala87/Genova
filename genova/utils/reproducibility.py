"""Reproducibility utilities for Genova.

Provides deterministic seeding for all random number generators and an
optional fully-deterministic mode (at the cost of performance).

Example::

    from genova.utils.reproducibility import set_seed, enable_deterministic_mode

    set_seed(42)
    enable_deterministic_mode()
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Also seeds CUDA if a GPU is available.

    Args:
        seed: The integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure hash-based operations are reproducible.
    os.environ["PYTHONHASHSEED"] = str(seed)


def enable_deterministic_mode(warn: bool = True) -> None:
    """Enable PyTorch deterministic algorithms globally.

    This forces cuDNN and CUDA operations to use deterministic
    implementations. Training throughput may decrease.

    Args:
        warn: If *True* (default), PyTorch will emit warnings when a
            non-deterministic operation is encountered. If *False*,
            those operations will raise an error instead.
    """
    torch.use_deterministic_algorithms(True, warn_only=warn)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def disable_deterministic_mode() -> None:
    """Restore default (non-deterministic) PyTorch behaviour.

    Re-enables cuDNN auto-tuner for faster convolution selection.
    """
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
