"""Device management, mixed-precision setup, and GPU memory tracking.

Example::

    from genova.utils.device import DeviceManager

    dm = DeviceManager(mixed_precision="bf16")
    device = dm.device
    scaler = dm.grad_scaler  # None when using bf16 or cpu
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


def get_device(device_id: Optional[int] = None) -> torch.device:
    """Select the best available device.

    Args:
        device_id: Specific CUDA device ordinal. If *None*, selects
            ``cuda:0`` when available, else ``mps`` on Apple Silicon,
            else ``cpu``.

    Returns:
        A :class:`torch.device` instance.
    """
    if device_id is not None:
        return torch.device(f"cuda:{device_id}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DeviceManager:
    """Centralised device, dtype, and memory management.

    Attributes:
        mixed_precision: One of ``"no"``, ``"fp16"``, ``"bf16"``.
        device_id: CUDA device ordinal or *None* for auto-detection.
    """

    mixed_precision: str = "bf16"
    device_id: Optional[int] = None

    # Resolved at post-init
    device: torch.device = field(init=False)
    dtype: torch.dtype = field(init=False)
    grad_scaler: Optional[torch.amp.GradScaler] = field(init=False)

    def __post_init__(self) -> None:
        self.device = get_device(self.device_id)
        self.dtype, self.grad_scaler = _resolve_precision(
            self.mixed_precision, self.device
        )

    # ------------------------------------------------------------------
    # Context manager for autocast
    # ------------------------------------------------------------------

    def autocast_ctx(self) -> torch.amp.autocast:
        """Return an ``autocast`` context manager matching the current config."""
        device_type = self.device.type if self.device.type != "mps" else "cpu"
        enabled = self.mixed_precision != "no"
        return torch.amp.autocast(device_type=device_type, dtype=self.dtype, enabled=enabled)

    # ------------------------------------------------------------------
    # Memory utilities
    # ------------------------------------------------------------------

    @staticmethod
    def memory_stats() -> Dict[str, float]:
        """Return current CUDA memory statistics in megabytes.

        Returns an empty dict on non-CUDA devices.
        """
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
            "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        }

    @staticmethod
    def reset_peak_memory() -> None:
        """Reset CUDA peak memory tracking counters."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def empty_cache() -> None:
        """Free cached CUDA memory back to the allocator."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def list_gpus() -> List[Dict[str, object]]:
        """Return a summary of all visible CUDA devices."""
        if not torch.cuda.is_available():
            return []
        gpus: List[Dict[str, object]] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "id": i,
                    "name": props.name,
                    "total_memory_mb": props.total_mem / (1024 ** 2),
                    "major": props.major,
                    "minor": props.minor,
                }
            )
        return gpus


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_precision(
    mode: str, device: torch.device
) -> tuple[torch.dtype, Optional[torch.amp.GradScaler]]:
    """Map a precision string to a dtype and optional GradScaler."""
    if mode == "fp16":
        return torch.float16, torch.amp.GradScaler("cuda")
    if mode == "bf16":
        return torch.bfloat16, None
    # "no" or anything else
    return torch.float32, None
