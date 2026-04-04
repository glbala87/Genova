"""Model quantization utilities for Genova.

Provides post-training dynamic/static quantization, quantization-aware training
(QAT) preparation, GPTQ-style quantization stubs, and benchmarking helpers.

Example::

    from genova.models.quantization import quantize_dynamic, compare_model_sizes
    q_model = quantize_dynamic(model)
    sizes = compare_model_sizes(model, q_model)
"""

from __future__ import annotations

import copy
import io
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn

try:
    import torch.quantization as tq
    from torch.quantization import (
        DeQuantStub,
        QuantStub,
        default_dynamic_qconfig,
        default_qconfig,
        get_default_qat_qconfig,
        prepare,
        convert,
        prepare_qat,
    )

    _QUANTIZATION_AVAILABLE = True
except ImportError:
    _QUANTIZATION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_quantization() -> None:
    """Raise if PyTorch quantization APIs are unavailable."""
    if not _QUANTIZATION_AVAILABLE:
        raise ImportError(
            "torch.quantization is not available in your PyTorch build. "
            "Install a CPU-capable PyTorch build to use quantization features."
        )


def _model_size_bytes(model: nn.Module) -> int:
    """Return the serialized size of a model in bytes."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


# ---------------------------------------------------------------------------
# Dynamic quantization
# ---------------------------------------------------------------------------


def quantize_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    layer_types: Optional[List[Type[nn.Module]]] = None,
) -> nn.Module:
    """Apply post-training dynamic quantization to *model*.

    Dynamic quantization quantizes weights ahead of time and activations
    dynamically at inference time.  It is the simplest quantization approach
    and does not require calibration data.

    Args:
        model: The model to quantize.  A deep copy is made internally so the
            original model is not modified.
        dtype: Target dtype for quantized weights (``torch.qint8`` or
            ``torch.float16``).
        layer_types: Module types to quantize.  Defaults to
            ``[nn.Linear, nn.LSTM, nn.GRU]``.

    Returns:
        A dynamically-quantized copy of the model on CPU.
    """
    _check_quantization()

    if layer_types is None:
        layer_types = [nn.Linear, nn.LSTM, nn.GRU]

    # Dynamic quantization requires CPU
    model_copy = copy.deepcopy(model).cpu().eval()

    # Try available quantization engines; prefer fbgemm, fall back to qnnpack
    last_err: Optional[Exception] = None
    for engine in ("fbgemm", "qnnpack"):
        if engine in torch.backends.quantized.supported_engines:
            try:
                torch.backends.quantized.engine = engine
                quantized = torch.quantization.quantize_dynamic(
                    model_copy,
                    qconfig_spec=set(layer_types),
                    dtype=dtype,
                )
                return quantized
            except RuntimeError as exc:
                last_err = exc
                # Re-deepcopy because the failed attempt may have mutated model_copy
                model_copy = copy.deepcopy(model).cpu().eval()

    raise RuntimeError(
        "No working quantization engine found. Tried engines available in "
        f"torch.backends.quantized.supported_engines = "
        f"{torch.backends.quantized.supported_engines}."
    ) from last_err


# ---------------------------------------------------------------------------
# Static quantization
# ---------------------------------------------------------------------------


class _QuantWrapper(nn.Module):
    """Wraps a model with QuantStub / DeQuantStub for static quantization."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        # Only quantize the first positional tensor argument (input_ids)
        if args:
            first = self.quant(args[0].float())
            args = (first,) + args[1:]
        out = self.model(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            return self.dequant(out)
        if isinstance(out, dict) and "last_hidden_state" in out:
            out["last_hidden_state"] = self.dequant(out["last_hidden_state"])
        return out


def quantize_static(
    model: nn.Module,
    calibration_dataloader: Any,
    dtype: torch.dtype = torch.qint8,
    num_calibration_batches: int = 100,
    backend: str = "fbgemm",
) -> nn.Module:
    """Apply post-training static quantization with calibration.

    Static quantization quantizes both weights and activations.  It requires a
    representative calibration dataset to determine activation ranges.

    Args:
        model: The model to quantize (will be deep-copied).
        calibration_dataloader: An iterable yielding dicts with at least
            ``input_ids`` and optionally ``attention_mask``.
        dtype: Target weight dtype (currently ``torch.qint8`` only for static).
        num_calibration_batches: Maximum number of calibration batches.
        backend: Quantization backend (``"fbgemm"`` for x86,
            ``"qnnpack"`` for ARM).

    Returns:
        A statically-quantized copy of the model on CPU.
    """
    _check_quantization()

    torch.backends.quantized.engine = backend

    model_copy = copy.deepcopy(model).cpu().eval()
    wrapped = _QuantWrapper(model_copy)
    wrapped.qconfig = torch.quantization.get_default_qconfig(backend)

    prepared = torch.quantization.prepare(wrapped, inplace=False)

    # Calibration pass
    prepared.eval()
    with torch.no_grad():
        for i, batch in enumerate(calibration_dataloader):
            if i >= num_calibration_batches:
                break
            input_ids = batch["input_ids"].cpu()
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.cpu()
            try:
                prepared(input_ids=input_ids, attention_mask=attention_mask)
            except Exception:
                # Some layers may not be quantizable; skip errors during
                # calibration and let convert handle it.
                prepared(input_ids)

    quantized = torch.quantization.convert(prepared, inplace=False)
    return quantized


# ---------------------------------------------------------------------------
# Quantization-Aware Training (QAT)
# ---------------------------------------------------------------------------


def prepare_qat_model(
    model: nn.Module,
    backend: str = "fbgemm",
) -> nn.Module:
    """Prepare a model for quantization-aware training.

    Inserts fake-quantization modules so that the model can be fine-tuned
    while simulating quantization effects.  After QAT training, call
    ``torch.quantization.convert(model)`` to obtain the final quantized model.

    Args:
        model: The model to prepare (modified in-place on a copy).
        backend: Quantization backend.

    Returns:
        A model with fake-quantize observers inserted, ready for training.
    """
    _check_quantization()

    torch.backends.quantized.engine = backend

    model_copy = copy.deepcopy(model).cpu()
    wrapped = _QuantWrapper(model_copy)
    wrapped.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    wrapped.train()

    qat_model = torch.quantization.prepare_qat(wrapped, inplace=False)
    return qat_model


# ---------------------------------------------------------------------------
# GPTQ-style quantization (stub)
# ---------------------------------------------------------------------------


def quantize_gptq(
    model: nn.Module,
    calibration_dataloader: Any,
    bits: int = 4,
    group_size: int = 128,
    damp_percent: float = 0.01,
) -> nn.Module:
    """GPTQ-style weight-only quantization (stub).

    GPTQ performs layer-wise quantization using approximate second-order
    information (inverse Hessian) to minimize quantization error.  This
    implementation provides the correct interface but requires the ``auto_gptq``
    or ``gptq`` library for the actual algorithm.

    Args:
        model: The model to quantize.
        calibration_dataloader: Representative data for Hessian estimation.
        bits: Target bit-width (typically 4 or 3).
        group_size: Quantization group size for weight matrices.
        damp_percent: Dampening percentage for the Hessian diagonal.

    Returns:
        A GPTQ-quantized model.

    Raises:
        ImportError: If neither ``auto_gptq`` nor ``optimum`` is installed.
    """
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # type: ignore
    except ImportError:
        pass
    else:
        # auto_gptq path (if model is compatible)
        raise NotImplementedError(
            "auto_gptq integration is available but model-specific wiring "
            "is not yet implemented for Genova architectures.  Contributions welcome."
        )

    try:
        from optimum.gptq import GPTQQuantizer  # type: ignore
    except ImportError:
        pass
    else:
        raise NotImplementedError(
            "optimum.gptq integration is available but model-specific wiring "
            "is not yet implemented.  Contributions welcome."
        )

    raise ImportError(
        "GPTQ quantization requires 'auto-gptq' or 'optimum' to be installed.\n"
        "  pip install auto-gptq   # or\n"
        "  pip install optimum[gptq]"
    )


# ---------------------------------------------------------------------------
# Comparison / benchmarking
# ---------------------------------------------------------------------------


def compare_model_sizes(
    original: nn.Module,
    quantized: nn.Module,
) -> Dict[str, Any]:
    """Compare the serialized sizes of the original and quantized models.

    Args:
        original: The original (unquantized) model.
        quantized: The quantized model.

    Returns:
        Dict with ``original_size_mb``, ``quantized_size_mb``,
        ``compression_ratio``, and ``size_reduction_pct``.
    """
    orig_bytes = _model_size_bytes(original)
    quant_bytes = _model_size_bytes(quantized)

    orig_mb = orig_bytes / (1024 * 1024)
    quant_mb = quant_bytes / (1024 * 1024)
    ratio = orig_bytes / max(quant_bytes, 1)
    reduction = (1.0 - quant_bytes / max(orig_bytes, 1)) * 100.0

    return {
        "original_size_mb": round(orig_mb, 2),
        "quantized_size_mb": round(quant_mb, 2),
        "compression_ratio": round(ratio, 2),
        "size_reduction_pct": round(reduction, 2),
    }


def benchmark_inference(
    model: nn.Module,
    input_data: Dict[str, torch.Tensor],
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """Benchmark inference latency of a model.

    Args:
        model: The model to benchmark (should be in eval mode).
        input_data: A dict with at least ``input_ids`` (and optionally
            ``attention_mask``), matching what the model's ``forward`` expects.
        num_runs: Number of timed forward passes.
        warmup_runs: Number of untimed warmup passes.

    Returns:
        Dict with ``mean_latency_ms``, ``std_latency_ms``,
        ``min_latency_ms``, ``max_latency_ms``, ``throughput_samples_per_sec``.
    """
    model.eval()
    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device("cpu")

    # Move inputs to model device
    input_data = {k: v.to(device) for k, v in input_data.items()}
    batch_size = input_data["input_ids"].shape[0]

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(**input_data)

    # Synchronize for CUDA timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(**input_data)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)  # ms

    import statistics

    mean_lat = statistics.mean(latencies)
    std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    min_lat = min(latencies)
    max_lat = max(latencies)
    throughput = (batch_size / (mean_lat / 1000.0)) if mean_lat > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_lat, 3),
        "std_latency_ms": round(std_lat, 3),
        "min_latency_ms": round(min_lat, 3),
        "max_latency_ms": round(max_lat, 3),
        "throughput_samples_per_sec": round(throughput, 2),
        "num_runs": num_runs,
        "batch_size": batch_size,
    }
