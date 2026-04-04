"""Model export utilities for ONNX, TorchScript, and TensorRT.

Supports exporting Genova models to optimised inference formats with
automatic verification that outputs match the original PyTorch model.

Usage::

    from genova.models.export import export_onnx, verify_onnx, export_torchscript

    export_onnx(model, sample_input, "model.onnx")
    verify_onnx("model.onnx", model, sample_input)
    export_torchscript(model, sample_input, "model.pt")
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import onnx
    import onnxruntime as ort

    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

try:
    import tensorrt as trt

    _HAS_TRT = True
except ImportError:
    _HAS_TRT = False


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_onnx(
    model: "nn.Module",
    sample_input: Any,
    path: Union[str, Path],
    opset_version: int = 14,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Export a PyTorch model to ONNX format.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model (must be in eval mode).
    sample_input : Tensor or tuple of Tensors
        Example input(s) for tracing.
    path : str or Path
        Output ``.onnx`` file path.
    opset_version : int
        ONNX opset version (default 14).
    dynamic_axes : dict, optional
        Dynamic axis specification. If None, batch dimension is set dynamic.
    input_names : list of str, optional
        Names for model inputs.
    output_names : list of str, optional
        Names for model outputs.

    Returns
    -------
    dict
        Export metadata including file size and timings.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for ONNX export.")
    if not _HAS_ONNX:
        raise ImportError(
            "onnx and onnxruntime are required. "
            "Install with: pip install onnx onnxruntime"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Default names
    if input_names is None:
        if isinstance(sample_input, (tuple, list)):
            input_names = [f"input_{i}" for i in range(len(sample_input))]
        else:
            input_names = ["input_ids"]

    if output_names is None:
        output_names = ["output"]

    # Default dynamic axes: batch dimension
    if dynamic_axes is None:
        dynamic_axes = {}
        for name in input_names:
            dynamic_axes[name] = {0: "batch_size"}
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}

    start_time = time.monotonic()

    with torch.no_grad():
        torch.onnx.export(
            model,
            sample_input if isinstance(sample_input, tuple) else (sample_input,),
            str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

    export_time = time.monotonic() - start_time
    file_size = path.stat().st_size

    # Get original model size
    orig_size = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )

    info = {
        "path": str(path),
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / 1e6, 2),
        "original_size_mb": round(orig_size / 1e6, 2),
        "compression_ratio": round(orig_size / file_size, 2) if file_size > 0 else 0,
        "export_time_s": round(export_time, 3),
        "opset_version": opset_version,
    }

    logger.info(
        "ONNX export complete: {} ({:.1f} MB, {:.1f}x compression, {:.2f}s)",
        path, info["file_size_mb"], info["compression_ratio"], export_time,
    )
    return info


# ---------------------------------------------------------------------------
# ONNX Verification
# ---------------------------------------------------------------------------

def verify_onnx(
    onnx_path: Union[str, Path],
    model: "nn.Module",
    sample_input: Any,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Verify that an ONNX model produces the same outputs as the PyTorch model.

    Parameters
    ----------
    onnx_path : str or Path
        Path to the ``.onnx`` file.
    model : nn.Module
        The original PyTorch model.
    sample_input : Tensor or tuple of Tensors
        The same sample input used for export.
    rtol : float
        Relative tolerance for comparison.
    atol : float
        Absolute tolerance for comparison.

    Returns
    -------
    dict
        Verification results including max absolute difference and pass/fail.

    Raises
    ------
    AssertionError
        If outputs differ beyond tolerance.
    """
    if not _HAS_ONNX:
        raise ImportError("onnx and onnxruntime are required for verification.")

    onnx_path = Path(onnx_path)

    # Check ONNX model validity
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # PyTorch inference
    model.eval()
    with torch.no_grad():
        if isinstance(sample_input, (tuple, list)):
            pt_output = model(*sample_input)
        else:
            pt_output = model(sample_input)

    if isinstance(pt_output, dict):
        pt_arrays = {k: v.cpu().numpy() for k, v in pt_output.items() if isinstance(v, torch.Tensor)}
    elif isinstance(pt_output, torch.Tensor):
        pt_arrays = {"output": pt_output.cpu().numpy()}
    else:
        pt_arrays = {"output": pt_output[0].cpu().numpy() if hasattr(pt_output, "__getitem__") else np.array([])}

    # ONNX Runtime inference
    session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {}

    if isinstance(sample_input, (tuple, list)):
        for i, inp in enumerate(sample_input):
            name = session.get_inputs()[i].name
            ort_inputs[name] = inp.cpu().numpy() if hasattr(inp, "numpy") else np.array(inp)
    else:
        name = session.get_inputs()[0].name
        ort_inputs[name] = sample_input.cpu().numpy() if hasattr(sample_input, "numpy") else np.array(sample_input)

    ort_outputs = session.run(None, ort_inputs)

    # Compare
    pt_first = list(pt_arrays.values())[0]
    ort_first = ort_outputs[0]

    max_diff = float(np.max(np.abs(pt_first - ort_first)))
    mean_diff = float(np.mean(np.abs(pt_first - ort_first)))
    passed = np.allclose(pt_first, ort_first, rtol=rtol, atol=atol)

    result = {
        "passed": passed,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "rtol": rtol,
        "atol": atol,
        "output_shape": list(ort_first.shape),
    }

    if passed:
        logger.info(
            "ONNX verification PASSED (max_diff={:.6f}, mean_diff={:.6f})",
            max_diff, mean_diff,
        )
    else:
        logger.warning(
            "ONNX verification FAILED (max_diff={:.6f}, mean_diff={:.6f})",
            max_diff, mean_diff,
        )

    return result


# ---------------------------------------------------------------------------
# TorchScript Export
# ---------------------------------------------------------------------------

def export_torchscript(
    model: "nn.Module",
    sample_input: Any,
    path: Union[str, Path],
    method: str = "trace",
) -> Dict[str, Any]:
    """Export a PyTorch model to TorchScript.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model.
    sample_input : Tensor or tuple of Tensors
        Example input for tracing.
    path : str or Path
        Output ``.pt`` file path.
    method : str
        ``"trace"`` (default) or ``"script"``.

    Returns
    -------
    dict
        Export metadata.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for TorchScript export.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    start_time = time.monotonic()

    if method == "script":
        scripted = torch.jit.script(model)
    else:
        with torch.no_grad():
            if isinstance(sample_input, (tuple, list)):
                scripted = torch.jit.trace(model, sample_input)
            else:
                scripted = torch.jit.trace(model, (sample_input,))

    scripted.save(str(path))
    export_time = time.monotonic() - start_time
    file_size = path.stat().st_size

    orig_size = sum(p.numel() * p.element_size() for p in model.parameters())

    info = {
        "path": str(path),
        "method": method,
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / 1e6, 2),
        "original_size_mb": round(orig_size / 1e6, 2),
        "export_time_s": round(export_time, 3),
    }

    logger.info(
        "TorchScript export complete: {} ({:.1f} MB, {:.2f}s)",
        path, info["file_size_mb"], export_time,
    )
    return info


# ---------------------------------------------------------------------------
# TensorRT optimisation (optional)
# ---------------------------------------------------------------------------

def optimize_tensorrt(
    onnx_path: Union[str, Path],
    output_path: Union[str, Path],
    max_batch_size: int = 8,
    fp16: bool = True,
    workspace_mb: int = 1024,
) -> Dict[str, Any]:
    """Optimise an ONNX model with TensorRT.

    Parameters
    ----------
    onnx_path : str or Path
        Path to the ONNX model.
    output_path : str or Path
        Path for the TensorRT engine file.
    max_batch_size : int
        Maximum batch size.
    fp16 : bool
        Enable FP16 precision.
    workspace_mb : int
        GPU workspace in MB.

    Returns
    -------
    dict
        Optimisation metadata.

    Raises
    ------
    ImportError
        If TensorRT is not installed.
    """
    if not _HAS_TRT:
        raise ImportError(
            "TensorRT is required. Install with: pip install tensorrt"
        )

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("TensorRT parse error: {}", parser.get_error(i))
            raise RuntimeError("TensorRT ONNX parsing failed.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    start_time = time.monotonic()
    engine = builder.build_serialized_network(network, config)
    build_time = time.monotonic() - start_time

    with open(output_path, "wb") as f:
        f.write(engine)

    file_size = output_path.stat().st_size
    onnx_size = onnx_path.stat().st_size

    info = {
        "path": str(output_path),
        "file_size_mb": round(file_size / 1e6, 2),
        "onnx_size_mb": round(onnx_size / 1e6, 2),
        "fp16": fp16,
        "build_time_s": round(build_time, 3),
    }

    logger.info(
        "TensorRT optimisation complete: {} ({:.1f} MB, {:.2f}s)",
        output_path, info["file_size_mb"], build_time,
    )
    return info


# ---------------------------------------------------------------------------
# Model size comparison
# ---------------------------------------------------------------------------

def compare_model_sizes(
    model: "nn.Module",
    paths: Dict[str, Union[str, Path]],
) -> Dict[str, Any]:
    """Compare file sizes across export formats.

    Parameters
    ----------
    model : nn.Module
        The original PyTorch model.
    paths : dict
        Mapping from format name to file path.

    Returns
    -------
    dict
        Size comparison table.
    """
    orig_size = sum(p.numel() * p.element_size() for p in model.parameters())
    comparison: Dict[str, Any] = {
        "pytorch_memory_mb": round(orig_size / 1e6, 2),
    }

    for name, path in paths.items():
        path = Path(path)
        if path.exists():
            size = path.stat().st_size
            comparison[f"{name}_file_mb"] = round(size / 1e6, 2)
            comparison[f"{name}_ratio"] = round(orig_size / size, 2) if size > 0 else 0
        else:
            comparison[f"{name}_file_mb"] = "N/A"

    return comparison
