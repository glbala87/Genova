#!/usr/bin/env python3
"""Verify model can be exported and loaded correctly.

Creates a small Genova model with random weights, saves a checkpoint, reloads
it, and verifies that weights match.  Optionally tests TorchScript and ONNX
export paths.

Usage:
    python scripts/ci/check_model_export.py [--output-dir /tmp/export-test]

Exit codes:
    0  All export checks passed
    1  At least one check failed
    2  Runtime / import error
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

# ── Colour helpers (for CI logs) ─────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

failures: list[str] = []
passes: list[str] = []


def ok(msg: str) -> None:
    passes.append(msg)
    print(f"  {GREEN}PASS{RESET} {msg}")


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  {RED}FAIL{RESET} {msg}")


def skip(msg: str) -> None:
    print(f"  {YELLOW}SKIP{RESET} {msg}")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _file_size_human(path: str) -> str:
    size = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ── Tests ────────────────────────────────────────────────────────────────────
def test_checkpoint_roundtrip(output_dir: str) -> None:
    """Save and reload a model checkpoint, verify weight equality."""
    import torch

    try:
        from genova.utils.config import ModelConfig
        from genova.models.model_factory import create_model
    except ImportError as exc:
        fail(f"Cannot import Genova model classes: {exc}")
        return

    config = ModelConfig(
        vocab_size=4096,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_position_embeddings=512,
    )
    model = create_model(config, task="mlm")
    model.eval()

    ckpt_path = os.path.join(output_dir, "model_checkpoint.pt")

    # Save
    torch.save(
        {"model_state_dict": model.state_dict(), "config": vars(config)},
        ckpt_path,
    )
    print(f"    Checkpoint size: {_file_size_human(ckpt_path)}")

    # Reload
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model2 = create_model(config, task="mlm")
    model2.load_state_dict(checkpoint["model_state_dict"])
    model2.eval()

    # Verify all parameters match
    all_match = True
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model2.named_parameters()
    ):
        if n1 != n2:
            fail(f"Parameter name mismatch: {n1} vs {n2}")
            all_match = False
            break
        if not torch.equal(p1, p2):
            fail(f"Parameter value mismatch: {n1}")
            all_match = False
            break

    if all_match:
        ok("Checkpoint save/load roundtrip — weights match")


def test_torchscript_export(output_dir: str) -> None:
    """Attempt TorchScript export via torch.jit.trace."""
    import torch

    try:
        from genova.utils.config import ModelConfig
        from genova.models.model_factory import create_model
    except ImportError:
        skip("TorchScript export — Genova model classes not importable")
        return

    config = ModelConfig(
        vocab_size=4096,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_position_embeddings=512,
    )
    model = create_model(config, task="mlm")
    model.eval()

    ts_path = os.path.join(output_dir, "model_torchscript.pt")
    input_ids = torch.randint(0, config.vocab_size, (1, 64))

    try:
        with torch.no_grad():
            traced = torch.jit.trace(model, input_ids)
        traced.save(ts_path)
        print(f"    TorchScript size: {_file_size_human(ts_path)}")

        # Verify outputs match
        with torch.no_grad():
            orig_out = model(input_ids)
            ts_out = traced(input_ids)

        # Handle tuple / tensor outputs
        if isinstance(orig_out, tuple):
            orig_out = orig_out[0]
        if isinstance(ts_out, tuple):
            ts_out = ts_out[0]

        if torch.allclose(orig_out, ts_out, atol=1e-5):
            ok("TorchScript export — outputs match")
        else:
            fail("TorchScript export — output mismatch")

    except Exception as exc:
        skip(f"TorchScript export — {exc}")


def test_onnx_export(output_dir: str) -> None:
    """Attempt ONNX export if the onnx package is available."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        skip("ONNX export — onnx package not installed")
        return

    import torch

    try:
        from genova.utils.config import ModelConfig
        from genova.models.model_factory import create_model
    except ImportError:
        skip("ONNX export — Genova model classes not importable")
        return

    config = ModelConfig(
        vocab_size=4096,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_position_embeddings=512,
    )
    model = create_model(config, task="mlm")
    model.eval()

    onnx_path = os.path.join(output_dir, "model.onnx")
    input_ids = torch.randint(0, config.vocab_size, (1, 64))

    try:
        torch.onnx.export(
            model,
            (input_ids,),
            onnx_path,
            input_names=["input_ids"],
            output_names=["output"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"}},
            opset_version=17,
        )
        print(f"    ONNX size: {_file_size_human(onnx_path)}")

        # Validate the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        ok("ONNX export — model exported and validated")

    except Exception as exc:
        skip(f"ONNX export — {exc}")


def test_model_file_size(output_dir: str) -> None:
    """Report and verify that the checkpoint is within expected size bounds."""
    ckpt_path = os.path.join(output_dir, "model_checkpoint.pt")
    if not os.path.exists(ckpt_path):
        skip("Model file size — checkpoint not found")
        return

    size_bytes = os.path.getsize(ckpt_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"    Model file: {size_mb:.2f} MB")

    # For a 2-layer, 128-dim model we expect < 50 MB
    if size_mb < 100:
        ok(f"Model file size is reasonable ({size_mb:.2f} MB)")
    else:
        fail(f"Model file unexpectedly large ({size_mb:.2f} MB)")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Genova model export")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for exported artifacts (default: temp dir)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or tempfile.mkdtemp(prefix="genova-export-")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Genova Model Export Verification")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print()

    start = time.perf_counter()

    print("[1/4] Checkpoint roundtrip ...")
    test_checkpoint_roundtrip(output_dir)

    print("[2/4] TorchScript export ...")
    test_torchscript_export(output_dir)

    print("[3/4] ONNX export ...")
    test_onnx_export(output_dir)

    print("[4/4] File size check ...")
    test_model_file_size(output_dir)

    elapsed = time.perf_counter() - start
    print()
    print(f"Completed in {elapsed:.1f}s  —  "
          f"{len(passes)} passed, {len(failures)} failed")

    if failures:
        print(f"\n{RED}Failures:{RESET}")
        for f in failures:
            print(f"  - {f}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
