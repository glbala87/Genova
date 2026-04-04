#!/usr/bin/env python3
"""Run performance benchmarks for CI pipeline.

Measures tokenizer throughput, model forward-pass latency and memory usage,
and DataLoader throughput.  Results are saved as JSON and optionally compared
against a baseline file.

Usage:
    python scripts/ci/run_benchmarks.py [--baseline baseline.json] [--output results.json]

Exit codes:
    0  All benchmarks within threshold
    1  At least one benchmark regressed >20% vs. baseline
    2  Runtime error (import failure, missing module, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Regression threshold: a benchmark is considered a failure if it is more than
# this fraction slower (for latency) or slower (for throughput) than baseline.
REGRESSION_THRESHOLD = 0.20  # 20%


# ── Result dataclass ─────────────────────────────────────────────────────────
@dataclass
class BenchmarkResult:
    name: str
    metric: str  # e.g. "ops_per_second", "latency_ms", "samples_per_second"
    value: float
    unit: str
    iterations: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# ── Tokenizer benchmark ─────────────────────────────────────────────────────
def bench_tokenizer(seq_len: int = 1024, n_iter: int = 1000) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    try:
        from genova.data.tokenizer import GenomicTokenizer

        tokenizer = GenomicTokenizer(mode="kmer", k=6)
        tokenizer.build_vocab()
    except Exception as exc:
        results.append(
            BenchmarkResult(
                name="tokenizer_encode",
                metric="ops_per_second",
                value=0.0,
                unit="ops/s",
                error=str(exc),
            )
        )
        return results

    seq = "ATCGATCG" * (seq_len // 8)

    # Encode throughput
    start = time.perf_counter()
    for _ in range(n_iter):
        ids = tokenizer.encode(seq)
    elapsed = time.perf_counter() - start
    results.append(
        BenchmarkResult(
            name=f"tokenizer_encode_{seq_len}bp",
            metric="ops_per_second",
            value=round(n_iter / elapsed, 2),
            unit="ops/s",
            iterations=n_iter,
            extra={"total_seconds": round(elapsed, 4)},
        )
    )

    # Decode throughput
    try:
        start = time.perf_counter()
        for _ in range(n_iter):
            tokenizer.decode(ids)
        elapsed = time.perf_counter() - start
        results.append(
            BenchmarkResult(
                name=f"tokenizer_decode_{seq_len}bp",
                metric="ops_per_second",
                value=round(n_iter / elapsed, 2),
                unit="ops/s",
                iterations=n_iter,
                extra={"total_seconds": round(elapsed, 4)},
            )
        )
    except Exception as exc:
        results.append(
            BenchmarkResult(
                name=f"tokenizer_decode_{seq_len}bp",
                metric="ops_per_second",
                value=0.0,
                unit="ops/s",
                error=str(exc),
            )
        )

    return results


# ── Model forward-pass benchmark ────────────────────────────────────────────
def bench_model_forward(
    seq_len: int = 256, n_warmup: int = 3, n_iter: int = 20
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    try:
        import torch
        from genova.utils.config import ModelConfig
        from genova.models.model_factory import create_model

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

        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                model(input_ids)

        # Benchmark
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iter):
                model(input_ids)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / n_iter) * 1000.0
        results.append(
            BenchmarkResult(
                name=f"model_forward_{seq_len}tok_cpu",
                metric="latency_ms",
                value=round(avg_ms, 2),
                unit="ms",
                iterations=n_iter,
                extra={"total_seconds": round(elapsed, 4)},
            )
        )

        # Memory usage
        import psutil

        process = psutil.Process(os.getpid())
        rss_mb = process.memory_info().rss / (1024 * 1024)
        n_params = sum(p.numel() for p in model.parameters())
        results.append(
            BenchmarkResult(
                name="model_memory_small",
                metric="rss_mb",
                value=round(rss_mb, 2),
                unit="MB",
                extra={"param_count": n_params, "param_count_human": f"{n_params / 1e6:.2f}M"},
            )
        )

    except Exception as exc:
        results.append(
            BenchmarkResult(
                name=f"model_forward_{seq_len}tok_cpu",
                metric="latency_ms",
                value=0.0,
                unit="ms",
                error=str(exc),
            )
        )

    return results


# ── DataLoader throughput ────────────────────────────────────────────────────
def bench_dataloader(n_batches: int = 50, batch_size: int = 8) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # Synthetic dataset simulating tokenized genomic sequences
        seq_len = 512
        dataset = TensorDataset(
            torch.randint(0, 4096, (n_batches * batch_size, seq_len)),
            torch.randint(0, 2, (n_batches * batch_size,)),
        )
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

        start = time.perf_counter()
        count = 0
        for batch in loader:
            count += 1
        elapsed = time.perf_counter() - start

        samples_per_sec = (count * batch_size) / elapsed
        results.append(
            BenchmarkResult(
                name="dataloader_throughput",
                metric="samples_per_second",
                value=round(samples_per_sec, 2),
                unit="samples/s",
                iterations=count,
                extra={"batch_size": batch_size, "total_seconds": round(elapsed, 4)},
            )
        )

    except Exception as exc:
        results.append(
            BenchmarkResult(
                name="dataloader_throughput",
                metric="samples_per_second",
                value=0.0,
                unit="samples/s",
                error=str(exc),
            )
        )

    return results


# ── Comparison logic ─────────────────────────────────────────────────────────
def compare_results(
    current: list[BenchmarkResult], baseline_path: str
) -> tuple[bool, list[str]]:
    """Compare current results against a baseline JSON file.

    Returns (passed, report_lines).  passed is False if any benchmark regressed
    beyond REGRESSION_THRESHOLD.
    """
    try:
        with open(baseline_path) as f:
            baseline_data = json.load(f)
    except FileNotFoundError:
        return True, ["No baseline file found; skipping comparison."]

    baseline_by_name: dict[str, dict] = {}
    for b in baseline_data.get("benchmarks", []):
        if b.get("error") is None:
            baseline_by_name[b["name"]] = b

    passed = True
    lines: list[str] = []

    for result in current:
        if result.error is not None:
            continue
        prev = baseline_by_name.get(result.name)
        if prev is None:
            lines.append(f"  {result.name}: no baseline (new)")
            continue

        prev_val = prev["value"]
        cur_val = result.value

        if prev_val == 0:
            lines.append(f"  {result.name}: baseline is 0, skipping")
            continue

        # For latency metrics, higher is worse; for throughput, lower is worse.
        if result.metric in ("latency_ms",):
            pct_change = (cur_val - prev_val) / prev_val
            direction = "slower" if pct_change > 0 else "faster"
        else:
            pct_change = (prev_val - cur_val) / prev_val
            direction = "slower" if pct_change > 0 else "faster"

        status = "OK"
        if pct_change > REGRESSION_THRESHOLD:
            status = "REGRESSION"
            passed = False

        lines.append(
            f"  {result.name}: {cur_val} {result.unit} "
            f"(was {prev_val}, {pct_change:+.1%} {direction}) [{status}]"
        )

    return passed, lines


# ── Formatted output ─────────────────────────────────────────────────────────
def print_table(results: list[BenchmarkResult]) -> None:
    """Print a formatted table of benchmark results."""
    header = f"{'Benchmark':<35} {'Value':>12} {'Unit':<15} {'Iters':>6}  Status"
    sep = "-" * len(header)

    print()
    print(header)
    print(sep)

    for r in results:
        status = "ERROR" if r.error else "OK"
        val = f"{r.value:>12.2f}" if r.error is None else f"{'N/A':>12}"
        print(f"{r.name:<35} {val} {r.unit:<15} {r.iterations:>6}  {status}")
        if r.error:
            print(f"  Error: {r.error}")

    print(sep)
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Run Genova performance benchmarks")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline JSON for regression comparison",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark-results.json",
        help="Output path for results JSON (default: benchmark-results.json)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Genova Performance Benchmarks")
    print("=" * 60)

    all_results: list[BenchmarkResult] = []

    print("\n[1/3] Tokenizer benchmarks ...")
    all_results.extend(bench_tokenizer())

    print("[2/3] Model forward-pass benchmarks ...")
    all_results.extend(bench_model_forward())

    print("[3/3] DataLoader throughput ...")
    all_results.extend(bench_dataloader())

    # Print table
    print_table(all_results)

    # Serialize
    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": os.environ.get("GITHUB_SHA", "local")[:8],
        "benchmarks": [asdict(r) for r in all_results],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results written to {output_path}")

    # Compare with baseline
    exit_code = 0
    if args.baseline:
        print(f"\nComparing against baseline: {args.baseline}")
        passed, report = compare_results(all_results, args.baseline)
        for line in report:
            print(line)
        if not passed:
            print("\nREGRESSION DETECTED: one or more benchmarks exceeded the "
                  f"{REGRESSION_THRESHOLD:.0%} threshold.")
            exit_code = 1
        else:
            print("\nAll benchmarks within acceptable range.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
