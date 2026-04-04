#!/usr/bin/env python3
"""Standalone load test for Genova API.

Starts a minimal FastAPI test app with mock /health and /embed endpoints,
sends 100 concurrent requests, measures latency percentiles and throughput,
saves results to outputs/genova_expanded/load_test_results.json, then stops.

Dependencies: only stdlib + fastapi + uvicorn (already installed).
"""

from __future__ import annotations

import json
import os
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn
from fastapi import FastAPI

# ---------------------------------------------------------------------------
# 1. Minimal FastAPI test app
# ---------------------------------------------------------------------------

app = FastAPI(title="Genova Load-Test Mock")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed")
def embed(payload: dict):
    # Simulate a tiny bit of work
    seq = payload.get("sequence", "")
    return {"embedding": [0.1] * min(len(seq), 64), "length": len(seq)}


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout: float = 10.0) -> bool:
    """Block until the server responds on /health or timeout expires."""
    deadline = time.monotonic() + timeout
    url = f"http://{host}:{port}/health"
    while time.monotonic() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=2)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


# ---------------------------------------------------------------------------
# 3. Request sender (stdlib only)
# ---------------------------------------------------------------------------

def _send_request(url: str, method: str = "GET", body: bytes | None = None) -> dict:
    """Send a single HTTP request and return timing info."""
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    start = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        elapsed = time.monotonic() - start
        _ = resp.read()
        return {"status": resp.status, "latency": elapsed, "error": None}
    except Exception as exc:
        elapsed = time.monotonic() - start
        return {"status": 0, "latency": elapsed, "error": str(exc)}


def _worker(results: list, index: int, url: str, method: str, body: bytes | None):
    """Thread worker that stores its result at results[index]."""
    results[index] = _send_request(url, method, body)


# ---------------------------------------------------------------------------
# 4. Load test orchestrator
# ---------------------------------------------------------------------------

def run_load_test(base_url: str, total_requests: int = 100) -> dict:
    """Send total_requests concurrently and compute latency percentiles."""

    # Build a mix: 60 health GETs + 40 embed POSTs
    n_health = int(total_requests * 0.6)
    n_embed = total_requests - n_health

    tasks: list[tuple[str, str, bytes | None]] = []
    tasks.extend(
        (f"{base_url}/health", "GET", None) for _ in range(n_health)
    )
    embed_body = json.dumps({"sequence": "ATCGATCG" * 16}).encode()
    tasks.extend(
        (f"{base_url}/embed", "POST", embed_body) for _ in range(n_embed)
    )

    results: list[dict | None] = [None] * total_requests
    threads: list[threading.Thread] = []

    wall_start = time.monotonic()
    for i, (url, method, body) in enumerate(tasks):
        t = threading.Thread(target=_worker, args=(results, i, url, method, body))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=30)

    wall_elapsed = time.monotonic() - wall_start

    # Compute stats
    latencies = [r["latency"] for r in results if r and r["error"] is None]
    errors = [r for r in results if r and r["error"] is not None]

    if not latencies:
        return {"error": "All requests failed", "failures": len(errors)}

    latencies.sort()
    n = len(latencies)

    def percentile(pct: float) -> float:
        idx = int(pct / 100 * n)
        idx = min(idx, n - 1)
        return latencies[idx]

    summary = {
        "total_requests": total_requests,
        "successful": len(latencies),
        "failed": len(errors),
        "wall_time_s": round(wall_elapsed, 4),
        "throughput_rps": round(len(latencies) / wall_elapsed, 2),
        "latency_ms": {
            "min": round(min(latencies) * 1000, 2),
            "p50": round(percentile(50) * 1000, 2),
            "p95": round(percentile(95) * 1000, 2),
            "p99": round(percentile(99) * 1000, 2),
            "max": round(max(latencies) * 1000, 2),
            "mean": round(statistics.mean(latencies) * 1000, 2),
            "stdev": round(statistics.stdev(latencies) * 1000, 2) if n > 1 else 0,
        },
    }

    return summary


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main() -> int:
    port = _find_free_port()
    host = "127.0.0.1"

    # Start uvicorn in a daemon thread
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    if not _wait_for_server(host, port):
        print("ERROR: Mock server did not start in time.", file=sys.stderr)
        return 1

    base_url = f"http://{host}:{port}"
    print(f"Mock server running at {base_url}")
    print("Sending 100 concurrent requests (60 GET /health + 40 POST /embed)...")
    print()

    summary = run_load_test(base_url, total_requests=100)

    # Print results
    if "error" in summary:
        print(f"ERROR: {summary['error']}", file=sys.stderr)
        return 1

    print("=== Load Test Results ===")
    print(f"  Total requests : {summary['total_requests']}")
    print(f"  Successful     : {summary['successful']}")
    print(f"  Failed         : {summary['failed']}")
    print(f"  Wall time      : {summary['wall_time_s']}s")
    print(f"  Throughput     : {summary['throughput_rps']} req/s")
    print()
    lat = summary["latency_ms"]
    print("  Latency (ms):")
    print(f"    min  : {lat['min']}")
    print(f"    p50  : {lat['p50']}")
    print(f"    p95  : {lat['p95']}")
    print(f"    p99  : {lat['p99']}")
    print(f"    max  : {lat['max']}")
    print(f"    mean : {lat['mean']}")
    print(f"    stdev: {lat['stdev']}")
    print()

    # Save to outputs directory
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "outputs" / "genova_expanded"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "load_test_results.json"

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output_path}")

    # Shutdown server
    server.should_exit = True
    server_thread.join(timeout=5)

    return 0


if __name__ == "__main__":
    sys.exit(main())
