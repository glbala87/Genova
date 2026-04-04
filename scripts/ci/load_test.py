#!/usr/bin/env python3
"""Load test for Genova API using Locust.

Can be executed in two modes:

1. As a Locust file (with Locust UI or headless):
       locust -f scripts/ci/load_test.py --host http://localhost:8000

2. As a standalone script (headless, with built-in result parsing):
       python scripts/ci/load_test.py \
           --host http://localhost:8000 \
           --users 20 \
           --spawn-rate 5 \
           --duration 120

Environment variables (override CLI defaults):
    GENOVA_LOAD_TEST_HOST       Target host URL
    GENOVA_LOAD_TEST_USERS      Number of concurrent users
    GENOVA_LOAD_TEST_SPAWN_RATE User spawn rate per second
    GENOVA_LOAD_TEST_DURATION   Test duration in seconds
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Locust test classes — these are picked up when this file is used as a
# locustfile (i.e. `locust -f scripts/ci/load_test.py`).
# ---------------------------------------------------------------------------
try:
    from locust import HttpUser, between, task

    class GenovaUser(HttpUser):
        """Simulates a typical consumer of the Genova API."""

        wait_time = between(0.5, 2.0)

        # ── Health / readiness (lightweight, high frequency) ─────────
        @task(5)
        def health(self):
            self.client.get("/health", name="/health")

        # ── Embedding request ────────────────────────────────────────
        @task(3)
        def embed(self):
            self.client.post(
                "/embed",
                json={"sequence": "ATCGATCG" * 16},
                name="/embed",
            )

        # ── Variant prediction ───────────────────────────────────────
        @task(2)
        def predict_variant(self):
            self.client.post(
                "/predict_variant",
                json={
                    "sequence": "ATCGATCG" * 32,
                    "variant_position": 64,
                    "ref_allele": "A",
                    "alt_allele": "G",
                },
                name="/predict_variant",
            )

except ImportError:
    # Locust is not installed — the standalone runner below will still work
    # by invoking locust as a subprocess.
    pass


# ---------------------------------------------------------------------------
# Standalone headless runner with result parsing
# ---------------------------------------------------------------------------

def _parse_stats_csv(csv_path: str) -> dict:
    """Parse Locust stats CSV into a summary dict."""
    summary: dict = {}
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Name", row.get("name", "unknown"))
                summary[name] = {
                    "method": row.get("Type", row.get("type", "")),
                    "requests": int(row.get("Request Count", row.get("# Requests", 0))),
                    "failures": int(row.get("Failure Count", row.get("# Failures", 0))),
                    "median_ms": row.get("Median Response Time", row.get("50%", "N/A")),
                    "p95_ms": row.get("95%", "N/A"),
                    "p99_ms": row.get("99%", "N/A"),
                    "avg_ms": row.get("Average Response Time", row.get("Average", "N/A")),
                    "max_ms": row.get("Max Response Time", row.get("Max", "N/A")),
                    "rps": row.get("Requests/s", "N/A"),
                }
    except FileNotFoundError:
        summary["error"] = f"Stats file not found: {csv_path}"

    return summary


def _print_summary(summary: dict) -> None:
    """Print a human-readable summary table."""
    print()
    header = (
        f"{'Endpoint':<25} {'Reqs':>6} {'Fail':>5} {'Avg':>8} "
        f"{'P95':>8} {'P99':>8} {'RPS':>8}"
    )
    print(header)
    print("-" * len(header))

    for name, stats in summary.items():
        if name in ("error", "Aggregated"):
            continue
        print(
            f"{name:<25} {stats['requests']:>6} {stats['failures']:>5} "
            f"{stats['avg_ms']:>8} {stats['p95_ms']:>8} {stats['p99_ms']:>8} "
            f"{stats['rps']:>8}"
        )

    # Aggregated row
    if "Aggregated" in summary:
        s = summary["Aggregated"]
        print("-" * len(header))
        print(
            f"{'TOTAL':<25} {s['requests']:>6} {s['failures']:>5} "
            f"{s['avg_ms']:>8} {s['p95_ms']:>8} {s['p99_ms']:>8} "
            f"{s['rps']:>8}"
        )

    print()


def run_headless(
    host: str,
    users: int,
    spawn_rate: int,
    duration: int,
    output_dir: str | None = None,
) -> int:
    """Run Locust in headless mode and parse results.

    Returns 0 on success, 1 on failure.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="genova-load-")

    csv_prefix = os.path.join(output_dir, "results")
    html_report = os.path.join(output_dir, "report.html")

    cmd = [
        sys.executable, "-m", "locust",
        "-f", __file__,
        "--host", host,
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", f"{duration}s",
        "--headless",
        "--csv", csv_prefix,
        "--html", html_report,
    ]

    print(f"Starting load test: {users} users, spawn rate {spawn_rate}/s, "
          f"duration {duration}s")
    print(f"Target: {host}")
    print(f"Output: {output_dir}")
    print()

    result = subprocess.run(cmd, capture_output=False)

    # Parse results
    stats_csv = f"{csv_prefix}_stats.csv"
    summary = _parse_stats_csv(stats_csv)

    if "error" not in summary:
        _print_summary(summary)

        # Save JSON summary
        json_path = os.path.join(output_dir, "load-test-summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {json_path}")

        # Check failure rate
        agg = summary.get("Aggregated", {})
        total_reqs = agg.get("requests", 0)
        total_fails = agg.get("failures", 0)
        if total_reqs > 0:
            fail_rate = total_fails / total_reqs
            print(f"Failure rate: {fail_rate:.2%}")
            if fail_rate > 0.05:
                print("WARN: Failure rate exceeds 5% threshold")
                return 1
    else:
        print(f"Error: {summary['error']}")
        return 1

    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Genova load test (headless)")
    parser.add_argument(
        "--host",
        default=os.environ.get("GENOVA_LOAD_TEST_HOST", "http://localhost:8000"),
        help="Target host URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=int(os.environ.get("GENOVA_LOAD_TEST_USERS", "10")),
        help="Number of concurrent simulated users (default: 10)",
    )
    parser.add_argument(
        "--spawn-rate",
        type=int,
        default=int(os.environ.get("GENOVA_LOAD_TEST_SPAWN_RATE", "2")),
        help="User spawn rate per second (default: 2)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=int(os.environ.get("GENOVA_LOAD_TEST_DURATION", "60")),
        help="Test duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files (default: temp dir)",
    )

    args = parser.parse_args()
    return run_headless(
        host=args.host,
        users=args.users,
        spawn_rate=args.spawn_rate,
        duration=args.duration,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
