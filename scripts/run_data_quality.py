#!/usr/bin/env python3
"""Generate data quality report for genomic data.

Analyzes a FASTA reference genome and produces a quality report with
GC content, N content, chromosome sizes, and k-mer frequency analysis.

Usage:
    python scripts/run_data_quality.py \\
        --fasta data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa \\
        --output reports/data_quality.md

    python scripts/run_data_quality.py \\
        --fasta data/reference/hg38.fa \\
        --output reports/data_quality.html \\
        --format html
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_quality")


# ---------------------------------------------------------------------------
# FASTA analysis
# ---------------------------------------------------------------------------

def analyze_fasta(fasta_path: str) -> Dict[str, Any]:
    """Analyze a FASTA reference genome file.

    Args:
        fasta_path: Path to the FASTA file.

    Returns:
        Dict with analysis results per chromosome and overall statistics.
    """
    try:
        import pyfaidx
    except ImportError:
        logger.error("pyfaidx is required. Install with: pip install pyfaidx")
        sys.exit(1)

    logger.info("Analyzing FASTA: %s", fasta_path)
    start_time = time.time()

    fasta = pyfaidx.Fasta(fasta_path, read_ahead=10_000)
    chromosomes = list(fasta.keys())

    results: Dict[str, Any] = {
        "fasta_path": fasta_path,
        "n_sequences": len(chromosomes),
        "chromosomes": {},
        "overall": {},
    }

    total_length = 0
    total_gc = 0
    total_n = 0
    total_at = 0

    for chrom in chromosomes:
        seq = str(fasta[chrom][:]).upper()
        length = len(seq)

        # Base composition
        count_a = seq.count("A")
        count_t = seq.count("T")
        count_g = seq.count("G")
        count_c = seq.count("C")
        count_n = seq.count("N")
        count_other = length - (count_a + count_t + count_g + count_c + count_n)

        gc_count = count_g + count_c
        at_count = count_a + count_t
        non_n_length = length - count_n

        gc_content = gc_count / non_n_length if non_n_length > 0 else 0.0
        n_fraction = count_n / length if length > 0 else 0.0

        chrom_info = {
            "length": length,
            "length_mb": round(length / 1_000_000, 2),
            "gc_content": round(gc_content, 4),
            "n_fraction": round(n_fraction, 4),
            "n_count": count_n,
            "base_counts": {
                "A": count_a,
                "T": count_t,
                "G": count_g,
                "C": count_c,
                "N": count_n,
                "other": count_other,
            },
        }

        results["chromosomes"][chrom] = chrom_info

        total_length += length
        total_gc += gc_count
        total_n += count_n
        total_at += at_count

        logger.info(
            "  %s: %d bp (%.1f Mb), GC=%.2f%%, N=%.2f%%",
            chrom, length, length / 1e6,
            gc_content * 100, n_fraction * 100,
        )

    # Overall statistics
    total_non_n = total_length - total_n
    overall_gc = total_gc / total_non_n if total_non_n > 0 else 0.0
    overall_n_fraction = total_n / total_length if total_length > 0 else 0.0

    results["overall"] = {
        "total_length": total_length,
        "total_length_gb": round(total_length / 1_000_000_000, 3),
        "gc_content": round(overall_gc, 4),
        "n_fraction": round(overall_n_fraction, 4),
        "n_count": total_n,
        "non_n_length": total_non_n,
    }

    fasta.close()

    elapsed = time.time() - start_time
    results["analysis_time_s"] = round(elapsed, 1)
    logger.info("FASTA analysis completed in %.1f seconds.", elapsed)

    return results


def analyze_kmer_frequencies(
    fasta_path: str,
    k: int = 6,
    sample_chroms: Optional[List[str]] = None,
    max_bases: int = 50_000_000,
) -> Dict[str, Any]:
    """Perform k-mer frequency analysis on a FASTA file.

    Args:
        fasta_path: Path to the FASTA file.
        k: K-mer size.
        sample_chroms: Chromosomes to sample. Default: chr1 and chr22.
        max_bases: Maximum number of bases to process per chromosome.

    Returns:
        Dict with k-mer statistics and top/bottom k-mers.
    """
    try:
        import pyfaidx
    except ImportError:
        logger.error("pyfaidx is required for k-mer analysis.")
        return {}

    logger.info("Running %d-mer frequency analysis...", k)
    start_time = time.time()

    fasta = pyfaidx.Fasta(fasta_path, read_ahead=10_000)
    all_chroms = list(fasta.keys())

    if sample_chroms is None:
        # Default: sample a couple of chromosomes
        candidates = ["chr1", "1", "chr22", "22"]
        sample_chroms = [c for c in candidates if c in all_chroms]
        if not sample_chroms:
            sample_chroms = all_chroms[:2]

    kmer_counts: Counter = Counter()
    total_kmers = 0

    for chrom in sample_chroms:
        if chrom not in fasta:
            continue

        seq = str(fasta[chrom][:max_bases]).upper()
        logger.info("  Counting %d-mers in %s (%d bp)...", k, chrom, len(seq))

        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            if "N" not in kmer:
                kmer_counts[kmer] += 1
                total_kmers += 1

    fasta.close()

    if total_kmers == 0:
        logger.warning("No k-mers found.")
        return {"k": k, "total_kmers": 0}

    # Compute statistics
    n_unique = len(kmer_counts)
    n_possible = 4 ** k
    coverage = n_unique / n_possible if n_possible > 0 else 0.0

    frequencies = list(kmer_counts.values())
    import numpy as np

    freq_array = np.array(frequencies, dtype=float)
    freq_array /= total_kmers  # Normalize to probabilities

    # Shannon entropy
    entropy = -np.sum(freq_array * np.log2(freq_array + 1e-12))
    max_entropy = np.log2(n_possible)
    entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.0

    # Top and bottom k-mers
    top_kmers = kmer_counts.most_common(20)
    bottom_kmers = kmer_counts.most_common()[:-21:-1] if len(kmer_counts) >= 20 else []

    results = {
        "k": k,
        "sampled_chromosomes": sample_chroms,
        "total_kmers": total_kmers,
        "unique_kmers": n_unique,
        "possible_kmers": n_possible,
        "kmer_coverage": round(coverage, 4),
        "shannon_entropy": round(float(entropy), 4),
        "max_entropy": round(float(max_entropy), 4),
        "entropy_ratio": round(float(entropy_ratio), 4),
        "top_20_kmers": [
            {"kmer": kmer, "count": count, "frequency": round(count / total_kmers, 6)}
            for kmer, count in top_kmers
        ],
        "bottom_20_kmers": [
            {"kmer": kmer, "count": count, "frequency": round(count / total_kmers, 6)}
            for kmer, count in bottom_kmers
        ],
    }

    elapsed = time.time() - start_time
    results["analysis_time_s"] = round(elapsed, 1)
    logger.info(
        "K-mer analysis complete: %d unique %d-mers, entropy=%.2f (%.1f%% of max)",
        n_unique, k, entropy, entropy_ratio * 100,
    )

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def format_number(n: int) -> str:
    """Format a large integer with comma separators."""
    return f"{n:,}"


def generate_markdown_report(
    fasta_results: Dict[str, Any],
    kmer_results: Dict[str, Any],
) -> str:
    """Generate a markdown data quality report.

    Args:
        fasta_results: Results from analyze_fasta.
        kmer_results: Results from analyze_kmer_frequencies.

    Returns:
        Markdown string.
    """
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections = []

    # Header
    sections.append("# Genova Data Quality Report\n")
    sections.append(f"**Generated**: {date_str}  ")
    sections.append(f"**FASTA**: `{fasta_results.get('fasta_path', 'N/A')}`\n")

    # Overall summary
    overall = fasta_results.get("overall", {})
    sections.append("## Overall Summary\n")
    sections.append("| Metric | Value |")
    sections.append("|---|---|")
    sections.append(f"| Total sequences | {fasta_results.get('n_sequences', 0)} |")
    sections.append(f"| Total length | {format_number(overall.get('total_length', 0))} bp ({overall.get('total_length_gb', 0)} Gb) |")
    sections.append(f"| GC content | {overall.get('gc_content', 0) * 100:.2f}% |")
    sections.append(f"| N fraction | {overall.get('n_fraction', 0) * 100:.2f}% |")
    sections.append(f"| N bases | {format_number(overall.get('n_count', 0))} |")
    sections.append(f"| Non-N bases | {format_number(overall.get('non_n_length', 0))} |")
    sections.append(f"| Analysis time | {fasta_results.get('analysis_time_s', 0)}s |")
    sections.append("")

    # Chromosome table
    chroms = fasta_results.get("chromosomes", {})
    if chroms:
        sections.append("## Chromosome Details\n")
        sections.append("| Chromosome | Length (Mb) | GC Content (%) | N Fraction (%) | N Bases |")
        sections.append("|---|---|---|---|---|")

        # Sort chromosomes: numeric first, then alpha
        def chrom_sort_key(name: str) -> Tuple[int, str]:
            stripped = name.replace("chr", "")
            try:
                return (0, f"{int(stripped):03d}")
            except ValueError:
                return (1, stripped)

        for chrom in sorted(chroms.keys(), key=chrom_sort_key):
            info = chroms[chrom]
            sections.append(
                f"| {chrom} | {info['length_mb']} | "
                f"{info['gc_content'] * 100:.2f} | "
                f"{info['n_fraction'] * 100:.2f} | "
                f"{format_number(info['n_count'])} |"
            )
        sections.append("")

    # K-mer analysis
    if kmer_results and kmer_results.get("total_kmers", 0) > 0:
        k = kmer_results.get("k", 6)
        sections.append(f"## {k}-mer Frequency Analysis\n")
        sections.append(f"**Sampled chromosomes**: {', '.join(kmer_results.get('sampled_chromosomes', []))}\n")
        sections.append("| Metric | Value |")
        sections.append("|---|---|")
        sections.append(f"| K-mer size | {k} |")
        sections.append(f"| Total k-mers counted | {format_number(kmer_results.get('total_kmers', 0))} |")
        sections.append(f"| Unique k-mers | {format_number(kmer_results.get('unique_kmers', 0))} |")
        sections.append(f"| Possible k-mers | {format_number(kmer_results.get('possible_kmers', 0))} |")
        sections.append(f"| K-mer coverage | {kmer_results.get('kmer_coverage', 0) * 100:.2f}% |")
        sections.append(f"| Shannon entropy | {kmer_results.get('shannon_entropy', 0)} bits |")
        sections.append(f"| Max entropy | {kmer_results.get('max_entropy', 0)} bits |")
        sections.append(f"| Entropy ratio | {kmer_results.get('entropy_ratio', 0) * 100:.2f}% |")
        sections.append("")

        # Top k-mers
        top_kmers = kmer_results.get("top_20_kmers", [])
        if top_kmers:
            sections.append(f"### Most Frequent {k}-mers\n")
            sections.append("| Rank | K-mer | Count | Frequency |")
            sections.append("|---|---|---|---|")
            for i, item in enumerate(top_kmers, 1):
                sections.append(
                    f"| {i} | `{item['kmer']}` | "
                    f"{format_number(item['count'])} | "
                    f"{item['frequency']:.6f} |"
                )
            sections.append("")

        # Bottom k-mers
        bottom_kmers = kmer_results.get("bottom_20_kmers", [])
        if bottom_kmers:
            sections.append(f"### Least Frequent {k}-mers\n")
            sections.append("| Rank | K-mer | Count | Frequency |")
            sections.append("|---|---|---|---|")
            for i, item in enumerate(bottom_kmers, 1):
                sections.append(
                    f"| {i} | `{item['kmer']}` | "
                    f"{format_number(item['count'])} | "
                    f"{item['frequency']:.6f} |"
                )
            sections.append("")

    # Quality assessment
    sections.append("## Quality Assessment\n")

    checks = []
    gc = overall.get("gc_content", 0)
    n_frac = overall.get("n_fraction", 0)

    if 0.35 <= gc <= 0.45:
        checks.append("- [x] GC content within expected range for human genome (35-45%)")
    else:
        checks.append(f"- [ ] GC content ({gc * 100:.1f}%) outside expected range (35-45%)")

    if n_frac < 0.05:
        checks.append("- [x] N fraction below 5% threshold")
    else:
        checks.append(f"- [ ] N fraction ({n_frac * 100:.1f}%) exceeds 5% threshold")

    total_len = overall.get("total_length", 0)
    if total_len > 2_800_000_000:
        checks.append("- [x] Total genome length consistent with human reference (>2.8 Gb)")
    else:
        checks.append(f"- [ ] Total genome length ({total_len / 1e9:.2f} Gb) shorter than expected")

    if kmer_results and kmer_results.get("entropy_ratio", 0) > 0.9:
        checks.append("- [x] K-mer entropy ratio >90% (good sequence diversity)")
    elif kmer_results and kmer_results.get("entropy_ratio", 0) > 0:
        checks.append(
            f"- [ ] K-mer entropy ratio ({kmer_results['entropy_ratio'] * 100:.1f}%) "
            f"may indicate low sequence diversity"
        )

    sections.extend(checks)
    sections.append("")

    return "\n".join(sections)


def generate_html_report(
    fasta_results: Dict[str, Any],
    kmer_results: Dict[str, Any],
) -> str:
    """Generate an HTML data quality report by wrapping the markdown content.

    Args:
        fasta_results: Results from analyze_fasta.
        kmer_results: Results from analyze_kmer_frequencies.

    Returns:
        HTML string.
    """
    md_content = generate_markdown_report(fasta_results, kmer_results)

    # Simple HTML wrapper with basic styling
    html = textwrap.dedent("""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Genova Data Quality Report</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                   max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }
            h1 { color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 10px; }
            h2 { color: #2c3e50; margin-top: 30px; }
            h3 { color: #34495e; }
            table { border-collapse: collapse; width: 100%; margin: 15px 0; }
            th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
            th { background-color: #f8f9fa; font-weight: 600; }
            tr:nth-child(even) { background-color: #f8f9fa; }
            code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px;
                   font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.9em; }
            pre { background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
    """)

    # Convert markdown to simple HTML (basic conversion)
    content = md_content
    # Headers
    for level in range(3, 0, -1):
        prefix = "#" * level
        import re
        content = re.sub(
            rf"^{prefix}\s+(.+)$",
            rf"<h{level}>\1</h{level}>",
            content,
            flags=re.MULTILINE,
        )

    # Bold
    content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)

    # Code
    content = re.sub(r"`(.+?)`", r"<code>\1</code>", content)

    # Tables (already in markdown table format, convert to HTML)
    lines = content.split("\n")
    in_table = False
    html_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if "|---|" in stripped or "|--" in stripped:
                continue  # Skip separator rows
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if not in_table:
                in_table = True
                html_lines.append("<table>")
                html_lines.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
            else:
                html_lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        else:
            if in_table:
                html_lines.append("</table>")
                in_table = False

            # List items
            if stripped.startswith("- [x]"):
                html_lines.append(f"<p>&#9745; {stripped[5:].strip()}</p>")
            elif stripped.startswith("- [ ]"):
                html_lines.append(f"<p>&#9744; {stripped[5:].strip()}</p>")
            elif stripped.startswith("- "):
                html_lines.append(f"<li>{stripped[2:]}</li>")
            elif stripped:
                html_lines.append(f"<p>{stripped}</p>")

    if in_table:
        html_lines.append("</table>")

    html += "\n".join(html_lines)
    html += "\n</body>\n</html>\n"

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate data quality report for genomic data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python scripts/run_data_quality.py \\
              --fasta data/reference/hg38.fa \\
              --output reports/data_quality.md

          python scripts/run_data_quality.py \\
              --fasta data/reference/hg38.fa \\
              --output reports/data_quality.html \\
              --format html

          python scripts/run_data_quality.py \\
              --fasta data/reference/hg38.fa \\
              --output reports/quality.md \\
              --kmer-size 4
        """),
    )

    parser.add_argument(
        "--fasta", type=str, required=True,
        help="Path to the reference genome FASTA file.",
    )
    parser.add_argument(
        "--output", type=str, default="reports/data_quality.md",
        help="Output path for the quality report (default: reports/data_quality.md).",
    )
    parser.add_argument(
        "--format", type=str, default="markdown",
        choices=["markdown", "html"],
        help="Output format: markdown or html (default: markdown).",
    )
    parser.add_argument(
        "--kmer-size", type=int, default=6,
        help="K-mer size for frequency analysis (default: 6).",
    )
    parser.add_argument(
        "--skip-kmer", action="store_true",
        help="Skip k-mer frequency analysis (faster).",
    )
    parser.add_argument(
        "--max-kmer-bases", type=int, default=50_000_000,
        help="Maximum bases per chromosome for k-mer analysis (default: 50M).",
    )

    args = parser.parse_args()

    # Validate FASTA path
    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        # Try resolving symlinks
        if fasta_path.is_symlink():
            logger.error("FASTA symlink broken: %s", fasta_path)
        else:
            logger.error("FASTA file not found: %s", fasta_path)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Genova Data Quality Report")
    logger.info("=" * 70)
    logger.info("FASTA:  %s", fasta_path)
    logger.info("Output: %s", output_path)
    logger.info("Format: %s", args.format)
    logger.info("")

    # Run FASTA analysis
    fasta_results = analyze_fasta(str(fasta_path))

    # Run k-mer analysis
    kmer_results: Dict[str, Any] = {}
    if not args.skip_kmer:
        kmer_results = analyze_kmer_frequencies(
            str(fasta_path),
            k=args.kmer_size,
            max_bases=args.max_kmer_bases,
        )

    # Generate report
    logger.info("Generating %s report...", args.format)

    if args.format == "html":
        report = generate_html_report(fasta_results, kmer_results)
    else:
        report = generate_markdown_report(fasta_results, kmer_results)

    # Write report
    with open(output_path, "w") as fh:
        fh.write(report)
    logger.info("Report written to: %s", output_path)

    # Also save raw results as JSON
    json_path = output_path.with_suffix(".json")
    raw_data = {
        "fasta_analysis": fasta_results,
        "kmer_analysis": kmer_results,
    }
    with open(json_path, "w") as fh:
        json.dump(raw_data, fh, indent=2)
    logger.info("Raw data saved to: %s", json_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
