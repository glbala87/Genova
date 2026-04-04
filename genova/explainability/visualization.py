"""Visualization utilities for Genova explainability.

Provides publication-quality plots for sequence importance, nucleotide-level
attributions, motif contributions, and variant effects. All functions support
saving to PNG or SVG format.

Example::

    from genova.explainability.visualization import plot_sequence_importance

    plot_sequence_importance(
        shap_values=shap_values,
        tokens=tokens,
        save_path="importance.png",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from loguru import logger

# Lazy matplotlib import to avoid backend issues at module level.
_MPL_AVAILABLE = True
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
except ImportError:
    _MPL_AVAILABLE = False


def _check_matplotlib() -> None:
    """Raise ImportError if matplotlib is not available."""
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


# Nucleotide colour palette (colour-blind friendly)
NUCLEOTIDE_COLORS: Dict[str, str] = {
    "A": "#2ca02c",  # green
    "C": "#1f77b4",  # blue
    "G": "#ff7f0e",  # orange
    "T": "#d62728",  # red
    "N": "#7f7f7f",  # grey
}


# ---------------------------------------------------------------------------
# Sequence importance heatmap
# ---------------------------------------------------------------------------


def plot_sequence_importance(
    shap_values: np.ndarray,
    tokens: List[str],
    *,
    title: str = "Sequence Importance",
    figsize: Tuple[float, float] = (16, 3),
    cmap: str = "RdBu_r",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    max_display: int = 200,
    show_colorbar: bool = True,
) -> Figure:
    """Plot a heatmap of per-token importance values.

    Args:
        shap_values: 1-D array of SHAP or attribution values per token.
        tokens: List of token strings (same length as *shap_values*).
        title: Plot title.
        figsize: Figure size ``(width, height)`` in inches.
        cmap: Matplotlib colormap name.
        save_path: If provided, save figure to this path (PNG or SVG).
        dpi: Resolution for raster output.
        max_display: Maximum number of tokens to display (truncates).
        show_colorbar: Whether to add a colour bar.

    Returns:
        The matplotlib Figure object.
    """
    _check_matplotlib()

    n = min(len(shap_values), len(tokens), max_display)
    sv = shap_values[:n]
    tok = tokens[:n]

    fig, ax = plt.subplots(figsize=figsize)

    # Reshape for imshow: (1, L)
    data = sv.reshape(1, -1)
    vmax = max(abs(sv.max()), abs(sv.min()), 1e-8)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    # Token labels
    if n <= 100:
        ax.set_xticks(range(n))
        ax.set_xticklabels(tok, rotation=90, fontsize=max(4, 8 - n // 20))
    else:
        step = max(1, n // 50)
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels(
            [tok[i] for i in range(0, n, step)],
            rotation=90,
            fontsize=6,
        )

    ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Token position")

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Attribution value", fontsize=9)

    fig.tight_layout()

    if save_path:
        _save_figure(fig, save_path, dpi)

    return fig


# ---------------------------------------------------------------------------
# Nucleotide-level attribution plot
# ---------------------------------------------------------------------------


def plot_attribution(
    attributions: np.ndarray,
    sequence: str,
    *,
    title: str = "Nucleotide Attribution",
    figsize: Tuple[float, float] = (16, 4),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    max_display: int = 200,
    bar_width: float = 0.8,
) -> Figure:
    """Plot nucleotide-level attribution as a coloured bar chart.

    Each bar is coloured by the nucleotide identity and its height
    represents the attribution magnitude (positive = up, negative = down).

    Args:
        attributions: 1-D array of attribution values, one per nucleotide.
        sequence: DNA sequence string (same length as attributions).
        title: Plot title.
        figsize: Figure size.
        save_path: Output file path.
        dpi: Resolution.
        max_display: Maximum nucleotides to display.
        bar_width: Bar width.

    Returns:
        The matplotlib Figure object.
    """
    _check_matplotlib()

    n = min(len(attributions), len(sequence), max_display)
    attr = attributions[:n]
    seq = sequence[:n].upper()

    fig, ax = plt.subplots(figsize=figsize)

    colors = [NUCLEOTIDE_COLORS.get(nt, "#7f7f7f") for nt in seq]
    positions = np.arange(n)
    ax.bar(positions, attr, width=bar_width, color=colors, edgecolor="none")

    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")

    # X-axis labels
    if n <= 100:
        ax.set_xticks(positions)
        ax.set_xticklabels(list(seq), fontsize=max(4, 8 - n // 20), fontfamily="monospace")
    else:
        step = max(1, n // 50)
        ax.set_xticks(positions[::step])
        ax.set_xticklabels(
            [seq[i] for i in range(0, n, step)],
            fontsize=6,
            fontfamily="monospace",
        )

    ax.set_xlabel("Position")
    ax.set_ylabel("Attribution")
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Legend for nucleotides
    legend_patches = [
        mpatches.Patch(color=color, label=nt)
        for nt, color in NUCLEOTIDE_COLORS.items()
        if nt != "N"
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, ncol=4)

    fig.tight_layout()

    if save_path:
        _save_figure(fig, save_path, dpi)

    return fig


# ---------------------------------------------------------------------------
# Variant effect comparison plot
# ---------------------------------------------------------------------------


def plot_variant_effect(
    ref_attributions: np.ndarray,
    alt_attributions: np.ndarray,
    ref_sequence: str,
    alt_sequence: str,
    *,
    title: str = "Variant Effect",
    figsize: Tuple[float, float] = (16, 8),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    max_display: int = 200,
    highlight_variant_pos: Optional[int] = None,
) -> Figure:
    """Plot reference vs alternate attributions to visualize variant effects.

    Creates a three-panel figure:
    1. Reference sequence attributions
    2. Alternate sequence attributions
    3. Difference (alt - ref)

    Args:
        ref_attributions: Attribution values for reference sequence.
        alt_attributions: Attribution values for alternate sequence.
        ref_sequence: Reference DNA sequence.
        alt_sequence: Alternate DNA sequence.
        title: Overall plot title.
        figsize: Figure size.
        save_path: Output file path.
        dpi: Resolution.
        max_display: Maximum positions to display.
        highlight_variant_pos: If provided, highlight this position
            (0-based) with a vertical line.

    Returns:
        The matplotlib Figure object.
    """
    _check_matplotlib()

    min_len = min(
        len(ref_attributions), len(alt_attributions),
        len(ref_sequence), len(alt_sequence),
        max_display,
    )

    ref_attr = ref_attributions[:min_len]
    alt_attr = alt_attributions[:min_len]
    ref_seq = ref_sequence[:min_len].upper()
    alt_seq = alt_sequence[:min_len].upper()
    diff = alt_attr - ref_attr

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    positions = np.arange(min_len)

    # Panel 1: Reference
    ref_colors = [NUCLEOTIDE_COLORS.get(nt, "#7f7f7f") for nt in ref_seq]
    axes[0].bar(positions, ref_attr, color=ref_colors, width=0.8, edgecolor="none")
    axes[0].set_ylabel("Reference")
    axes[0].axhline(y=0, color="black", linewidth=0.5)

    # Panel 2: Alternate
    alt_colors = [NUCLEOTIDE_COLORS.get(nt, "#7f7f7f") for nt in alt_seq]
    axes[1].bar(positions, alt_attr, color=alt_colors, width=0.8, edgecolor="none")
    axes[1].set_ylabel("Alternate")
    axes[1].axhline(y=0, color="black", linewidth=0.5)

    # Panel 3: Difference
    diff_colors = ["#d62728" if d > 0 else "#1f77b4" for d in diff]
    axes[2].bar(positions, diff, color=diff_colors, width=0.8, edgecolor="none")
    axes[2].set_ylabel("Alt - Ref")
    axes[2].set_xlabel("Position")
    axes[2].axhline(y=0, color="black", linewidth=0.5)

    # Highlight variant position
    if highlight_variant_pos is not None and 0 <= highlight_variant_pos < min_len:
        for ax in axes:
            ax.axvline(
                x=highlight_variant_pos,
                color="red",
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
            )

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        _save_figure(fig, save_path, dpi)

    return fig


# ---------------------------------------------------------------------------
# Motif contribution visualization
# ---------------------------------------------------------------------------


def plot_motif_contributions(
    motif_scores: Dict[str, float],
    *,
    title: str = "Motif Contributions",
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    top_k: int = 20,
    horizontal: bool = True,
) -> Figure:
    """Visualize the contribution scores of identified motifs.

    Args:
        motif_scores: Dictionary mapping motif name/sequence to contribution score.
        title: Plot title.
        figsize: Figure size.
        save_path: Output file path.
        dpi: Resolution.
        top_k: Number of top motifs to display.
        horizontal: If True, use horizontal bar chart.

    Returns:
        The matplotlib Figure object.
    """
    _check_matplotlib()

    # Sort by absolute value and take top_k
    sorted_motifs = sorted(motif_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    sorted_motifs = sorted_motifs[:top_k]

    # Reverse for display (top at the top)
    names = [m[0] for m in reversed(sorted_motifs)]
    values = [m[1] for m in reversed(sorted_motifs)]

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]

    if horizontal:
        bars = ax.barh(range(len(names)), values, color=colors, edgecolor="none")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontfamily="monospace", fontsize=9)
        ax.set_xlabel("Contribution Score")
        ax.axvline(x=0, color="black", linewidth=0.5)
    else:
        bars = ax.bar(range(len(names)), values, color=colors, edgecolor="none")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontfamily="monospace", fontsize=9)
        ax.set_ylabel("Contribution Score")
        ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_title(title, fontsize=12, fontweight="bold")

    # Legend
    pos_patch = mpatches.Patch(color="#d62728", label="Positive")
    neg_patch = mpatches.Patch(color="#1f77b4", label="Negative")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9)

    fig.tight_layout()

    if save_path:
        _save_figure(fig, save_path, dpi)

    return fig


# ---------------------------------------------------------------------------
# Attention heatmap
# ---------------------------------------------------------------------------


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    tokens: Optional[List[str]] = None,
    *,
    title: str = "Attention Weights",
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = "viridis",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    layer: Optional[int] = None,
    head: Optional[int] = None,
) -> Figure:
    """Plot attention weights as a heatmap.

    Args:
        attention_weights: 2-D array of shape ``(L, L)`` or ``(H, L, L)``.
            If 3-D, averages over heads unless *head* is specified.
        tokens: Optional token labels for axes.
        title: Plot title.
        figsize: Figure size.
        cmap: Colormap.
        save_path: Output file path.
        dpi: Resolution.
        layer: Layer index for title annotation.
        head: Specific attention head to plot. If None, averages over heads.

    Returns:
        The matplotlib Figure object.
    """
    _check_matplotlib()

    # Handle multi-head input
    attn = attention_weights
    if attn.ndim == 3:
        if head is not None and 0 <= head < attn.shape[0]:
            attn = attn[head]
            title_suffix = f" (Layer {layer}, Head {head})" if layer is not None else f" (Head {head})"
        else:
            attn = attn.mean(axis=0)
            title_suffix = f" (Layer {layer}, avg heads)" if layer is not None else " (avg heads)"
    else:
        title_suffix = f" (Layer {layer})" if layer is not None else ""

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attn, cmap=cmap, aspect="auto")

    if tokens is not None and len(tokens) <= 50:
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7, fontfamily="monospace")
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=7, fontfamily="monospace")
    else:
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")

    ax.set_title(f"{title}{title_suffix}", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    if save_path:
        _save_figure(fig, save_path, dpi)

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_figure(fig: Figure, path: Union[str, Path], dpi: int = 150) -> None:
    """Save a figure to disk, inferring format from extension.

    Args:
        fig: Matplotlib figure.
        path: Output path (supports .png, .svg, .pdf).
        dpi: Resolution for raster formats.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = path.suffix.lstrip(".").lower() or "png"
    fig.savefig(str(path), format=fmt, dpi=dpi, bbox_inches="tight")
    logger.info("Figure saved to {}", path)
    plt.close(fig)
