"""Motif visualization utilities for Genova.

Provides sequence logos, motif occurrence heatmaps, and comparison plots
for discovered versus known motifs.  Uses matplotlib only -- no external
logo library required.

Example::

    from genova.motif.visualization import plot_sequence_logo

    fig = plot_sequence_logo(pwm, title="CTCF motif", save_path="logo.png")
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
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.transforms as mtransforms
    from matplotlib.figure import Figure
    from matplotlib.textpath import TextPath
    from matplotlib.patches import PathPatch
    from matplotlib.font_manager import FontProperties
except ImportError:
    _MPL_AVAILABLE = False

_SNS_AVAILABLE = True
try:
    import seaborn as sns
except ImportError:
    _SNS_AVAILABLE = False


def _check_matplotlib() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for motif visualization. "
            "Install with: pip install matplotlib"
        )


# Nucleotide colour palette (colour-blind friendly)
NUCLEOTIDE_COLORS: Dict[str, str] = {
    "A": "#2ca02c",  # green
    "C": "#1f77b4",  # blue
    "G": "#ff7f0e",  # orange
    "T": "#d62728",  # red
}

NUCLEOTIDE_ORDER = ["A", "C", "G", "T"]


# ---------------------------------------------------------------------------
# Information content helpers
# ---------------------------------------------------------------------------


def _information_content(
    pwm: np.ndarray,
    background: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute per-position information content (bits).

    Args:
        pwm: ``(L, 4)`` frequency matrix (rows sum to 1).
        background: ``(4,)`` background frequencies.  Defaults to uniform.

    Returns:
        ``(L,)`` array of information content in bits.
    """
    if background is None:
        background = np.full(4, 0.25)

    ic = np.zeros(pwm.shape[0])
    for pos in range(pwm.shape[0]):
        for j in range(4):
            freq = pwm[pos, j]
            if freq > 1e-12:
                ic[pos] += freq * np.log2(freq / background[j])
    return np.maximum(ic, 0.0)


def _height_matrix(
    pwm: np.ndarray,
    background: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute letter heights for a sequence logo.

    Height of letter *j* at position *i* = freq[i,j] * IC[i].

    Args:
        pwm: ``(L, 4)`` frequency matrix.
        background: Background frequencies.

    Returns:
        ``(L, 4)`` matrix of letter heights.
    """
    ic = _information_content(pwm, background)
    return pwm * ic[:, np.newaxis]


# ---------------------------------------------------------------------------
# Sequence logo
# ---------------------------------------------------------------------------


def plot_sequence_logo(
    pwm: np.ndarray,
    *,
    title: str = "Sequence Logo",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    background: Optional[np.ndarray] = None,
    ax: Optional[Any] = None,
) -> Figure:
    """Plot a sequence logo from a Position Weight Matrix.

    Letters are drawn as scaled text glyphs whose height is proportional
    to the information content contributed by each nucleotide at each
    position.

    Args:
        pwm: ``(L, 4)`` Position Weight Matrix (columns A, C, G, T).
        title: Plot title.
        figsize: Figure size.  Defaults to ``(max(6, L * 0.8), 3)``.
        save_path: If provided, save figure to this path.
        dpi: Resolution.
        background: ``(4,)`` background nucleotide frequencies.
        ax: Optional existing axes to draw on.

    Returns:
        The matplotlib Figure object.
    """
    _check_matplotlib()

    length = pwm.shape[0]
    if figsize is None:
        figsize = (max(6.0, length * 0.8), 3.0)

    heights = _height_matrix(pwm, background)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    fp = FontProperties(family="monospace", weight="bold")

    for pos in range(length):
        # Sort nucleotides by height (smallest on bottom)
        col_heights = [(heights[pos, j], NUCLEOTIDE_ORDER[j]) for j in range(4)]
        col_heights.sort(key=lambda x: x[0])

        y_offset = 0.0
        for h, letter in col_heights:
            if h < 1e-6:
                continue
            _draw_letter(ax, letter, pos, y_offset, h, fp)
            y_offset += h

    ax.set_xlim(-0.5, length - 0.5)
    max_ic = 2.0  # max bits for DNA
    ax.set_ylim(0, max_ic)
    ax.set_xticks(range(length))
    ax.set_xticklabels([str(i + 1) for i in range(length)], fontsize=8)
    ax.set_ylabel("bits", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if own_fig:
        fig.tight_layout()
        if save_path:
            _save_figure(fig, save_path, dpi)

    return fig


def _draw_letter(
    ax: Any,
    letter: str,
    x: float,
    y: float,
    height: float,
    fp: Any,
) -> None:
    """Draw a single scaled letter glyph on the axes.

    Uses matplotlib's TextPath for glyph rendering with precise scaling.
    """
    color = NUCLEOTIDE_COLORS.get(letter, "#333333")

    # Use TextPath for vector rendering
    tp = TextPath((0, 0), letter, size=1.0, prop=fp)
    bbox = tp.get_extents()

    # Scale to fit within [x-0.4, x+0.4] horizontally and [y, y+height] vertically
    letter_width = bbox.width if bbox.width > 0 else 0.6
    letter_height = bbox.height if bbox.height > 0 else 1.0

    sx = 0.8 / letter_width  # target width = 0.8
    sy = height / letter_height

    # Transform: scale then translate
    transform = (
        mtransforms.Affine2D()
        .scale(sx, sy)
        .translate(x - 0.4, y)
        + ax.transData
    )

    patch = PathPatch(tp, facecolor=color, edgecolor="none", transform=transform)
    ax.add_patch(patch)


# ---------------------------------------------------------------------------
# Motif occurrence heatmap
# ---------------------------------------------------------------------------


def plot_motif_heatmap(
    occurrences: np.ndarray,
    motif_names: Optional[List[str]] = None,
    region_labels: Optional[List[str]] = None,
    *,
    title: str = "Motif Occurrence Heatmap",
    figsize: Tuple[float, float] = (14, 8),
    cmap: str = "YlOrRd",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    annotate: bool = True,
) -> Figure:
    """Plot a heatmap of motif occurrences across genomic regions.

    Args:
        occurrences: ``(n_motifs, n_regions)`` count or score matrix.
        motif_names: Names for each motif (y-axis labels).
        region_labels: Names for each region (x-axis labels).
        title: Plot title.
        figsize: Figure size.
        cmap: Colormap.
        save_path: Output file path.
        dpi: Resolution.
        annotate: If ``True`` and the matrix is small enough, annotate
            cells with values.

    Returns:
        The matplotlib Figure object.
    """
    _check_matplotlib()

    n_motifs, n_regions = occurrences.shape

    fig, ax = plt.subplots(figsize=figsize)

    if _SNS_AVAILABLE and n_motifs <= 50 and n_regions <= 50:
        sns.heatmap(
            occurrences,
            ax=ax,
            cmap=cmap,
            annot=annotate and (n_motifs * n_regions <= 200),
            fmt=".0f" if occurrences.dtype in (np.int32, np.int64) else ".2f",
            xticklabels=region_labels if region_labels else True,
            yticklabels=motif_names if motif_names else True,
            linewidths=0.5,
        )
    else:
        im = ax.imshow(occurrences, aspect="auto", cmap=cmap)
        fig.colorbar(im, ax=ax, shrink=0.7)

        if motif_names and n_motifs <= 60:
            ax.set_yticks(range(n_motifs))
            ax.set_yticklabels(motif_names, fontsize=max(5, 10 - n_motifs // 10))
        if region_labels and n_regions <= 60:
            ax.set_xticks(range(n_regions))
            ax.set_xticklabels(
                region_labels,
                rotation=45,
                ha="right",
                fontsize=max(5, 10 - n_regions // 10),
            )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Genomic region")
    ax.set_ylabel("Motif")

    fig.tight_layout()

    if save_path:
        _save_figure(fig, save_path, dpi)

    return fig


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------


def plot_motif_comparison(
    discovered_pwms: List[np.ndarray],
    known_pwms: List[np.ndarray],
    discovered_names: Optional[List[str]] = None,
    known_names: Optional[List[str]] = None,
    similarities: Optional[List[float]] = None,
    *,
    title: str = "Discovered vs Known Motifs",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> Figure:
    """Plot discovered motifs alongside their best-matching known motifs.

    Each row shows a pair: the discovered motif logo on the left and the
    known motif logo on the right, with the similarity score annotated.

    Args:
        discovered_pwms: List of ``(L, 4)`` PWMs for discovered motifs.
        known_pwms: List of ``(L, 4)`` PWMs for matching known motifs.
        discovered_names: Optional names for discovered motifs.
        known_names: Optional names for known motifs.
        similarities: Optional similarity scores for each pair.
        title: Plot super-title.
        figsize: Figure size.  Defaults based on number of pairs.
        save_path: Output file path.
        dpi: Resolution.

    Returns:
        The matplotlib Figure object.

    Raises:
        ValueError: If list lengths are inconsistent.
    """
    _check_matplotlib()

    n_pairs = len(discovered_pwms)
    if len(known_pwms) != n_pairs:
        raise ValueError(
            f"discovered_pwms ({n_pairs}) and known_pwms ({len(known_pwms)}) "
            "must have the same length."
        )

    if n_pairs == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No motif pairs to compare", ha="center", va="center")
        ax.axis("off")
        return fig

    if figsize is None:
        figsize = (14, max(3, n_pairs * 2.5))

    fig, axes = plt.subplots(n_pairs, 2, figsize=figsize, squeeze=False)

    for i in range(n_pairs):
        # Discovered (left)
        d_name = discovered_names[i] if discovered_names and i < len(discovered_names) else f"Discovered {i + 1}"
        plot_sequence_logo(discovered_pwms[i], title=d_name, ax=axes[i, 0])

        # Known (right)
        k_name = known_names[i] if known_names and i < len(known_names) else f"Known {i + 1}"
        plot_sequence_logo(known_pwms[i], title=k_name, ax=axes[i, 1])

        # Annotate similarity
        if similarities and i < len(similarities):
            axes[i, 1].annotate(
                f"sim={similarities[i]:.3f}",
                xy=(1.02, 0.5),
                xycoords="axes fraction",
                fontsize=9,
                fontweight="bold",
                va="center",
            )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path:
        _save_figure(fig, save_path, dpi)

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_figure(fig: Figure, path: Union[str, Path], dpi: int = 150) -> None:
    """Save a figure to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = path.suffix.lstrip(".").lower() or "png"
    fig.savefig(str(path), format=fmt, dpi=dpi, bbox_inches="tight")
    logger.info("Figure saved to {}", path)
    plt.close(fig)
