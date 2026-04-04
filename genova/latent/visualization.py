"""Latent space visualisation for genomic embeddings.

Publication-quality UMAP scatter plots, cluster boundary visualisations,
embedding trajectory plots along chromosomes, and feature-space overviews.

All functions return ``matplotlib.figure.Figure`` objects and optionally
save to disk.

Example::

    from genova.latent.visualization import plot_umap, plot_clusters

    fig = plot_umap(reduced_2d, labels=annotations, save_path="umap.png")
    fig = plot_clusters(reduced_2d, cluster_ids, save_path="clusters.pdf")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Matplotlib / seaborn with non-interactive backend guard
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.figure import Figure
from loguru import logger


# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

_DEFAULT_PALETTE = "tab20"
_DEFAULT_DPI = 300
_DEFAULT_FIGSIZE = (8, 6)

# Genomic feature colour map (consistent across plots)
_FEATURE_COLORS: Dict[str, str] = {
    "promoter": "#E41A1C",
    "enhancer": "#FF7F00",
    "exon": "#377EB8",
    "intron": "#4DAF4A",
    "intergenic": "#984EA3",
    "repeat": "#A65628",
    "utr5": "#F781BF",
    "utr3": "#999999",
    "cpg_island": "#66C2A5",
    "unknown": "#CCCCCC",
}


def _apply_pub_style() -> None:
    """Apply publication-quality matplotlib style settings."""
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": _DEFAULT_DPI,
            "savefig.dpi": _DEFAULT_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def _save_fig(fig: Figure, save_path: Optional[Union[str, Path]]) -> None:
    """Optionally save a figure to disk."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=_DEFAULT_DPI, bbox_inches="tight")
        logger.info("Figure saved to {}", path)


# ---------------------------------------------------------------------------
# UMAP scatter plot
# ---------------------------------------------------------------------------


def plot_umap(
    reduced: np.ndarray,
    labels: Optional[Union[np.ndarray, Sequence[str]]] = None,
    label_names: Optional[Dict[int, str]] = None,
    colors: Optional[Dict[str, str]] = None,
    title: str = "UMAP Embedding Space",
    point_size: float = 3.0,
    alpha: float = 0.6,
    figsize: Tuple[int, int] = _DEFAULT_FIGSIZE,
    save_path: Optional[Union[str, Path]] = None,
    show_legend: bool = True,
) -> Figure:
    """UMAP scatter plot coloured by genomic feature type.

    Args:
        reduced: ``(N, 2)`` or ``(N, 3)`` array of reduced coordinates.
        labels: Per-point labels.  Integer array or string sequence.
        label_names: Mapping from integer label to display name.
        colors: Mapping from label name to hex colour.  Defaults to
            :data:`_FEATURE_COLORS`.
        title: Plot title.
        point_size: Marker size.
        alpha: Marker opacity.
        figsize: Figure dimensions in inches.
        save_path: If provided, save the figure here.
        show_legend: Whether to display the legend.

    Returns:
        The matplotlib ``Figure`` object.
    """
    _apply_pub_style()
    colors = colors or _FEATURE_COLORS

    fig, ax = plt.subplots(figsize=figsize)

    if labels is None:
        ax.scatter(reduced[:, 0], reduced[:, 1], s=point_size, alpha=alpha, c="#377EB8")
    else:
        labels_arr = np.asarray(labels)
        unique_labels = sorted(set(labels_arr))

        for lbl in unique_labels:
            mask = labels_arr == lbl
            name = str(lbl) if label_names is None else label_names.get(lbl, str(lbl))
            color = colors.get(str(name), colors.get(name, None))
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                s=point_size,
                alpha=alpha,
                label=name,
                c=color,
            )

        if show_legend:
            ax.legend(
                markerscale=3,
                frameon=True,
                framealpha=0.8,
                loc="best",
                ncol=max(1, len(unique_labels) // 10),
            )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    sns.despine(ax=ax)

    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Cluster visualisation
# ---------------------------------------------------------------------------


def plot_clusters(
    reduced: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_names: Optional[Dict[int, str]] = None,
    title: str = "Cluster Assignments",
    point_size: float = 3.0,
    alpha: float = 0.6,
    show_centroids: bool = True,
    figsize: Tuple[int, int] = _DEFAULT_FIGSIZE,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Scatter plot coloured by cluster assignment with optional centroids.

    Args:
        reduced: ``(N, 2)`` array of reduced coordinates.
        cluster_labels: ``(N,)`` integer cluster assignments.
        cluster_names: Optional mapping from cluster id to display name.
        title: Plot title.
        point_size: Marker size.
        alpha: Marker opacity.
        show_centroids: Plot cluster centroids as large markers.
        figsize: Figure dimensions.
        save_path: Optional save path.

    Returns:
        The matplotlib ``Figure`` object.
    """
    _apply_pub_style()

    fig, ax = plt.subplots(figsize=figsize)

    unique = sorted(set(cluster_labels) - {-1})
    palette = sns.color_palette(_DEFAULT_PALETTE, n_colors=max(len(unique), 1))

    # Noise points (DBSCAN label = -1)
    noise_mask = cluster_labels == -1
    if noise_mask.any():
        ax.scatter(
            reduced[noise_mask, 0],
            reduced[noise_mask, 1],
            s=point_size * 0.5,
            alpha=0.2,
            c="#CCCCCC",
            label="Noise",
        )

    for idx, cid in enumerate(unique):
        mask = cluster_labels == cid
        name = str(cid) if cluster_names is None else cluster_names.get(cid, str(cid))
        color = palette[idx % len(palette)]
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=point_size,
            alpha=alpha,
            label=f"Cluster {name}",
            color=color,
        )

        if show_centroids:
            cx = reduced[mask, 0].mean()
            cy = reduced[mask, 1].mean()
            ax.scatter(cx, cy, s=100, c=[color], edgecolors="black", linewidths=1.5, zorder=5)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.legend(
        markerscale=3,
        frameon=True,
        framealpha=0.8,
        loc="best",
        ncol=max(1, len(unique) // 10),
    )
    sns.despine(ax=ax)

    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Embedding trajectory along chromosome
# ---------------------------------------------------------------------------


def plot_embedding_trajectory(
    embeddings: np.ndarray,
    positions: np.ndarray,
    chrom: str = "",
    components: Tuple[int, int] = (0, 1),
    color_by_position: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Plot embedding trajectory as a path coloured by genomic position.

    Useful for visualising how embeddings change along a chromosome.

    Args:
        embeddings: ``(N, D)`` embeddings for consecutive genomic windows.
        positions: ``(N,)`` genomic start positions of each window.
        chrom: Chromosome name for the title.
        components: Which two embedding dimensions to plot (if D > 2).
        color_by_position: Colour the trajectory by genomic position.
        title: Custom title.
        figsize: Figure dimensions.
        save_path: Optional save path.

    Returns:
        The matplotlib ``Figure`` object.
    """
    _apply_pub_style()

    # Select two components
    if embeddings.shape[1] > 2:
        x = embeddings[:, components[0]]
        y = embeddings[:, components[1]]
        xlabel = f"Component {components[0]}"
        ylabel = f"Component {components[1]}"
    else:
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        xlabel = "Component 0"
        ylabel = "Component 1"

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left panel: trajectory in embedding space
    ax0 = axes[0]
    if color_by_position:
        scatter = ax0.scatter(x, y, c=positions, cmap="viridis", s=5, alpha=0.7)
        plt.colorbar(scatter, ax=ax0, label="Genomic position")
    else:
        ax0.plot(x, y, linewidth=0.5, alpha=0.5, color="#377EB8")
        ax0.scatter(x, y, s=3, alpha=0.6, color="#377EB8")
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    ax0.set_title(f"Embedding trajectory{f' ({chrom})' if chrom else ''}")
    sns.despine(ax=ax0)

    # Right panel: component values along chromosome
    ax1 = axes[1]
    ax1.plot(positions, x, label=xlabel, alpha=0.7, linewidth=0.8)
    ax1.plot(positions, y, label=ylabel, alpha=0.7, linewidth=0.8)
    ax1.set_xlabel("Genomic position")
    ax1.set_ylabel("Embedding value")
    ax1.set_title(f"Embedding components vs. position{f' ({chrom})' if chrom else ''}")
    ax1.legend()
    sns.despine(ax=ax1)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Feature space overview
# ---------------------------------------------------------------------------


def plot_feature_space(
    reduced: np.ndarray,
    feature_values: np.ndarray,
    feature_name: str = "GC content",
    cmap: str = "coolwarm",
    title: Optional[str] = None,
    point_size: float = 3.0,
    alpha: float = 0.7,
    figsize: Tuple[int, int] = _DEFAULT_FIGSIZE,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Scatter plot of reduced embeddings coloured by a continuous feature.

    Useful for overlaying biological properties (GC content, expression,
    conservation score, etc.) on top of the UMAP.

    Args:
        reduced: ``(N, 2)`` reduced coordinates.
        feature_values: ``(N,)`` continuous values for colour mapping.
        feature_name: Name of the feature for the colour-bar label.
        cmap: Matplotlib colourmap name.
        title: Custom title.
        point_size: Marker size.
        alpha: Marker opacity.
        figsize: Figure dimensions.
        save_path: Optional save path.

    Returns:
        The matplotlib ``Figure`` object.
    """
    _apply_pub_style()

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=feature_values,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(feature_name)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title or f"Feature Space: {feature_name}")
    sns.despine(ax=ax)

    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Multi-panel summary
# ---------------------------------------------------------------------------


def plot_latent_summary(
    reduced: np.ndarray,
    cluster_labels: np.ndarray,
    annotation_labels: Optional[np.ndarray] = None,
    feature_values: Optional[np.ndarray] = None,
    feature_name: str = "GC content",
    title: str = "Latent Space Summary",
    figsize: Tuple[int, int] = (16, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Create a multi-panel summary figure of the latent space.

    Panels:
        1. Cluster assignments
        2. Biological annotations (if provided)
        3. Continuous feature overlay (if provided)

    Args:
        reduced: ``(N, 2)`` reduced coordinates.
        cluster_labels: ``(N,)`` cluster assignments.
        annotation_labels: ``(N,)`` biological annotation strings.
        feature_values: ``(N,)`` continuous feature for panel 3.
        feature_name: Label for the continuous feature.
        title: Overall figure title.
        figsize: Figure dimensions.
        save_path: Optional save path.

    Returns:
        The matplotlib ``Figure`` object.
    """
    _apply_pub_style()

    n_panels = 1 + (annotation_labels is not None) + (feature_values is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # Panel 1: Clusters
    ax = axes[panel_idx]
    unique_clusters = sorted(set(cluster_labels) - {-1})
    palette = sns.color_palette(_DEFAULT_PALETTE, n_colors=max(len(unique_clusters), 1))
    for ci, cid in enumerate(unique_clusters):
        mask = cluster_labels == cid
        ax.scatter(
            reduced[mask, 0], reduced[mask, 1],
            s=2, alpha=0.5, color=palette[ci % len(palette)], label=f"C{cid}",
        )
    ax.set_title("Clusters")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    sns.despine(ax=ax)
    panel_idx += 1

    # Panel 2: Annotations
    if annotation_labels is not None:
        ax = axes[panel_idx]
        unique_annots = sorted(set(annotation_labels))
        for annot in unique_annots:
            mask = annotation_labels == annot
            color = _FEATURE_COLORS.get(str(annot), None)
            ax.scatter(
                reduced[mask, 0], reduced[mask, 1],
                s=2, alpha=0.5, c=color, label=str(annot),
            )
        ax.set_title("Annotations")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(markerscale=4, frameon=True, fontsize=7, loc="best")
        sns.despine(ax=ax)
        panel_idx += 1

    # Panel 3: Continuous feature
    if feature_values is not None:
        ax = axes[panel_idx]
        sc = ax.scatter(
            reduced[:, 0], reduced[:, 1],
            c=feature_values, cmap="coolwarm", s=2, alpha=0.5,
        )
        plt.colorbar(sc, ax=ax, label=feature_name)
        ax.set_title(feature_name)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        sns.despine(ax=ax)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig
