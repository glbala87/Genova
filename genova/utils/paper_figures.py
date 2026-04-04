"""Publication-quality figure generation for the Genova methods paper.

Generates Nature Methods / Genome Biology style figures with consistent
colour schemes, font sizes, and DPI settings.  All functions produce
matplotlib figures that can be saved as PDF, SVG, or PNG.

Example::

    from genova.utils.paper_figures import create_all_figures

    create_all_figures(output_dir="figures/", dpi=600)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from loguru import logger

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import seaborn as sns

    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False
    logger.warning("matplotlib/seaborn not available; figure generation disabled.")


# ---------------------------------------------------------------------------
# Style constants (Nature Methods / Genome Biology)
# ---------------------------------------------------------------------------

# Colour palette
COLORS = {
    "primary": "#2C3E50",
    "accent1": "#E74C3C",
    "accent2": "#3498DB",
    "accent3": "#2ECC71",
    "accent4": "#F39C12",
    "accent5": "#9B59B6",
    "light_grey": "#BDC3C7",
    "dark_grey": "#7F8C8D",
    "background": "#FAFAFA",
    "genova": "#2C3E50",
    "dnabert": "#E74C3C",
    "enformer": "#3498DB",
    "basenji": "#2ECC71",
    "evo": "#F39C12",
}

MODEL_COLORS = {
    "Genova": COLORS["genova"],
    "DNABERT": COLORS["dnabert"],
    "Enformer": COLORS["enformer"],
    "Basenji": COLORS["basenji"],
    "Evo": COLORS["evo"],
}

# Typography
FONT_SIZES = {
    "title": 12,
    "subtitle": 10,
    "axis_label": 9,
    "tick": 8,
    "legend": 8,
    "annotation": 7,
    "panel_label": 14,
}

# Standard figure sizes (inches) -- Nature Methods column widths
FIGURE_SIZES = {
    "single_column": (3.5, 3.0),
    "one_and_half": (5.5, 4.0),
    "double_column": (7.2, 5.0),
    "full_page": (7.2, 9.0),
}

DEFAULT_DPI = 300


def _set_publication_style() -> None:
    """Configure matplotlib for publication-quality output."""
    if not _HAS_PLOTTING:
        return
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZES["tick"],
        "axes.titlesize": FONT_SIZES["title"],
        "axes.labelsize": FONT_SIZES["axis_label"],
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "legend.fontsize": FONT_SIZES["legend"],
        "figure.dpi": DEFAULT_DPI,
        "savefig.dpi": DEFAULT_DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,  # Editable text in PDF
        "ps.fonttype": 42,
    })


def _add_panel_label(ax: Any, label: str, x: float = -0.12, y: float = 1.08) -> None:
    """Add a panel label (a, b, c, ...) to an axes.

    Args:
        ax: Matplotlib axes.
        label: Panel label string.
        x: X position in axes coordinates.
        y: Y position in axes coordinates.
    """
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=FONT_SIZES["panel_label"],
        fontweight="bold",
        va="top",
    )


# ---------------------------------------------------------------------------
# Figure 1: Model architecture diagram
# ---------------------------------------------------------------------------


def create_architecture_figure(
    save_path: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
) -> Optional[Any]:
    """Generate a programmatic model architecture diagram.

    Creates a schematic showing the Genova architecture: DNA input,
    tokenizer, embedding, transformer/Mamba blocks, and task heads.

    Args:
        save_path: Path to save the figure.
        dpi: Resolution.

    Returns:
        Matplotlib figure or ``None``.
    """
    if not _HAS_PLOTTING:
        return None

    _set_publication_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["double_column"])

    # Define component positions (x_center, y_center, width, height)
    components = [
        ("DNA Sequence\nACGT...TGCA", 0.5, 0.92, 0.35, 0.06, COLORS["light_grey"]),
        ("BPE Tokenizer", 0.5, 0.82, 0.25, 0.05, COLORS["accent4"]),
        ("Genomic Embedding\n+ Positional Encoding", 0.5, 0.72, 0.30, 0.06, COLORS["accent2"]),
        ("Transformer Encoder\n(Multi-Head Attention\n+ Feed-Forward)", 0.28, 0.54, 0.25, 0.12, COLORS["accent1"]),
        ("Mamba Block\n(Selective SSM)", 0.72, 0.54, 0.25, 0.12, COLORS["accent5"]),
        ("Hybrid Fusion Layer", 0.5, 0.38, 0.30, 0.05, COLORS["primary"]),
        ("MLM Head", 0.18, 0.22, 0.18, 0.05, COLORS["accent3"]),
        ("Classification\nHead", 0.42, 0.22, 0.18, 0.06, COLORS["accent3"]),
        ("Variant Effect\nHead", 0.66, 0.22, 0.18, 0.06, COLORS["accent3"]),
        ("Contrastive\nHead", 0.88, 0.22, 0.16, 0.06, COLORS["accent3"]),
    ]

    for text, x, y, w, h, color in components:
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.01",
            facecolor=color,
            edgecolor=COLORS["primary"],
            linewidth=0.8,
            alpha=0.85,
        )
        ax.add_patch(box)
        ax.text(
            x, y, text,
            ha="center", va="center",
            fontsize=FONT_SIZES["annotation"],
            fontweight="bold",
            color="white" if color in [COLORS["primary"], COLORS["accent1"], COLORS["accent5"]] else COLORS["primary"],
            wrap=True,
        )

    # Arrows connecting components
    arrow_style = "Simple,tail_width=2,head_width=6,head_length=4"
    arrows = [
        (0.5, 0.89, 0.5, 0.85),   # DNA -> Tokenizer
        (0.5, 0.795, 0.5, 0.755),  # Tokenizer -> Embedding
        (0.5, 0.69, 0.35, 0.605),  # Embedding -> Transformer
        (0.5, 0.69, 0.65, 0.605),  # Embedding -> Mamba
        (0.35, 0.48, 0.43, 0.41),  # Transformer -> Fusion
        (0.65, 0.48, 0.57, 0.41),  # Mamba -> Fusion
        (0.5, 0.355, 0.18, 0.25),  # Fusion -> MLM
        (0.5, 0.355, 0.42, 0.26),  # Fusion -> Classification
        (0.5, 0.355, 0.66, 0.26),  # Fusion -> Variant
        (0.5, 0.355, 0.86, 0.26),  # Fusion -> Contrastive
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=COLORS["dark_grey"], lw=1.0),
        )

    # Nx label for transformer block
    ax.text(
        0.15, 0.54, r"$\times L$",
        fontsize=FONT_SIZES["subtitle"],
        fontweight="bold",
        color=COLORS["accent1"],
        ha="center",
    )
    ax.text(
        0.88, 0.46, r"$\times L$",
        fontsize=FONT_SIZES["subtitle"],
        fontweight="bold",
        color=COLORS["accent5"],
        ha="center",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0.12, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Genova: Genomic Foundation Model Architecture",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
        pad=10,
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Architecture figure saved to {}", save_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 2: Benchmark comparison
# ---------------------------------------------------------------------------


def create_benchmark_figure(
    results: Dict[str, Dict[str, Dict[str, float]]],
    save_path: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    dpi: int = DEFAULT_DPI,
) -> Optional[Any]:
    """Generate grouped bar charts comparing models across benchmark tasks.

    Args:
        results: Nested dict ``{task: {model: {metric: value}}}``.
        save_path: Path to save figure.
        metrics: Subset of metrics to plot.  Defaults to
            ``["auroc", "auprc", "f1"]``.
        dpi: Resolution.

    Returns:
        Matplotlib figure or ``None``.
    """
    if not _HAS_PLOTTING:
        return None

    _set_publication_style()

    metrics = metrics or ["auroc", "auprc", "f1"]
    tasks = sorted(results.keys())
    models = sorted(
        {m for task_data in results.values() for m in task_data.keys()}
    )

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(3.0 * n_metrics, 4.0),
        squeeze=False,
    )

    for mi, metric in enumerate(metrics):
        ax = axes[0, mi]
        x = np.arange(len(tasks))
        width = 0.8 / max(len(models), 1)

        for ci, model in enumerate(models):
            values = []
            for task in tasks:
                val = results.get(task, {}).get(model, {}).get(metric, 0.0)
                values.append(val)

            offset = (ci - len(models) / 2 + 0.5) * width
            color = MODEL_COLORS.get(model, COLORS["dark_grey"])
            bars = ax.bar(
                x + offset, values,
                width=width,
                label=model if mi == 0 else None,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
            )

            # Add value labels on bars
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.2f}",
                        ha="center", va="bottom",
                        fontsize=5,
                        rotation=90,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [t.replace("_prediction", "").replace("_classification", "").replace("_", "\n")
             for t in tasks],
            fontsize=FONT_SIZES["annotation"],
        )
        ax.set_title(metric.upper(), fontsize=FONT_SIZES["subtitle"], fontweight="bold")
        ax.set_ylim(0, 1.15)
        _add_panel_label(ax, chr(ord("a") + mi))

    if len(models) > 1:
        axes[0, 0].legend(
            fontsize=FONT_SIZES["legend"],
            frameon=False,
            loc="lower left",
        )

    fig.suptitle(
        "Benchmark Comparison",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Benchmark figure saved to {}", save_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 3: Attention / importance visualisation
# ---------------------------------------------------------------------------


def create_attention_figure(
    attention_weights: np.ndarray,
    sequence: str,
    save_path: Optional[str] = None,
    layer: int = -1,
    head: Optional[int] = None,
    max_length: int = 100,
    dpi: int = DEFAULT_DPI,
) -> Optional[Any]:
    """Visualise attention weights or importance scores on a DNA sequence.

    Args:
        attention_weights: Attention matrix of shape
            ``(n_layers, n_heads, seq_len, seq_len)`` or importance scores
            of shape ``(seq_len,)``.
        sequence: Raw DNA string.
        save_path: Path to save figure.
        layer: Transformer layer index to visualise.
        head: Attention head index.  If ``None``, averages over heads.
        max_length: Maximum sequence length to display.
        dpi: Resolution.

    Returns:
        Matplotlib figure or ``None``.
    """
    if not _HAS_PLOTTING:
        return None

    _set_publication_style()

    seq_display = sequence[:max_length]

    if attention_weights.ndim == 1:
        # Importance scores
        scores = attention_weights[:max_length]

        fig, ax = plt.subplots(figsize=FIGURE_SIZES["double_column"])
        colors_map = {"A": "#E74C3C", "C": "#3498DB", "G": "#2ECC71", "T": "#F39C12"}

        bar_colors = [colors_map.get(nt, COLORS["dark_grey"]) for nt in seq_display]
        ax.bar(
            range(len(scores)), scores,
            color=bar_colors, edgecolor="none", width=1.0,
        )
        ax.set_xlabel("Position", fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel("Importance Score", fontsize=FONT_SIZES["axis_label"])
        ax.set_title(
            "Sequence Importance Scores",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
        )

        # Legend for nucleotides
        handles = [
            mpatches.Patch(color=c, label=nt)
            for nt, c in colors_map.items()
        ]
        ax.legend(handles=handles, fontsize=FONT_SIZES["legend"], frameon=False,
                  ncol=4, loc="upper right")
    else:
        # Attention heatmap
        attn = attention_weights[layer]
        if head is not None:
            attn_map = attn[head, :max_length, :max_length]
            title_suffix = f"Layer {layer}, Head {head}"
        else:
            attn_map = attn[:, :max_length, :max_length].mean(axis=0)
            title_suffix = f"Layer {layer}, Averaged"

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(attn_map, cmap="Blues", aspect="auto", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention Weight")

        ax.set_xlabel("Key Position", fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel("Query Position", fontsize=FONT_SIZES["axis_label"])
        ax.set_title(
            f"Attention Map ({title_suffix})",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
        )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Attention figure saved to {}", save_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 4: UMAP embedding space
# ---------------------------------------------------------------------------


def create_embedding_figure(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
    method: str = "umap",
    dpi: int = DEFAULT_DPI,
) -> Optional[Any]:
    """Visualise sequence embeddings in 2D via UMAP or t-SNE.

    Args:
        embeddings: High-dimensional embeddings, shape ``(N, D)``.
        labels: Optional integer labels for colouring, shape ``(N,)``.
        label_names: Mapping from label int to display name.
        save_path: Path to save figure.
        method: Dimensionality reduction method: ``"umap"`` or ``"tsne"``.
        dpi: Resolution.

    Returns:
        Matplotlib figure or ``None``.
    """
    if not _HAS_PLOTTING:
        return None

    _set_publication_style()

    # Dimensionality reduction
    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
            coords = reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("umap-learn not installed; falling back to PCA.")
            coords = _pca_2d(embeddings)
    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE

            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            coords = reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("scikit-learn not installed; falling back to PCA.")
            coords = _pca_2d(embeddings)
    else:
        coords = _pca_2d(embeddings)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_column"])

    if labels is not None:
        unique_labels = np.unique(labels)
        palette = sns.color_palette("Set2", n_colors=len(unique_labels))
        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            name = label_names.get(int(lab), str(lab)) if label_names else str(lab)
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[palette[i]], s=8, alpha=0.6, label=name,
                edgecolors="none",
            )
        ax.legend(
            fontsize=FONT_SIZES["legend"], frameon=False,
            markerscale=2, loc="best",
        )
    else:
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=COLORS["accent2"], s=8, alpha=0.5, edgecolors="none",
        )

    method_label = method.upper() if method != "pca" else "PCA"
    ax.set_xlabel(f"{method_label} 1", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel(f"{method_label} 2", fontsize=FONT_SIZES["axis_label"])
    ax.set_title(
        "Sequence Embedding Space",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Embedding figure saved to {}", save_path)

    return fig


def _pca_2d(data: np.ndarray) -> np.ndarray:
    """Simple PCA to 2D (no external dependency).

    Args:
        data: Shape ``(N, D)``.

    Returns:
        Shape ``(N, 2)`` projected coordinates.
    """
    data = np.asarray(data, dtype=np.float64)
    centered = data - data.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Top 2 eigenvectors (largest eigenvalues)
    idx = np.argsort(eigenvalues)[::-1][:2]
    components = eigenvectors[:, idx]
    return centered @ components


# ---------------------------------------------------------------------------
# Figure 5: Variant effect landscape
# ---------------------------------------------------------------------------


def create_variant_landscape_figure(
    positions: np.ndarray,
    effect_scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    gene_name: str = "",
    save_path: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
) -> Optional[Any]:
    """Visualise variant effect scores along a genomic region.

    Args:
        positions: Genomic positions, shape ``(N,)``.
        effect_scores: Predicted effect scores, shape ``(N,)``.
        labels: Optional pathogenicity labels (0/1), shape ``(N,)``.
        gene_name: Gene or region name for title.
        save_path: Path to save figure.
        dpi: Resolution.

    Returns:
        Matplotlib figure or ``None``.
    """
    if not _HAS_PLOTTING:
        return None

    _set_publication_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["double_column"])

    if labels is not None:
        benign_mask = labels == 0
        patho_mask = labels == 1

        ax.scatter(
            positions[benign_mask],
            effect_scores[benign_mask],
            c=COLORS["accent2"],
            s=15, alpha=0.6, label="Benign",
            edgecolors="none",
        )
        ax.scatter(
            positions[patho_mask],
            effect_scores[patho_mask],
            c=COLORS["accent1"],
            s=15, alpha=0.6, label="Pathogenic",
            edgecolors="none",
        )
        ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)
    else:
        ax.scatter(
            positions, effect_scores,
            c=effect_scores, cmap="RdYlBu_r",
            s=15, alpha=0.6, edgecolors="none",
        )

    ax.set_xlabel("Genomic Position", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Predicted Effect Score", fontsize=FONT_SIZES["axis_label"])

    title = "Variant Effect Landscape"
    if gene_name:
        title += f" - {gene_name}"
    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold")

    # Add significance threshold line
    ax.axhline(
        y=0.5, color=COLORS["dark_grey"], linestyle="--",
        linewidth=0.8, alpha=0.7, label="Threshold",
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Variant landscape figure saved to {}", save_path)

    return fig


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------


def create_all_figures(
    output_dir: str = "figures",
    benchmark_results: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    attention_weights: Optional[np.ndarray] = None,
    sequence: Optional[str] = None,
    embeddings: Optional[np.ndarray] = None,
    embedding_labels: Optional[np.ndarray] = None,
    variant_positions: Optional[np.ndarray] = None,
    variant_scores: Optional[np.ndarray] = None,
    variant_labels: Optional[np.ndarray] = None,
    dpi: int = DEFAULT_DPI,
    fmt: str = "pdf",
) -> Dict[str, Optional[Any]]:
    """Generate all publication figures.

    Creates Figures 1-5 for the methods paper.  Components that require
    data will be skipped if the data is not provided.

    Args:
        output_dir: Directory to save figures.
        benchmark_results: Results dict for Figure 2.
        attention_weights: Attention weights for Figure 3.
        sequence: DNA sequence for Figure 3.
        embeddings: Embeddings for Figure 4.
        embedding_labels: Labels for Figure 4.
        variant_positions: Positions for Figure 5.
        variant_scores: Scores for Figure 5.
        variant_labels: Labels for Figure 5.
        dpi: Resolution for all figures.
        fmt: Output format (``"pdf"``, ``"png"``, ``"svg"``).

    Returns:
        Dictionary mapping figure names to figure objects.
    """
    if not _HAS_PLOTTING:
        logger.error("matplotlib required for figure generation.")
        return {}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    figures: Dict[str, Optional[Any]] = {}

    # Figure 1: Architecture
    logger.info("Generating Figure 1: Architecture diagram")
    figures["fig1_architecture"] = create_architecture_figure(
        save_path=str(out / f"fig1_architecture.{fmt}"), dpi=dpi
    )

    # Figure 2: Benchmark comparison
    if benchmark_results:
        logger.info("Generating Figure 2: Benchmark comparison")
        figures["fig2_benchmark"] = create_benchmark_figure(
            benchmark_results,
            save_path=str(out / f"fig2_benchmark.{fmt}"),
            dpi=dpi,
        )
    else:
        logger.info("Skipping Figure 2: no benchmark results provided")

    # Figure 3: Attention / importance
    if attention_weights is not None:
        logger.info("Generating Figure 3: Attention visualisation")
        figures["fig3_attention"] = create_attention_figure(
            attention_weights,
            sequence=sequence or "ACGT" * 25,
            save_path=str(out / f"fig3_attention.{fmt}"),
            dpi=dpi,
        )
    else:
        logger.info("Skipping Figure 3: no attention weights provided")

    # Figure 4: Embedding space
    if embeddings is not None:
        logger.info("Generating Figure 4: Embedding space")
        figures["fig4_embeddings"] = create_embedding_figure(
            embeddings,
            labels=embedding_labels,
            save_path=str(out / f"fig4_embeddings.{fmt}"),
            dpi=dpi,
        )
    else:
        logger.info("Skipping Figure 4: no embeddings provided")

    # Figure 5: Variant landscape
    if variant_positions is not None and variant_scores is not None:
        logger.info("Generating Figure 5: Variant effect landscape")
        figures["fig5_variant_landscape"] = create_variant_landscape_figure(
            variant_positions,
            variant_scores,
            labels=variant_labels,
            save_path=str(out / f"fig5_variant_landscape.{fmt}"),
            dpi=dpi,
        )
    else:
        logger.info("Skipping Figure 5: no variant data provided")

    logger.info("Figure generation complete: {} figures created", len(figures))
    return figures
