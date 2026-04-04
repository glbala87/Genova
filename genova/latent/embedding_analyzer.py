"""Latent representation analysis for genomic embeddings.

Provides tools to extract embeddings from a trained Genova model, perform
dimensionality reduction (UMAP), cluster the latent space, and annotate
clusters with biological function labels.

Example::

    from genova.latent.embedding_analyzer import EmbeddingAnalyzer

    analyzer = EmbeddingAnalyzer(model, device="cuda")
    embeddings = analyzer.extract_embeddings(dataloader)
    reduced = analyzer.reduce_dimensions(embeddings, method="umap")
    labels = analyzer.cluster(reduced, method="kmeans", n_clusters=10)
    annotations = analyzer.annotate_clusters(labels, region_metadata)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Region metadata
# ---------------------------------------------------------------------------


@dataclass
class RegionMetadata:
    """Metadata for a single genomic region.

    Attributes:
        chrom: Chromosome name.
        start: 0-based start coordinate.
        end: 0-based end coordinate (exclusive).
        annotation: Biological annotation (e.g. ``"promoter"``,
            ``"enhancer"``, ``"intergenic"``, ``"repeat"``).
        extra: Arbitrary additional fields.
    """

    chrom: str = ""
    start: int = 0
    end: int = 0
    annotation: str = "unknown"
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Embedding analyzer
# ---------------------------------------------------------------------------


class EmbeddingAnalyzer:
    """Extract, reduce, cluster, and annotate genomic embeddings.

    Args:
        model: Trained Genova model (any ``nn.Module`` whose forward pass
            returns a dict with ``"last_hidden_state"``).
        device: Device for inference (``"cpu"``, ``"cuda"``, etc.).
        pooling: How to pool per-position hidden states into a region
            vector.  ``"mean"`` or ``"cls"``.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "cpu",
        pooling: str = "mean",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.pooling = pooling

        self.model.to(self.device)
        self.model.eval()

        # state caches
        self._embeddings: Optional[np.ndarray] = None
        self._reduced: Optional[np.ndarray] = None
        self._cluster_labels: Optional[np.ndarray] = None

    # -- extraction ----------------------------------------------------------

    def extract_embeddings(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
        layer_index: int = -1,
    ) -> np.ndarray:
        """Extract pooled embeddings from the model for every batch in *dataloader*.

        Args:
            dataloader: Yields dicts or tuples.  Must contain ``"input_ids"``
                (or be a plain tensor batch).  Optionally ``"attention_mask"``.
            max_samples: Stop after extracting this many samples.
            layer_index: Which transformer layer to extract from.  ``-1``
                uses the last layer (``"last_hidden_state"``).

        Returns:
            ``(N, D)`` numpy array of pooled embeddings.
        """
        all_embeddings: List[np.ndarray] = []
        n_collected = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = self._unpack_batch(batch)
                input_ids = input_ids.to(self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                out = self.model(input_ids, attention_mask=attention_mask)

                if isinstance(out, dict):
                    hidden = out.get("last_hidden_state", out.get("hidden_states"))
                    if hidden is None:
                        # If model returns a tensor directly
                        hidden = out
                elif isinstance(out, Tensor):
                    hidden = out
                else:
                    hidden = out[0] if isinstance(out, (tuple, list)) else out

                # Handle list of layer outputs
                if isinstance(hidden, (list, tuple)):
                    hidden = hidden[layer_index]

                pooled = self._pool(hidden, attention_mask)  # (B, D)
                all_embeddings.append(pooled.cpu().numpy())

                n_collected += pooled.size(0)
                if max_samples is not None and n_collected >= max_samples:
                    break

        result = np.concatenate(all_embeddings, axis=0)
        if max_samples is not None:
            result = result[:max_samples]

        self._embeddings = result
        logger.info("Extracted embeddings: shape={}", result.shape)
        return result

    # -- dimensionality reduction -------------------------------------------

    def reduce_dimensions(
        self,
        embeddings: Optional[np.ndarray] = None,
        method: str = "umap",
        n_components: int = 2,
        **kwargs: Any,
    ) -> np.ndarray:
        """Reduce embedding dimensionality with UMAP or PCA.

        Args:
            embeddings: ``(N, D)`` array.  Uses cached embeddings if ``None``.
            method: ``"umap"`` or ``"pca"``.
            n_components: Target dimensionality (2 or 3 for visualisation).
            **kwargs: Forwarded to the reducer constructor (e.g.
                ``n_neighbors``, ``min_dist`` for UMAP).

        Returns:
            ``(N, n_components)`` reduced array.
        """
        if embeddings is None:
            if self._embeddings is None:
                raise ValueError("No embeddings available. Call extract_embeddings() first.")
            embeddings = self._embeddings

        if method == "umap":
            import umap

            default_params = {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"}
            default_params.update(kwargs)
            reducer = umap.UMAP(n_components=n_components, **default_params)
            reduced = reducer.fit_transform(embeddings)
        elif method == "pca":
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=n_components, **kwargs)
            reduced = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method '{method}'; use 'umap' or 'pca'")

        self._reduced = reduced
        logger.info("Dimensionality reduction ({}): {} -> {}", method, embeddings.shape, reduced.shape)
        return reduced

    # -- clustering ---------------------------------------------------------

    def cluster(
        self,
        embeddings: Optional[np.ndarray] = None,
        method: str = "kmeans",
        n_clusters: int = 10,
        **kwargs: Any,
    ) -> np.ndarray:
        """Cluster embeddings with K-means or DBSCAN.

        Args:
            embeddings: ``(N, D)`` array (full or reduced).  Uses cached
                reduced embeddings if ``None``.
            method: ``"kmeans"`` or ``"dbscan"``.
            n_clusters: Number of clusters for K-means (ignored by DBSCAN).
            **kwargs: Forwarded to the clustering algorithm.

        Returns:
            ``(N,)`` integer cluster labels.
        """
        if embeddings is None:
            embeddings = self._reduced if self._reduced is not None else self._embeddings
        if embeddings is None:
            raise ValueError("No embeddings available. Extract or provide them first.")

        if method == "kmeans":
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, **kwargs)
            labels = model.fit_predict(embeddings)
        elif method == "dbscan":
            from sklearn.cluster import DBSCAN

            default_params = {"eps": 0.5, "min_samples": 5}
            default_params.update(kwargs)
            model = DBSCAN(**default_params)
            labels = model.fit_predict(embeddings)
        else:
            raise ValueError(f"Unknown method '{method}'; use 'kmeans' or 'dbscan'")

        self._cluster_labels = labels
        n_unique = len(set(labels) - {-1})
        logger.info("Clustering ({}): {} clusters found", method, n_unique)
        return labels

    # -- annotation ---------------------------------------------------------

    def annotate_clusters(
        self,
        cluster_labels: Optional[np.ndarray] = None,
        region_metadata: Optional[Sequence[RegionMetadata]] = None,
        annotation_field: str = "annotation",
    ) -> Dict[int, Dict[str, Any]]:
        """Annotate each cluster with the dominant biological category.

        Args:
            cluster_labels: ``(N,)`` cluster assignments.  Uses cached if ``None``.
            region_metadata: Per-region metadata aligned with the embeddings.
            annotation_field: Which metadata field to use for annotation.

        Returns:
            Dict mapping cluster id to an annotation summary with keys:
                - ``"dominant_type"``: most common annotation in the cluster.
                - ``"type_counts"``: dict of annotation -> count.
                - ``"size"``: number of regions in the cluster.
                - ``"fraction"``: fraction occupied by the dominant type.
        """
        if cluster_labels is None:
            if self._cluster_labels is None:
                raise ValueError("No cluster labels. Call cluster() first.")
            cluster_labels = self._cluster_labels

        if region_metadata is None:
            logger.warning("No region_metadata provided; returning size-only annotations")
            unique_labels = set(cluster_labels) - {-1}
            return {
                int(cid): {
                    "dominant_type": "unknown",
                    "type_counts": {},
                    "size": int((cluster_labels == cid).sum()),
                    "fraction": 0.0,
                }
                for cid in unique_labels
            }

        from collections import Counter

        annotations: Dict[int, Dict[str, Any]] = {}
        unique_labels = set(cluster_labels) - {-1}

        for cid in unique_labels:
            mask = cluster_labels == cid
            indices = np.where(mask)[0]
            types = [
                getattr(region_metadata[i], annotation_field, "unknown")
                for i in indices
                if i < len(region_metadata)
            ]
            counts = Counter(types)
            total = len(types)
            dominant = counts.most_common(1)[0] if counts else ("unknown", 0)

            annotations[int(cid)] = {
                "dominant_type": dominant[0],
                "type_counts": dict(counts),
                "size": total,
                "fraction": dominant[1] / max(total, 1),
            }

        logger.info("Annotated {} clusters", len(annotations))
        return annotations

    # -- silhouette evaluation -----------------------------------------------

    def compute_silhouette_score(
        self,
        embeddings: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the mean silhouette score for the current clustering.

        Args:
            embeddings: ``(N, D)`` array.
            labels: ``(N,)`` cluster labels.

        Returns:
            Mean silhouette coefficient in ``[-1, 1]``.
        """
        from sklearn.metrics import silhouette_score

        if embeddings is None:
            embeddings = self._reduced if self._reduced is not None else self._embeddings
        if labels is None:
            labels = self._cluster_labels

        if embeddings is None or labels is None:
            raise ValueError("Embeddings and labels required.")

        # Filter out noise label (-1) for DBSCAN
        valid = labels >= 0
        if valid.sum() < 2:
            return 0.0
        return float(silhouette_score(embeddings[valid], labels[valid]))

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _unpack_batch(
        batch: Any,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Unpack a dataloader batch into input_ids and optional mask."""
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
        elif isinstance(batch, (list, tuple)):
            input_ids = batch[0]
            attention_mask = batch[1] if len(batch) > 1 else None
        elif isinstance(batch, Tensor):
            input_ids = batch
            attention_mask = None
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        return input_ids, attention_mask

    def _pool(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Pool sequence dimension to get per-region vectors."""
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        # mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return hidden_states.mean(dim=1)
