"""Motif clustering, deduplication, and comparison with known motif databases.

Groups similar candidate motifs, merges redundant ones, and compares
discovered motifs against known databases in JASPAR / MEME format.

Example::

    clusterer = MotifClusterer(motifs)
    clusters = clusterer.cluster(threshold=0.8)
    matches = clusterer.compare_jaspar("JASPAR2024_CORE.meme")
    novel = clusterer.find_novel_motifs(matches)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from loguru import logger

from genova.motif.motif_discovery import (
    Motif,
    NUCLEOTIDE_INDEX,
    pwm_to_consensus,
    sequences_to_pwm,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MotifCluster:
    """A group of similar motifs.

    Attributes:
        cluster_id: Unique integer identifier.
        motifs: Member motifs.
        consensus: Consensus sequence from merged PWM.
        merged_pwm: Combined ``(L, 4)`` PWM of all member sequences.
        representative: The highest-scoring motif.
    """

    cluster_id: int
    motifs: List[Motif] = field(default_factory=list)
    consensus: str = ""
    merged_pwm: Optional[np.ndarray] = None
    representative: Optional[Motif] = None


@dataclass
class JASPARMotif:
    """A motif entry parsed from a JASPAR / MEME file.

    Attributes:
        name: Motif name / identifier.
        accession: Database accession (e.g., ``MA0001.1``).
        pwm: ``(L, 4)`` Position Weight Matrix.
        consensus: Consensus sequence.
    """

    name: str
    accession: str = ""
    pwm: Optional[np.ndarray] = None
    consensus: str = ""


@dataclass
class MotifMatch:
    """Result of comparing a discovered motif to a known database motif.

    Attributes:
        discovered: The candidate motif.
        known: The database motif.
        similarity: Similarity score in ``[0, 1]``.
        offset: Best alignment offset.
        strand: ``"+"`` or ``"-"`` (reverse complement).
    """

    discovered: Motif
    known: JASPARMotif
    similarity: float = 0.0
    offset: int = 0
    strand: str = "+"


# ---------------------------------------------------------------------------
# Distance / similarity functions
# ---------------------------------------------------------------------------


def edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Uses the classic dynamic-programming algorithm with O(min(m,n))
    space.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Integer edit distance.
    """
    if len(a) < len(b):
        return edit_distance(b, a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                curr[j - 1] + 1,       # insertion
                prev[j] + 1,           # deletion
                prev[j - 1] + cost,    # substitution
            )
        prev = curr

    return prev[-1]


def normalised_edit_similarity(a: str, b: str) -> float:
    """Compute normalised similarity from edit distance.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Similarity in ``[0, 1]``. 1 means identical.
    """
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - edit_distance(a, b) / max_len


def pwm_similarity(
    pwm_a: np.ndarray,
    pwm_b: np.ndarray,
) -> Tuple[float, int, str]:
    """Compute the best-alignment Pearson correlation between two PWMs.

    Tries all offsets and both strands, returning the maximum column-wise
    Pearson correlation averaged over aligned columns.

    Args:
        pwm_a: ``(La, 4)`` PWM.
        pwm_b: ``(Lb, 4)`` PWM.

    Returns:
        Tuple of ``(similarity, best_offset, strand)``.
    """
    def _rc_pwm(pwm: np.ndarray) -> np.ndarray:
        """Reverse-complement a PWM (reverse rows, swap A<->T and C<->G)."""
        return pwm[::-1, [3, 2, 1, 0]].copy()

    best_sim = -1.0
    best_offset = 0
    best_strand = "+"

    for strand, p_b in [("+", pwm_b), ("-", _rc_pwm(pwm_b))]:
        la, lb = pwm_a.shape[0], p_b.shape[0]
        min_overlap = max(3, min(la, lb) // 2)

        for offset in range(-(lb - min_overlap), la - min_overlap + 1):
            # Determine overlap region
            start_a = max(0, offset)
            end_a = min(la, offset + lb)
            start_b = max(0, -offset)
            end_b = start_b + (end_a - start_a)

            overlap_len = end_a - start_a
            if overlap_len < min_overlap:
                continue

            slice_a = pwm_a[start_a:end_a].flatten()
            slice_b = p_b[start_b:end_b].flatten()

            # Pearson correlation
            mean_a = slice_a.mean()
            mean_b = slice_b.mean()
            da = slice_a - mean_a
            db = slice_b - mean_b
            denom = np.sqrt(np.sum(da ** 2) * np.sum(db ** 2))
            if denom < 1e-12:
                continue
            corr = float(np.sum(da * db) / denom)

            if corr > best_sim:
                best_sim = corr
                best_offset = offset
                best_strand = strand

    # Map to [0, 1]
    similarity = max(0.0, (best_sim + 1.0) / 2.0)
    return similarity, best_offset, best_strand


# ---------------------------------------------------------------------------
# JASPAR / MEME parser
# ---------------------------------------------------------------------------


def parse_jaspar_file(path: Union[str, Path]) -> List[JASPARMotif]:
    """Parse a JASPAR-format PFM file.

    Supports the classic JASPAR ``.pfm`` format::

        >MA0001.1  AGL3
        A  [ 0  3 79 ...]
        C  [ 0  0  1 ...]
        G  [ 0  0  2 ...]
        T  [97 94 15 ...]

    Also handles tab/space-separated count matrices without brackets.

    Args:
        path: Path to JASPAR PFM file.

    Returns:
        List of parsed :class:`JASPARMotif` objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JASPAR file not found: {path}")

    motifs: List[JASPARMotif] = []
    current_name = ""
    current_accession = ""
    rows: Dict[str, List[float]] = {}

    nuc_order = ["A", "C", "G", "T"]

    with open(path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                # End of a motif block
                if rows and len(rows) == 4:
                    motifs.append(_build_jaspar_motif(current_accession, current_name, rows))
                    rows = {}
                continue

            if line.startswith(">"):
                # Flush previous
                if rows and len(rows) == 4:
                    motifs.append(_build_jaspar_motif(current_accession, current_name, rows))
                    rows = {}
                # Parse header
                parts = line[1:].split(None, 1)
                current_accession = parts[0] if parts else ""
                current_name = parts[1] if len(parts) > 1 else current_accession
                continue

            # Count line
            for nuc in nuc_order:
                if line.upper().startswith(nuc):
                    # Remove brackets and nucleotide prefix
                    cleaned = line[1:].strip().strip("[]").strip()
                    values = [float(v) for v in cleaned.split()]
                    rows[nuc] = values
                    break

    # Flush last
    if rows and len(rows) == 4:
        motifs.append(_build_jaspar_motif(current_accession, current_name, rows))

    logger.info("Parsed {} motifs from JASPAR file {}.", len(motifs), path)
    return motifs


def parse_meme_file(path: Union[str, Path]) -> List[JASPARMotif]:
    """Parse a minimal MEME-format motif file.

    Reads motif blocks delimited by ``MOTIF`` headers and ``letter-probability
    matrix`` sections.

    Args:
        path: Path to MEME file.

    Returns:
        List of parsed :class:`JASPARMotif` objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MEME file not found: {path}")

    motifs: List[JASPARMotif] = []
    current_name = ""
    current_accession = ""
    in_matrix = False
    matrix_rows: List[List[float]] = []

    with open(path) as fh:
        for raw_line in fh:
            line = raw_line.strip()

            if line.startswith("MOTIF"):
                # Flush previous
                if matrix_rows:
                    motifs.append(_build_meme_motif(current_accession, current_name, matrix_rows))
                    matrix_rows = []
                    in_matrix = False

                parts = line.split(None, 2)
                current_accession = parts[1] if len(parts) > 1 else ""
                current_name = parts[2] if len(parts) > 2 else current_accession
                continue

            if "letter-probability matrix" in line.lower():
                in_matrix = True
                matrix_rows = []
                continue

            if in_matrix:
                if not line or line.startswith("URL") or line.startswith("MOTIF"):
                    # End of matrix
                    if matrix_rows:
                        motifs.append(_build_meme_motif(current_accession, current_name, matrix_rows))
                        matrix_rows = []
                    in_matrix = False
                    if line.startswith("MOTIF"):
                        parts = line.split(None, 2)
                        current_accession = parts[1] if len(parts) > 1 else ""
                        current_name = parts[2] if len(parts) > 2 else current_accession
                    continue

                # Parse probability row
                values = line.split()
                try:
                    row = [float(v) for v in values[:4]]
                    if len(row) == 4:
                        matrix_rows.append(row)
                except ValueError:
                    in_matrix = False

    # Flush last
    if matrix_rows:
        motifs.append(_build_meme_motif(current_accession, current_name, matrix_rows))

    logger.info("Parsed {} motifs from MEME file {}.", len(motifs), path)
    return motifs


def _build_jaspar_motif(
    accession: str,
    name: str,
    rows: Dict[str, List[float]],
) -> JASPARMotif:
    """Construct a JASPARMotif from parsed count rows."""
    length = len(rows["A"])
    pwm = np.zeros((length, 4), dtype=np.float64)
    for nuc, idx in NUCLEOTIDE_INDEX.items():
        if nuc in rows:
            pwm[:, idx] = rows[nuc]

    # Normalise to frequencies
    row_sums = pwm.sum(axis=1, keepdims=True)
    pwm = pwm / np.maximum(row_sums, 1e-12)

    consensus = pwm_to_consensus(pwm)
    return JASPARMotif(name=name, accession=accession, pwm=pwm, consensus=consensus)


def _build_meme_motif(
    accession: str,
    name: str,
    matrix_rows: List[List[float]],
) -> JASPARMotif:
    """Construct a JASPARMotif from parsed MEME probability rows."""
    pwm = np.array(matrix_rows, dtype=np.float64)
    # Ensure rows sum to 1
    row_sums = pwm.sum(axis=1, keepdims=True)
    pwm = pwm / np.maximum(row_sums, 1e-12)

    consensus = pwm_to_consensus(pwm)
    return JASPARMotif(name=name, accession=accession, pwm=pwm, consensus=consensus)


# ---------------------------------------------------------------------------
# MotifClusterer
# ---------------------------------------------------------------------------


class MotifClusterer:
    """Cluster, merge, and compare discovered motifs.

    Supports two similarity metrics:

    * **edit** -- Normalised edit-distance similarity on consensus strings.
    * **pwm** -- Pearson-correlation-based PWM alignment similarity.

    Args:
        motifs: Collection of discovered :class:`Motif` objects.
        metric: ``"edit"`` or ``"pwm"``.
    """

    def __init__(
        self,
        motifs: Sequence[Motif],
        *,
        metric: str = "edit",
    ) -> None:
        if metric not in ("edit", "pwm"):
            raise ValueError(f"metric must be 'edit' or 'pwm', got '{metric}'")

        self.motifs = list(motifs)
        self.metric = metric
        self.clusters: List[MotifCluster] = []

        logger.info(
            "MotifClusterer initialised with {} motifs (metric={}).",
            len(self.motifs),
            metric,
        )

    # ------------------------------------------------------------------
    # Pairwise similarity
    # ------------------------------------------------------------------

    def _compute_similarity(self, a: Motif, b: Motif) -> float:
        """Compute similarity between two motifs using the configured metric."""
        if self.metric == "pwm" and a.pwm is not None and b.pwm is not None:
            sim, _, _ = pwm_similarity(a.pwm, b.pwm)
            return sim
        return normalised_edit_similarity(a.sequence, b.sequence)

    def _compute_similarity_matrix(self) -> np.ndarray:
        """Build the full pairwise similarity matrix."""
        n = len(self.motifs)
        sim = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            sim[i, i] = 1.0
            for j in range(i + 1, n):
                s = self._compute_similarity(self.motifs[i], self.motifs[j])
                sim[i, j] = s
                sim[j, i] = s
        return sim

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster(
        self,
        threshold: float = 0.8,
    ) -> List[MotifCluster]:
        """Cluster motifs using single-linkage agglomerative clustering.

        Merges motif pairs whose similarity exceeds *threshold* using a
        greedy union-find approach (efficient for moderate motif counts).

        Args:
            threshold: Minimum similarity to merge two motifs into the
                same cluster.  Value in ``[0, 1]``.

        Returns:
            List of :class:`MotifCluster` objects.
        """
        n = len(self.motifs)
        if n == 0:
            self.clusters = []
            return self.clusters

        sim_matrix = self._compute_similarity_matrix()

        # Union-Find
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= threshold:
                    union(i, j)

        # Group by root
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        self.clusters = []
        for cid, (_, indices) in enumerate(sorted(groups.items())):
            cluster_motifs = [self.motifs[i] for i in indices]
            representative = max(cluster_motifs, key=lambda m: m.score)

            # Build merged PWM from all member sequences
            seqs = [m.sequence for m in cluster_motifs]
            target_len = len(representative.sequence)
            aligned = [s for s in seqs if len(s) == target_len]

            merged_pwm = None
            consensus = representative.sequence
            if aligned:
                try:
                    merged_pwm = sequences_to_pwm(aligned)
                    consensus = pwm_to_consensus(merged_pwm)
                except ValueError:
                    pass

            self.clusters.append(
                MotifCluster(
                    cluster_id=cid,
                    motifs=cluster_motifs,
                    consensus=consensus,
                    merged_pwm=merged_pwm,
                    representative=representative,
                )
            )

        logger.info(
            "Clustered {} motifs into {} clusters (threshold={}).",
            n,
            len(self.clusters),
            threshold,
        )
        return self.clusters

    # ------------------------------------------------------------------
    # Database comparison
    # ------------------------------------------------------------------

    def compare_jaspar(
        self,
        database_path: Union[str, Path],
        *,
        format: str = "auto",
        top_k: int = 5,
        min_similarity: float = 0.6,
    ) -> List[MotifMatch]:
        """Compare discovered motifs against a JASPAR / MEME database.

        Args:
            database_path: Path to JASPAR ``.pfm`` or MEME ``.meme`` file.
            format: ``"jaspar"``, ``"meme"``, or ``"auto"`` (infer from
                extension).
            top_k: Maximum matches per discovered motif.
            min_similarity: Minimum similarity to report a match.

        Returns:
            List of :class:`MotifMatch` objects sorted by similarity.
        """
        db_path = Path(database_path)

        # Auto-detect format
        if format == "auto":
            ext = db_path.suffix.lower()
            if ext in (".meme", ".txt"):
                format = "meme"
            else:
                format = "jaspar"

        if format == "meme":
            known_motifs = parse_meme_file(db_path)
        else:
            known_motifs = parse_jaspar_file(db_path)

        if not known_motifs:
            logger.warning("No motifs parsed from {}.", db_path)
            return []

        # Use cluster representatives if clustered, otherwise raw motifs
        query_motifs = self.motifs
        if self.clusters:
            query_motifs = [
                c.representative for c in self.clusters if c.representative is not None
            ]

        matches: List[MotifMatch] = []

        for qm in query_motifs:
            if qm.pwm is None:
                # Build a quick PWM from single sequence
                try:
                    qm.pwm = sequences_to_pwm([qm.sequence])
                except ValueError:
                    continue

            best_matches: List[Tuple[float, int, str, JASPARMotif]] = []
            for km in known_motifs:
                if km.pwm is None:
                    continue
                sim, offset, strand = pwm_similarity(qm.pwm, km.pwm)
                if sim >= min_similarity:
                    best_matches.append((sim, offset, strand, km))

            # Sort and take top_k
            best_matches.sort(key=lambda x: x[0], reverse=True)
            for sim, offset, strand, km in best_matches[:top_k]:
                matches.append(
                    MotifMatch(
                        discovered=qm,
                        known=km,
                        similarity=sim,
                        offset=offset,
                        strand=strand,
                    )
                )

        matches.sort(key=lambda m: m.similarity, reverse=True)
        logger.info(
            "Found {} matches (>={:.2f} similarity) against {} known motifs.",
            len(matches),
            min_similarity,
            len(known_motifs),
        )
        return matches

    # ------------------------------------------------------------------
    # Novel motif identification
    # ------------------------------------------------------------------

    def find_novel_motifs(
        self,
        matches: Sequence[MotifMatch],
        *,
        novelty_threshold: float = 0.7,
    ) -> List[Motif]:
        """Identify discovered motifs that do not match any known motif.

        A motif is considered novel if its best similarity to any known
        motif is below *novelty_threshold*.

        Args:
            matches: Matches from :meth:`compare_jaspar`.
            novelty_threshold: Maximum similarity for a motif to be
                considered novel.

        Returns:
            List of novel :class:`Motif` objects.
        """
        # Build map: discovered motif sequence -> best similarity
        best_sim: Dict[str, float] = {}
        for m in matches:
            key = m.discovered.sequence
            if key not in best_sim or m.similarity > best_sim[key]:
                best_sim[key] = m.similarity

        # Collect motifs with best match below threshold
        query_motifs = self.motifs
        if self.clusters:
            query_motifs = [
                c.representative for c in self.clusters if c.representative is not None
            ]

        novel: List[Motif] = []
        for qm in query_motifs:
            sim = best_sim.get(qm.sequence, 0.0)
            if sim < novelty_threshold:
                novel.append(qm)

        logger.info(
            "Identified {} novel motifs (best similarity < {}).",
            len(novel),
            novelty_threshold,
        )
        return novel
