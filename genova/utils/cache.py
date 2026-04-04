"""Embedding cache for Genova inference.

Provides thread-safe in-memory (LRU) and disk-backed (SQLite) caches for
sequence embeddings, keyed by the hash of (sequence, model_version).

Example::

    from genova.utils.cache import MemoryCache, DiskCache

    cache = MemoryCache(max_size=10_000, model_version="v1.0")
    cache.put("ACGTACGT", embedding_array)
    hit = cache.get("ACGTACGT")  # np.ndarray or None
    print(cache.stats())
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Cache key generation
# ---------------------------------------------------------------------------


def _cache_key(sequence: str, model_version: str) -> str:
    """Generate a deterministic cache key from (sequence, model_version).

    Uses SHA-256 to produce a fixed-length, collision-resistant key.

    Args:
        sequence: The DNA/RNA sequence string.
        model_version: Model version identifier.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    payload = f"{sequence}||{model_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize a numpy array to bytes."""
    buf = io.BytesIO()
    np.save(buf, embedding, allow_pickle=False)
    return buf.getvalue()


def _deserialize_embedding(data: bytes) -> np.ndarray:
    """Deserialize bytes back to a numpy array."""
    buf = io.BytesIO(data)
    return np.load(buf, allow_pickle=False)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class EmbeddingCache(ABC):
    """Abstract base class for embedding caches.

    All cache implementations are thread-safe and support concurrent reads
    and writes from multiple API request handlers.

    Args:
        model_version: Version string used as part of the cache key so that
            embeddings from different model versions do not collide.
    """

    def __init__(self, model_version: str = "default") -> None:
        self.model_version = model_version
        self._hits: int = 0
        self._misses: int = 0
        self._lock = threading.Lock()

    @abstractmethod
    def get(self, sequence: str) -> Optional[np.ndarray]:
        """Retrieve a cached embedding.

        Args:
            sequence: The DNA/RNA sequence string.

        Returns:
            The cached embedding as a numpy array, or ``None`` on miss.
        """
        ...

    @abstractmethod
    def put(self, sequence: str, embedding: np.ndarray) -> None:
        """Store an embedding in the cache.

        Args:
            sequence: The DNA/RNA sequence string.
            embedding: The embedding array to cache.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from the cache."""
        ...

    @abstractmethod
    def _size(self) -> int:
        """Return the number of entries in the cache (internal, no lock)."""
        ...

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dict with ``hits``, ``misses``, ``size``, ``hit_rate``, and
            ``model_version``.
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": self._size(),
                "hit_rate": round(self._hits / max(total, 1), 4),
                "model_version": self.model_version,
            }


# ---------------------------------------------------------------------------
# In-memory LRU cache
# ---------------------------------------------------------------------------


class MemoryCache(EmbeddingCache):
    """Thread-safe in-memory LRU cache for embeddings.

    Uses an :class:`OrderedDict` to maintain access order and evict the
    least-recently-used entry when the cache exceeds ``max_size``.

    Args:
        max_size: Maximum number of entries to store.
        model_version: Model version for cache key derivation.
    """

    def __init__(
        self,
        max_size: int = 10_000,
        model_version: str = "default",
    ) -> None:
        super().__init__(model_version=model_version)
        self.max_size = max_size
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, sequence: str) -> Optional[np.ndarray]:
        """Retrieve an embedding, promoting it to most-recently-used."""
        key = _cache_key(sequence, self.model_version)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._hits += 1
                return self._store[key].copy()
            self._misses += 1
            return None

    def put(self, sequence: str, embedding: np.ndarray) -> None:
        """Store an embedding, evicting LRU entries if necessary."""
        key = _cache_key(sequence, self.model_version)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = embedding.copy()
            else:
                self._store[key] = embedding.copy()
                if len(self._store) > self.max_size:
                    self._store.popitem(last=False)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def _size(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Disk-backed SQLite cache
# ---------------------------------------------------------------------------


class DiskCache(EmbeddingCache):
    """Thread-safe disk-backed embedding cache using SQLite.

    Embeddings are serialized to numpy binary format and stored in an SQLite
    database.  The cache persists across process restarts.

    Args:
        db_path: Path to the SQLite database file.  Created if it does not
            exist.  Use ``":memory:"`` for a non-persistent SQLite store.
        max_size: Maximum number of entries.  When exceeded, the oldest
            (by insertion time) entries are evicted.  ``0`` means unlimited.
        model_version: Model version for cache key derivation.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = ".cache/genova/embeddings.db",
        max_size: int = 0,
        model_version: str = "default",
    ) -> None:
        super().__init__(model_version=model_version)
        self.db_path = str(db_path)
        self.max_size = max_size

        # Ensure parent directory exists
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, timeout=30.0)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_db(self) -> None:
        """Create the embeddings table if it does not exist."""
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                data BLOB NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_created ON embeddings(created_at)"
        )
        conn.commit()

    def get(self, sequence: str) -> Optional[np.ndarray]:
        """Retrieve a cached embedding from disk."""
        key = _cache_key(sequence, self.model_version)
        conn = self._get_conn()
        with self._lock:
            row = conn.execute(
                "SELECT data FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
            if row is not None:
                self._hits += 1
                return _deserialize_embedding(row[0])
            self._misses += 1
            return None

    def put(self, sequence: str, embedding: np.ndarray) -> None:
        """Store an embedding to disk, evicting old entries if max_size is set."""
        key = _cache_key(sequence, self.model_version)
        data = _serialize_embedding(embedding)
        now = time.time()
        conn = self._get_conn()
        with self._lock:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings (key, data, created_at)
                VALUES (?, ?, ?)
                """,
                (key, data, now),
            )
            conn.commit()

            # Evict if over capacity
            if self.max_size > 0:
                count = conn.execute(
                    "SELECT COUNT(*) FROM embeddings"
                ).fetchone()[0]
                if count > self.max_size:
                    excess = count - self.max_size
                    conn.execute(
                        """
                        DELETE FROM embeddings WHERE key IN (
                            SELECT key FROM embeddings
                            ORDER BY created_at ASC LIMIT ?
                        )
                        """,
                        (excess,),
                    )
                    conn.commit()

    def clear(self) -> None:
        """Remove all cached entries from disk."""
        conn = self._get_conn()
        with self._lock:
            conn.execute("DELETE FROM embeddings")
            conn.commit()
            self._hits = 0
            self._misses = 0

    def _size(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the thread-local database connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
