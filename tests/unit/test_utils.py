"""Tests for genova.utils — reproducibility, cache, and device modules."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from genova.utils.reproducibility import set_seed, enable_deterministic_mode
from genova.utils.cache import (
    MemoryCache,
    DiskCache,
    _cache_key,
    _serialize_embedding,
    _deserialize_embedding,
)
from genova.utils.device import get_device, DeviceManager


# =========================================================================
# reproducibility.py
# =========================================================================


class TestSetSeed:
    """Tests for set_seed()."""

    def test_python_random_deterministic(self):
        set_seed(123)
        a = [random.random() for _ in range(10)]
        set_seed(123)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_numpy_random_deterministic(self):
        set_seed(99)
        a = np.random.rand(5).tolist()
        set_seed(99)
        b = np.random.rand(5).tolist()
        assert a == b

    def test_torch_random_deterministic(self):
        set_seed(7)
        a = torch.randn(4, 4)
        set_seed(7)
        b = torch.randn(4, 4)
        assert torch.equal(a, b)

    def test_different_seeds_produce_different_output(self):
        set_seed(0)
        a = torch.randn(8)
        set_seed(999)
        b = torch.randn(8)
        assert not torch.equal(a, b)

    def test_sets_pythonhashseed_env_var(self):
        set_seed(42)
        assert os.environ["PYTHONHASHSEED"] == "42"

    @pytest.mark.parametrize("seed", [0, 1, 2**31 - 1])
    def test_various_seed_values(self, seed: int):
        set_seed(seed)
        assert os.environ["PYTHONHASHSEED"] == str(seed)


class TestEnableDeterministicMode:
    """Tests for enable_deterministic_mode."""

    def test_enable_does_not_crash(self):
        enable_deterministic_mode(warn=True)
        assert torch.are_deterministic_algorithms_enabled()
        # Clean up
        torch.use_deterministic_algorithms(False)

    def test_enable_warn_false(self):
        enable_deterministic_mode(warn=False)
        assert torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)


# =========================================================================
# cache.py — helper functions
# =========================================================================


class TestCacheKey:

    def test_deterministic(self):
        k1 = _cache_key("ACGT", "v1")
        k2 = _cache_key("ACGT", "v1")
        assert k1 == k2

    def test_different_sequences_different_keys(self):
        k1 = _cache_key("ACGT", "v1")
        k2 = _cache_key("TGCA", "v1")
        assert k1 != k2

    def test_different_versions_different_keys(self):
        k1 = _cache_key("ACGT", "v1")
        k2 = _cache_key("ACGT", "v2")
        assert k1 != k2

    def test_key_is_hex_string(self):
        key = _cache_key("ACGT", "v1")
        assert isinstance(key, str)
        assert len(key) == 64
        int(key, 16)  # valid hex

    @pytest.mark.parametrize(
        "seq,ver",
        [("", ""), ("A" * 10_000, "long-version"), ("ACGT\nTGCA", "v1")],
    )
    def test_handles_edge_case_inputs(self, seq: str, ver: str):
        key = _cache_key(seq, ver)
        assert len(key) == 64


class TestSerializeDeserialize:

    def test_roundtrip_1d(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data = _serialize_embedding(arr)
        recovered = _deserialize_embedding(data)
        np.testing.assert_array_equal(arr, recovered)

    def test_roundtrip_2d(self):
        arr = np.random.randn(4, 8).astype(np.float64)
        data = _serialize_embedding(arr)
        recovered = _deserialize_embedding(data)
        np.testing.assert_array_equal(arr, recovered)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_preserves_dtype(self, dtype):
        arr = np.zeros(3, dtype=dtype)
        recovered = _deserialize_embedding(_serialize_embedding(arr))
        assert recovered.dtype == dtype

    def test_serialized_is_bytes(self):
        arr = np.array([0.0])
        data = _serialize_embedding(arr)
        assert isinstance(data, bytes)
        assert len(data) > 0


# =========================================================================
# cache.py — MemoryCache
# =========================================================================


class TestMemoryCache:

    @staticmethod
    def _make_emb(dim: int = 4, val: float = 1.0) -> np.ndarray:
        return np.full(dim, val, dtype=np.float32)

    def test_put_and_get(self):
        cache = MemoryCache(max_size=10, model_version="test")
        emb = self._make_emb(val=3.14)
        cache.put("ACGT", emb)
        result = cache.get("ACGT")
        assert result is not None
        np.testing.assert_array_almost_equal(result, emb)

    def test_get_miss_returns_none(self):
        cache = MemoryCache(max_size=10, model_version="test")
        assert cache.get("MISSING") is None

    def test_stats_initial(self):
        cache = MemoryCache(max_size=5, model_version="v1")
        s = cache.stats()
        assert s["hits"] == 0
        assert s["misses"] == 0
        assert s["size"] == 0

    def test_stats_after_operations(self):
        cache = MemoryCache(max_size=10, model_version="v1")
        cache.put("A", self._make_emb())
        cache.get("A")  # hit
        cache.get("B")  # miss
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["size"] == 1

    def test_lru_eviction(self):
        cache = MemoryCache(max_size=2, model_version="test")
        cache.put("A", self._make_emb(val=1.0))
        cache.put("B", self._make_emb(val=2.0))
        cache.put("C", self._make_emb(val=3.0))  # Should evict "A"
        assert cache.get("A") is None
        assert cache.get("B") is not None
        assert cache.get("C") is not None

    def test_overwrite_existing_key(self):
        cache = MemoryCache(max_size=10, model_version="test")
        cache.put("SEQ", self._make_emb(val=1.0))
        cache.put("SEQ", self._make_emb(val=2.0))
        result = cache.get("SEQ")
        np.testing.assert_array_almost_equal(result, self._make_emb(val=2.0))
        assert cache.stats()["size"] == 1

    def test_clear(self):
        cache = MemoryCache(max_size=10, model_version="test")
        cache.put("A", self._make_emb())
        cache.clear()
        assert cache.stats()["size"] == 0
        assert cache.get("A") is None


# =========================================================================
# cache.py — DiskCache
# =========================================================================


class TestDiskCache:

    @staticmethod
    def _make_emb(dim: int = 4, val: float = 1.0) -> np.ndarray:
        return np.full(dim, val, dtype=np.float32)

    def test_put_and_get(self, tmp_path):
        db_path = tmp_path / "test.db"
        cache = DiskCache(db_path=db_path, model_version="test")
        emb = self._make_emb(val=2.5)
        cache.put("ACGT", emb)
        result = cache.get("ACGT")
        assert result is not None
        np.testing.assert_array_almost_equal(result, emb)
        cache.close()

    def test_get_miss_returns_none(self, tmp_path):
        db_path = tmp_path / "test.db"
        cache = DiskCache(db_path=db_path, model_version="test")
        assert cache.get("MISSING") is None
        cache.close()

    def test_stats(self, tmp_path):
        db_path = tmp_path / "test.db"
        cache = DiskCache(db_path=db_path, model_version="v2")
        cache.put("A", self._make_emb())
        cache.get("A")
        cache.get("B")
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["size"] == 1
        cache.close()

    def test_clear(self, tmp_path):
        db_path = tmp_path / "test.db"
        cache = DiskCache(db_path=db_path, model_version="test")
        cache.put("A", self._make_emb())
        cache.clear()
        assert cache.stats()["size"] == 0
        assert cache.get("A") is None
        cache.close()

    def test_preserves_dtype_and_shape(self, tmp_path):
        db_path = tmp_path / "test.db"
        cache = DiskCache(db_path=db_path, model_version="test")
        emb = np.random.randn(3, 8).astype(np.float64)
        cache.put("SEQ", emb)
        result = cache.get("SEQ")
        assert result.shape == emb.shape
        assert result.dtype == emb.dtype
        np.testing.assert_array_equal(result, emb)
        cache.close()


# =========================================================================
# device.py
# =========================================================================


class TestGetDevice:

    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_returns_known_device_type(self):
        device = get_device()
        assert device.type in ("cpu", "cuda", "mps")

    def test_explicit_device_id(self):
        device = get_device(device_id=0)
        assert device == torch.device("cuda:0")


class TestDeviceManager:

    def test_default_construction(self):
        dm = DeviceManager()
        assert isinstance(dm.device, torch.device)

    def test_mixed_precision_no(self):
        dm = DeviceManager(mixed_precision="no")
        assert dm.dtype == torch.float32
        assert dm.grad_scaler is None
