"""Tests for API schemas and security modules."""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from genova.api.schemas import (
    VariantType,
    VariantInput,
    EmbeddingRequest,
    HealthResponse,
)
from genova.api.security import _load_api_keys


# ---------------------------------------------------------------------------
# VariantType enum
# ---------------------------------------------------------------------------


class TestVariantType:

    def test_enum_values(self):
        assert VariantType.SNV == "snv"
        assert VariantType.INSERTION == "insertion"
        assert VariantType.DELETION == "deletion"
        assert VariantType.MNV == "mnv"

    def test_from_string(self):
        assert VariantType("snv") == VariantType.SNV


# ---------------------------------------------------------------------------
# VariantInput validation
# ---------------------------------------------------------------------------


class TestVariantInput:

    def test_valid_sequence_accepted(self):
        vi = VariantInput(sequence="ACGTACGT")
        assert vi.sequence == "ACGTACGT"

    def test_valid_lowercase_accepted(self):
        vi = VariantInput(sequence="acgtacgt")
        assert vi.sequence == "acgtacgt"

    def test_valid_with_n(self):
        vi = VariantInput(sequence="ACGTNNN")
        assert vi.sequence == "ACGTNNN"

    def test_invalid_chars_rejected(self):
        with pytest.raises(ValidationError):
            VariantInput(sequence="ACGTXYZ")

    def test_chrom_pos_ref_alt(self):
        vi = VariantInput(chrom="chr1", pos=100, ref="A", alt="G")
        assert vi.chrom == "chr1"
        assert vi.pos == 100
        assert vi.ref == "A"
        assert vi.alt == "G"

    def test_pos_must_be_positive(self):
        with pytest.raises(ValidationError):
            VariantInput(chrom="chr1", pos=0, ref="A", alt="G")

    def test_empty_sequence_rejected(self):
        with pytest.raises(ValidationError):
            VariantInput(sequence="")

    def test_none_fields_by_default(self):
        vi = VariantInput()
        assert vi.sequence is None
        assert vi.chrom is None


# ---------------------------------------------------------------------------
# EmbeddingRequest
# ---------------------------------------------------------------------------


class TestEmbeddingRequest:

    def test_creation(self):
        req = EmbeddingRequest(sequences=["ACGTACGT"])
        assert req.sequences == ["ACGTACGT"]

    def test_pooling_default(self):
        req = EmbeddingRequest(sequences=["ACGT"])
        assert req.pooling == "mean"

    def test_multiple_sequences(self):
        req = EmbeddingRequest(sequences=["ACGT", "TGCA"])
        assert len(req.sequences) == 2


# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------


class TestHealthResponse:

    def test_creation(self):
        resp = HealthResponse(status="ok", version="0.1.0")
        assert resp.status == "ok"
        assert resp.version == "0.1.0"

    def test_defaults(self):
        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.model_loaded is False


# ---------------------------------------------------------------------------
# Security: _load_api_keys
# ---------------------------------------------------------------------------


class TestLoadApiKeys:

    def test_from_env_var(self, monkeypatch):
        monkeypatch.setenv("GENOVA_API_KEYS", "key1,key2,key3")
        monkeypatch.delenv("GENOVA_API_KEYS_FILE", raising=False)
        keys = _load_api_keys()
        assert keys == {"key1", "key2", "key3"}

    def test_empty_env_returns_empty(self, monkeypatch):
        monkeypatch.setenv("GENOVA_API_KEYS", "")
        monkeypatch.delenv("GENOVA_API_KEYS_FILE", raising=False)
        keys = _load_api_keys()
        assert keys == set()

    def test_from_file(self, monkeypatch, tmp_path):
        keys_file = tmp_path / "keys.txt"
        keys_file.write_text("file_key1\nfile_key2\n")
        monkeypatch.setenv("GENOVA_API_KEYS", "")
        monkeypatch.setenv("GENOVA_API_KEYS_FILE", str(keys_file))
        keys = _load_api_keys()
        assert "file_key1" in keys
        assert "file_key2" in keys

    def test_combined_env_and_file(self, monkeypatch, tmp_path):
        keys_file = tmp_path / "keys.txt"
        keys_file.write_text("file_key\n")
        monkeypatch.setenv("GENOVA_API_KEYS", "env_key")
        monkeypatch.setenv("GENOVA_API_KEYS_FILE", str(keys_file))
        keys = _load_api_keys()
        assert "env_key" in keys
        assert "file_key" in keys

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("GENOVA_API_KEYS", "  key1  ,  key2  ")
        monkeypatch.delenv("GENOVA_API_KEYS_FILE", raising=False)
        keys = _load_api_keys()
        assert keys == {"key1", "key2"}

    def test_no_env_vars_set(self, monkeypatch):
        monkeypatch.delenv("GENOVA_API_KEYS", raising=False)
        monkeypatch.delenv("GENOVA_API_KEYS_FILE", raising=False)
        keys = _load_api_keys()
        assert keys == set()
